import hydra
import os
import torch
import numpy as np
import time
import math
from collections import deque
from omegaconf import OmegaConf

from hydra.utils import to_absolute_path

from torchrl.data.tensor_specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BoundedTensorSpec,
    TensorSpec,
)

import torch.nn as nn
from einops.layers.torch import Rearrange
from resources.learning.modules.rnn import GRU
from resources.learning.ppo.ppo import PPOConfig, make_mlp, Actor, IndependentNormal
from resources.utils.safety_shield import apply_target_acc_safety_shield
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule, TensorDictModuleBase
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry

from resources.NeuFlow_v2.infer_lidar import init_neuflow
from resources.utils.torch import (
    euler_to_quaternion,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_euler,
)



class BatchConv2dWrapper(nn.Module):
    """Make a Conv2d-based CNN accept leading batch dims like [E,T,C,H,W].

    In infer we typically have [N,C,H,W], but training uses this wrapper.
    Keeping it here ensures state_dict keys match training checkpoints.
    """

    def __init__(self, cnn: nn.Module):
        super().__init__()
        self.cnn = cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 4:
            raise RuntimeError(f"[BatchConv2dWrapper] Expected >=4D, got {tuple(x.shape)}")
        leading = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        y = self.cnn(x)
        return y.reshape(*leading, -1)

def compute_rayhitsdir(device, num_envs, h_fov, v_fov, h_num, v_num):
    if h_fov == 360:
        horizontal_angles = torch.linspace(0, h_fov, h_num + 1, device=device)
        horizontal_angles = horizontal_angles[:h_num]
    else:
        horizontal_angles = torch.linspace(0, h_fov, h_num, device=device)
    vertical_angles = torch.linspace(v_fov[0], v_fov[1], v_num, device=device) 
    horizontal_radians = horizontal_angles * torch.pi/180 
    vertical_radians = vertical_angles * torch.pi/180
    horizontal_grid, vertical_grid = torch.meshgrid(horizontal_radians, vertical_radians)
    directions = torch.stack((
        torch.cos(vertical_grid) * torch.cos(horizontal_grid),
        torch.cos(vertical_grid) * torch.sin(horizontal_grid),
        torch.sin(vertical_grid) 
    ), dim=-1) 
    ray_hits_dir = directions.reshape(h_num * v_num, -1).unsqueeze(0).expand(num_envs, -1, -1)

    return ray_hits_dir

def _resolve_device():
    env_device = os.getenv("SIM_DEVICE")
    if env_device:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOConfig,
        model_cfg,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.device = device
        self.action_dim = 3
        temporal_cfg = getattr(model_cfg, "temporal", None) if model_cfg is not None else None
        self.temporal_enable = bool(getattr(temporal_cfg, "enable", False))
        self.temporal_type = str(getattr(temporal_cfg, "type", "gru")).lower()
        self.temporal_hidden_size = int(getattr(temporal_cfg, "hidden_size", 128))
        if self.temporal_enable and self.temporal_type != "gru":
            raise NotImplementedError(f"Unsupported temporal core: {self.temporal_type}")
        if self.temporal_enable and self.temporal_hidden_size != 128:
            raise ValueError("Infer temporal core currently requires hidden_size=128.")
        self._rollout_hidden = None

        self.cnn = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]),
            nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
        )
        self.feature_mlp = make_mlp([256, 256])

        self.encoder = TensorDictSequential(
            TensorDictModule(BatchConv2dWrapper(self.cnn), [("agents", "observation", "lidar")], ["_cnn_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
            TensorDictModule(self.feature_mlp, ["_feature"], ["_feature"]),
        ).to(self.device)
        self.temporal_core = GRU(128, self.temporal_hidden_size).to(self.device) if self.temporal_enable else None

        self.actor = ProbabilisticActor(
            TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=False,
        ).to(self.device)

        fake_input = observation_spec.zero()
        self._encode_features(fake_input)
        self.actor(fake_input)
        self.reset_rollout_state()

    def reset_rollout_state(self):
        self._rollout_hidden = None

    def _run_cnn(self, lidar: torch.Tensor) -> torch.Tensor:
        leading = lidar.shape[:-3]
        x = lidar.reshape(-1, *lidar.shape[-3:])
        y = self.cnn(x)
        return y.reshape(*leading, -1)

    def _apply_last_dim(self, module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        leading = tensor.shape[:-1]
        y = module(tensor.reshape(-1, tensor.shape[-1]))
        return y.reshape(*leading, -1)

    def _encode_features(self, tensordict: TensorDict):
        lidar = tensordict[("agents", "observation", "lidar")]
        state = tensordict[("agents", "observation", "state")]
        cnn_feature = self._run_cnn(lidar)
        if self.temporal_enable and self.temporal_core is not None:
            batch = int(cnn_feature.shape[0])
            if self._rollout_hidden is None or self._rollout_hidden.shape[0] != batch:
                self._rollout_hidden = torch.zeros(batch, self.temporal_hidden_size, device=cnn_feature.device, dtype=cnn_feature.dtype)
            cnn_feature_mem, next_hidden = self.temporal_core(cnn_feature, h=self._rollout_hidden)
            self._rollout_hidden = next_hidden.detach()
        else:
            cnn_feature_mem = cnn_feature
        feature = self._apply_last_dim(self.feature_mlp, torch.cat([cnn_feature_mem, state], dim=-1))
        tensordict.set("_cnn_feature", cnn_feature)
        tensordict.set("_cnn_feature_mem", cnn_feature_mem)
        tensordict.set("_feature", feature)
        return tensordict

    def forward(self, tensordict: TensorDict):
        self._encode_features(tensordict)
        self.actor(tensordict)
        tensordict.exclude("loc", "scale", "_feature", "_cnn_feature", "_cnn_feature_mem", inplace=True)
        return tensordict

    @torch.no_grad()
    def act_mean(self, tensordict: TensorDict) -> torch.Tensor:
        self._encode_features(tensordict)
        dist = self.actor.get_dist(tensordict)
        return dist.mean


class Infer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = _resolve_device()
        
        # --- ROS sim state (we publish /sim/odom for the lidar package)
        self.odom = Odometry()

        # altitude reference for goals (keep goal z constant in current setup)
        self.zpose = float(self.cfg.task.get("zpose", 2.0))

        # target goal (world frame)
        default_goal = self.cfg.task.get("default_goal_xyz", None)
        if default_goal is None:
            default_goal = [0.0, 15.0, self.zpose]
        if len(default_goal) == 2:
            default_goal = [default_goal[0], default_goal[1], self.zpose]
        self.target_pos_np = list(default_goal)
        self.target_pos = None

        # start state (world frame)
        start_xyz = self.cfg.task.get("start_xyz", None)
        if start_xyz is None:
            start_xyz = [0.0, -15.0, self.zpose]
        if len(start_xyz) == 2:
            start_xyz = [start_xyz[0], start_xyz[1], self.zpose]
        self.posx = float(start_xyz[0])
        self.posy = float(start_xyz[1])
        self.posz = float(start_xyz[2])

        self.velox = 0.0
        self.veloy = 0.0
        self.veloz = 0.0

        # simulation time step used for kinematic integration
        self.sim_dt = float(self.cfg.task.get("sim_dt", 0.02))

        # "virtual" vertical constraints (match training env defaults)
        self.virtual_ground = float(self.cfg.task.get("virtual_ground", 0.5))
        self.virtual_ceiling = float(self.cfg.task.get("virtual_ceiling", 3.5))

        # XY boundary (virtual walls) - keeps inference inside map bounds
        self.enable_xy_bound = bool(self.cfg.task.get("enable_xy_bound", True))
        self.bound_margin = float(self.cfg.task.get("bound_margin", 0.3))
        self.bound_restitution = float(self.cfg.task.get("bound_restitution", 0.0))
        self.map_x_wall_extra = float(self.cfg.task.get("map_x_wall_extra", 3.0))
        self.map_y_wall_extra = float(self.cfg.task.get("map_y_wall_extra", 10.0))
        self.x_bound = None
        self.y_bound = None

        # collision handling (virtual stop when too close to obstacles)
        self.collision_stop = bool(self.cfg.task.get("collision_stop", True))
        self.safety_dis = float(self.cfg.task.get("safety_dis", 0.3))
        self.safety_shield_enable = bool(self.cfg.task.get("safety_shield_enable", True))
        self.safety_shield_soft_margin = float(self.cfg.task.get("safety_shield_soft_margin", 0.8))
        self.safety_shield_floor_margin = float(self.cfg.task.get("safety_shield_floor_margin", 0.35))
        self.safety_shield_floor_gain = float(self.cfg.task.get("safety_shield_floor_gain", 4.0))
        self.safety_shield_floor_bias_max = float(self.cfg.task.get("safety_shield_floor_bias_max", 2.0))
        self.min_depth = None
        self.last_shield_info = {
            "shield_active": False,
            "shield_reason": "none",
            "shield_scale_xy": 1.0,
            "shield_floor_bias": 0.0,
        }

        # optionally freeze motion when reaching the goal
        self.stop_when_reach_goal = bool(self.cfg.task.get("stop_when_reach_goal", False))
        self.reach_goal_dis = float(self.cfg.task.get("reach_goal_dis", 1.0))
        self.freeze_motion = False

        # velocity damping helps reduce drift in the pure-kinematic sim
        self.vel_damping = float(self.cfg.task.get("vel_damping", 0.05))
        self.vel_limit = float(self.cfg.task.get("vel_limit", 6.0))

        # flow/radial perception settings (will be overwritten by cfg.task.* in init_params)
        self.flow_gap = 25
        self.flow_slide_window = 5
        # reference limits (match training defaults; can be overridden in cfg.task)
        self.vel_ref = float(self.cfg.task.get("vel_max", 5.0))
        self.acc_ref = float(self.cfg.task.get("acc_ref", self.cfg.task.get("acc_max", 10.0)))

        # Kinematic sim yaw (used only for publishing /sim/odom orientation).
        # Training controller locks yaw to pi/4 by default; keep the same default here.
        self.yaw = float(self.cfg.task.get("fixed_yaw", math.pi / 4))

        # auto goal mode
        # - auto_test=True  : use default_goal_xyz if no external goal is given
        # - auto_test_toggle=True : keep toggling between auto_goal_a/auto_goal_b
        self.auto_test = bool(self.cfg.task.get("auto_test", True))
        self.auto_test_toggle = bool(self.cfg.task.get("auto_test_toggle", False))
        self.auto_goal_a = self.cfg.task.get("auto_goal_a", [0.0, 15.0, self.zpose])
        self.auto_goal_b = self.cfg.task.get("auto_goal_b", [0.0, -15.0, self.zpose])
        if len(self.auto_goal_a) == 2:
            self.auto_goal_a = [self.auto_goal_a[0], self.auto_goal_a[1], self.zpose]
        if len(self.auto_goal_b) == 2:
            self.auto_goal_b = [self.auto_goal_b[0], self.auto_goal_b[1], self.zpose]
        self.toggle_left_right = False
        self.reach_goal_time = None
        self.wait_goal_time = float(self.cfg.task.get("wait_goal_time", 3.0))
        self._last_published_goal = None

        # --- internal buffers
        self.dismap_image_queue = None
        self.dismap_flow_queue = None
        self.dismap_image_size = None
        self.dismap_flow_size = None
        self.flow_est_model = None
        self.depth_image_queue = None
        self.prev_depth_smoothed = None
        self.prev_pos = None
        self.prev_rot = None
        self.prev_time = None
        self.lidar_dirs = None
        self.risk_virtual_depth = None
        self.risk_virtual_age = None

        # last commanded acceleration in world frame (m/s^2), used for the state "acc_input"
        self.last_target_acc = torch.zeros(1, 3)

        self.init_params()
        self.init_policy()
        self.init_neuflow()
        
        self.odom_pub = rospy.Publisher("/sim/odom", Odometry, queue_size=10)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

        # Publish an initial odom immediately, so raycast/map nodes that depend on /sim/odom
        # can start producing /ray2array_hits without deadlocking.
        self._publish_odom()
        
        self.rayhits_sub = rospy.Subscriber('/ray2array_hits', Float32MultiArray, self.lidar_callback)   
        self.odom_sub = rospy.Subscriber("/sim/odom", Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)

    def init_params(self):
        self.lidar_range = self.cfg.task.lidar_range
        self.lidar_h_res = self.cfg.task.lidar_h_res
        self.lidar_v_res = self.cfg.task.lidar_v_res
        self.lidar_h_sample = self.cfg.task.lidar_h_sample
        self.lidar_v_sample = self.cfg.task.lidar_v_sample
        self.lidar_hfov = self.cfg.task.lidar_hfov
        self.lidar_vfov = self.cfg.task.lidar_vfov
        self.lidar_use_height_filter = self.cfg.task.get("lidar_use_height_filter", True)
        self.lidar_radial_window = int(self.cfg.task.get("lidar_radial_window", 3))
        self.lidar_radial_min_depth = float(self.cfg.task.get("lidar_radial_min_depth", 0.1))
        self.lidar_radial_max_depth = float(self.cfg.task.get("lidar_radial_max_depth", self.lidar_range))
        self.lidar_radial_min_speed = float(self.cfg.task.get("lidar_radial_min_speed", 0.0))
        self.lidar_radial_max_speed = float(self.cfg.task.get("lidar_radial_max_speed", 10.0))
        self.lidar_radial_invalid_value = float(self.cfg.task.get("lidar_radial_invalid_value", 0.0))
        self.lidar_radial_dt = float(self.cfg.task.get("lidar_radial_dt", 0.02))
        self.lidar_risk_decay_tau = float(self.cfg.task.get("lidar_risk_decay_tau", 1.0))
        self.lidar_risk_clear_time = float(self.cfg.task.get("lidar_risk_clear_time", 3.0))
        self.lidar_risk_clear_margin = float(self.cfg.task.get("lidar_risk_clear_margin", 1.0))
        self.lidar_risk_enable = bool(self.cfg.task.get("lidar_risk_enable", True))
        self.lidar_v_edge_margin_deg = self.cfg.task.get("lidar_v_edge_margin_deg", None)
        # --- height / kinematics bounds (match training env.py) ---
        # Training has a virtual floor/ceiling; without this, the policy can keep drifting down in ROS.
        self.virtual_ground = float(self.cfg.task.get("virtual_ground", 0.5))
        self.virtual_ceiling = float(self.cfg.task.get("virtual_ceiling", 3.5))
        # Training terminates if speed > 1.2 * vel_ref (vel_ref defaults to 5.0 -> limit 6.0)
        self.vel_limit = float(self.cfg.task.get("vel_limit", 1.2 * self.vel_ref))
        # Use fixed dt consistent with training (default 0.02s)
        self.dt = float(self.cfg.task.get("dt", self.lidar_radial_dt))
        self.bound_h = self.cfg.task.bound_h
        self.lidar_resolution = (self.lidar_h_res, self.lidar_v_res)
        self.flow_gap = self.cfg.task.flow_gap
        self.flow_slide_window = self.cfg.task.flow_slide_window
        self.num_envs = torch.tensor(1).to(self.device)
        self.last_target_acc = self.last_target_acc.to(self.device)
        self.lidar_dirs = compute_rayhitsdir(
            self.device,
            1,
            self.lidar_hfov,
            self.lidar_vfov,
            self.lidar_h_res,
            self.lidar_v_res,
        )

        # Try to read map size from ROS params (map_generator/dynamic_env).
        # In dynamic_env.cpp: x_size_wall = x_size + 3, y_size_wall = y_size + 10.
        # We use those to build "virtual walls" in inference so the policy can't drift outside.
        try:
            x_size = rospy.get_param("/dynamic_env/map/x_size", None)
            y_size = rospy.get_param("/dynamic_env/map/y_size", None)
            if x_size is None:
                x_size = self.cfg.task.get("map_x_size", None)
            if y_size is None:
                y_size = self.cfg.task.get("map_y_size", None)
            if x_size is not None:
                self.x_bound = (float(x_size) + float(self.map_x_wall_extra)) / 2.0
            if y_size is not None:
                self.y_bound = (float(y_size) + float(self.map_y_wall_extra)) / 2.0
        except Exception:
            pass
        x_bound_cfg = self.cfg.task.get("x_bound", None)
        y_bound_cfg = self.cfg.task.get("y_bound", None)
        if x_bound_cfg is not None:
            self.x_bound = float(x_bound_cfg)
        if y_bound_cfg is not None:
            self.y_bound = float(y_bound_cfg)

    def init_policy(self):
        batch_size = 1
        device = self.device
        
        observation_spec_ = CompositeSpec(
            state=UnboundedContinuousTensorSpec(
                shape=(batch_size, 9), device=device, dtype=torch.float32
            ),
            lidar=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4, *self.lidar_resolution),
                device=device,
                dtype=torch.float32,
            ),
            device=device,
        )

        intrinsics_spec = CompositeSpec(
            mass=UnboundedContinuousTensorSpec(
                shape=(batch_size, 1), device=device, dtype=torch.float32
            ),
            inertia=UnboundedContinuousTensorSpec(
                shape=(batch_size, 3), device=device, dtype=torch.float32
            ),
            com=UnboundedContinuousTensorSpec(
                shape=(batch_size, 3), device=device, dtype=torch.float32
            ),
            KF=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4), device=device, dtype=torch.float32
            ),
            KM=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4), device=device, dtype=torch.float32
            ),
            tau_up=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4), device=device, dtype=torch.float32
            ),
            tau_down=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4), device=device, dtype=torch.float32
            ),
            drag_coef=UnboundedContinuousTensorSpec(
                shape=(batch_size, 1), device=device, dtype=torch.float32
            ),
            device=device,
        )

        stats_spec = CompositeSpec(
            **{
                "return": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device=device, dtype=torch.float32
                ),
                "episode_len": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device=device, dtype=torch.float32
                ),
                "action_smoothness": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device=device, dtype=torch.float32
                ),
                "safety": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device=device, dtype=torch.float32
                ),
            },
            device=device,
        )
        
        self.drone_intrinsics_spec_ = intrinsics_spec.zero().to(device)
        
        base_env_observation_spec = CompositeSpec(
            agents=CompositeSpec(
                observation=observation_spec_,
                intrinsics=intrinsics_spec,
                device=device,
            ),
            stats=stats_spec,
            device=device,
        )

        action_shape = torch.Size([128, 3])
        low_bound = torch.full(action_shape, -1.0, device=device, dtype=torch.float32)
        high_bound = torch.full(action_shape, 1.0, device=device, dtype=torch.float32)

        action_spec = BoundedTensorSpec(
            shape=action_shape,
            minimum=low_bound,
            maximum=high_bound,
            device=device,
            dtype=torch.float32,
            domain="continuous"
        )
        reward_shape = torch.Size([128, 1])
        reward_spec = UnboundedContinuousTensorSpec(
            shape=reward_shape,
            device=device,
            dtype=torch.float32,
            domain="continuous"
        )
        
        try:
            OmegaConf.register_new_resolver("eval", eval)
        except Exception:
            pass
        OmegaConf.resolve(self.cfg)
        OmegaConf.set_struct(self.cfg, False)
        
        self.stats = stats_spec.zero()
        self.observation_spec = base_env_observation_spec
        
        self.policy = PPOPolicy(
            self.cfg.algo,
            getattr(self.cfg, "model", None),
            self.observation_spec,
            action_spec,
            reward_spec,
            device=self.device
        )
        
        ckpt_path = str(self.cfg.task.get("ckpt_path", "../models/orp7.2.pt"))
        # Hydra may chdir into outputs/...; resolve checkpoint path robustly.
        if not os.path.isabs(ckpt_path):
            try:
                ckpt_path = to_absolute_path(ckpt_path)
            except Exception:
                ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ckpt_path))
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        rospy.loginfo(f"[infer] Loaded checkpoint: {ckpt_path}")
        
        filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                                if not any(prefix in k for prefix in ['critic.', 'gae.', 'value_norm.'])}
        
        self.policy.load_state_dict(filtered_checkpoint, strict=False)
        self.policy.eval()
        
        self.tensordict = self.observation_spec.zero()
        with torch.no_grad():
            _ = self.policy.act_mean(self.observation_spec.zero())
        self.policy.reset_rollout_state()

    def init_neuflow(self):
        self.dismap_flow_size = (96, 16)
        self.flow_est_model = init_neuflow(1, self.dismap_flow_size, device=self.device)

    def goal_callback(self, msg):
        """Goal in world frame.

        If auto_test_toggle is enabled, goals are internally controlled by odom_callback,
        so external RViz goals are ignored to avoid fighting commands.
        """
        if self.auto_test and self.auto_test_toggle:
            return
        z = self.zpose
        self.target_pos_np = [float(msg.pose.position.x), float(msg.pose.position.y), float(z)]
        if hasattr(self, "policy"):
            self.policy.reset_rollout_state()

    def odom_callback(self, msg):
        self.odom = msg

        # Optionally stop when we reach the goal (useful for evaluation).
        if (not self.freeze_motion) and self.stop_when_reach_goal and (self.target_pos_np is not None):
            dx = float(msg.pose.pose.position.x) - float(self.target_pos_np[0])
            dy = float(msg.pose.pose.position.y) - float(self.target_pos_np[1])
            dz = float(msg.pose.pose.position.z) - float(self.target_pos_np[2])
            dis = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dis <= float(self.reach_goal_dis):
                self.freeze_motion = True
                return

        # Auto toggle mode: switch between two goals after reaching them.
        if not (self.auto_test and self.auto_test_toggle):
            return

        # Always set the current internal target
        current_goal = self.auto_goal_a if (not self.toggle_left_right) else self.auto_goal_b
        other_goal = self.auto_goal_b if (not self.toggle_left_right) else self.auto_goal_a
        self.target_pos_np = [float(current_goal[0]), float(current_goal[1]), float(current_goal[2])]

        # Publish goal for visualization (RViz).
        if self._last_published_goal != self.target_pos_np:
            goal_msg = PoseStamped()
            goal_msg.header.stamp = rospy.Time.now()
            goal_msg.header.frame_id = "world"
            goal_msg.pose.position.x = self.target_pos_np[0]
            goal_msg.pose.position.y = self.target_pos_np[1]
            goal_msg.pose.position.z = self.target_pos_np[2]
            goal_msg.pose.orientation.w = 1.0
            self.goal_pub.publish(goal_msg)
            self._last_published_goal = list(self.target_pos_np)

        # Check if reached current goal
        dx = float(msg.pose.pose.position.x) - float(current_goal[0])
        dy = float(msg.pose.pose.position.y) - float(current_goal[1])
        dz = float(msg.pose.pose.position.z) - float(current_goal[2])
        dis_cur = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dis_cur <= float(self.reach_goal_dis):
            if self.reach_goal_time is None:
                self.reach_goal_time = time.time()
            elif (time.time() - self.reach_goal_time) >= float(self.wait_goal_time):
                # switch
                self.toggle_left_right = not self.toggle_left_right
                self.reach_goal_time = None
                self._last_published_goal = None  # force publish on next callback
        else:
            self.reach_goal_time = None

    def lidar_callback(self, msg):
        if (self.target_pos_np is None):
            return

        tensordict = self.prepare_input(msg)

        with torch.no_grad():
            raw_action = self.policy.act_mean(tensordict)

        raw_action = torch.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        # policy outputs normalized actions (roughly [-1, 1]), env scales by acc_ref
        raw_action = torch.clamp(raw_action, -1.0, 1.0)
        target_acc = raw_action * self.acc_ref
        # Optional goal-directed acceleration bias (helps if the policy prefers "just avoiding" and not making progress).
        goal_acc_bias = float(self.cfg.task.get("goal_acc_bias", 0.0))
        if (goal_acc_bias != 0.0) and (self.target_pos_np is not None):
            dx = float(self.target_pos_np[0]) - float(self.odom.pose.pose.position.x)
            dy = float(self.target_pos_np[1]) - float(self.odom.pose.pose.position.y)
            dz = float(self.target_pos_np[2]) - float(self.odom.pose.pose.position.z)
            goal_dir = torch.tensor([dx, dy, dz], device=self.device, dtype=target_acc.dtype)
            goal_dir = goal_dir / (goal_dir.norm() + 1e-6)
            target_acc = target_acc + goal_acc_bias * goal_dir.view(1, 3)
            target_acc = torch.clamp(target_acc, -self.acc_ref, self.acc_ref)

        target_acc = torch.clamp(target_acc, -self.acc_ref, self.acc_ref)
        if self.safety_shield_enable:
            shield = apply_target_acc_safety_shield(
                target_acc,
                min_depth=float("inf") if self.min_depth is None else float(self.min_depth),
                z=float(self.posz),
                virtual_ground=float(self.virtual_ground),
                safety_dis=float(self.safety_dis),
                acc_ref=float(self.acc_ref),
                soft_margin=float(self.safety_shield_soft_margin),
                floor_margin=float(self.safety_shield_floor_margin),
                floor_gain=float(self.safety_shield_floor_gain),
                floor_bias_max=float(self.safety_shield_floor_bias_max),
            )
            target_acc = shield["target_acc"]
            self.last_shield_info = {
                "shield_active": bool(shield["shield_active"][0].item()),
                "shield_reason": shield["shield_reason"][0],
                "shield_scale_xy": float(shield["shield_scale_xy"][0].item()),
                "shield_floor_bias": float(shield["shield_floor_bias"][0].item()),
                "target_acc_before": shield["target_acc_before"][0].detach().cpu().tolist(),
                "target_acc_after": shield["target_acc_after"][0].detach().cpu().tolist(),
            }
            if self.last_shield_info["shield_active"]:
                rospy.loginfo_throttle(
                    1.0,
                    "[infer] shield_active=1 shield_reason=%s shield_scale_xy=%.3f shield_floor_bias=%.3f target_acc_before=%s target_acc_after=%s"
                    % (
                        self.last_shield_info["shield_reason"],
                        self.last_shield_info["shield_scale_xy"],
                        self.last_shield_info["shield_floor_bias"],
                        self.last_shield_info["target_acc_before"],
                        self.last_shield_info["target_acc_after"],
                    ),
                )
        self.last_target_acc = target_acc
        self.acccmd_2_odom(target_acc)



    def prepare_input(self, msg):
        data = np.array(msg.data)
        reshaped_data = data.reshape(1, self.lidar_h_res*self.lidar_v_res*self.lidar_h_sample*self.lidar_v_sample, 3)
        ray_hits_w = torch.tensor(reshaped_data, dtype=torch.float32).to(self.device)

        drone_state = [self.odom.pose.pose.position.x,
                       self.odom.pose.pose.position.y,
                       self.odom.pose.pose.position.z,
                       self.odom.twist.twist.linear.x,
                       self.odom.twist.twist.linear.y,
                       self.odom.twist.twist.linear.z]
        drone_state = torch.tensor(np.array(drone_state).reshape(1, 6), dtype=torch.float32).unsqueeze(1).to(self.device)

        self.target_pos_np = np.array(self.target_pos_np).reshape(1, 3)
        self.target_pos = torch.tensor(self.target_pos_np, dtype=torch.float32).unsqueeze(1).to(self.device)

        distances = (ray_hits_w - drone_state[..., :3]).norm(dim=-1)
        valid = (distances > 0) & (distances <= self.lidar_range)
        if self.lidar_use_height_filter:
            pos_w_z = drone_state[..., 2].squeeze(1)
            ray_hits_w_z = ray_hits_w[:, :, 2]
            z_in_range = (ray_hits_w_z >= (pos_w_z - self.bound_h)) & (ray_hits_w_z <= (pos_w_z + 2*self.bound_h))
            valid = valid & z_in_range
        lidar_dis = torch.where(valid, distances, torch.full_like(distances, float("inf")))
        # store min depth (for optional collision-stop in the kinematic sim)
        self.min_depth = float(torch.nan_to_num(lidar_dis, nan=float("inf")).min().item())

        # Sector distance map (meters): [1,1,H,W]
        lidar_dis_unfold = lidar_dis.reshape(
            self.num_envs, self.lidar_h_res * self.lidar_h_sample, self.lidar_v_res * self.lidar_v_sample
        ).unfold(1, self.lidar_h_sample, self.lidar_h_sample).unfold(2, self.lidar_v_sample, self.lidar_v_sample)
        lidar_scan_raw = lidar_dis_unfold.reshape(
            self.num_envs, 1, self.lidar_h_res * self.lidar_v_res, self.lidar_h_sample * self.lidar_v_sample
        ).min(dim=-1)[0]
        lidar_scan_dis = lidar_scan_raw.reshape(1, 1, *self.lidar_resolution)
        lidar_scan_dis = torch.where(
            torch.isfinite(lidar_scan_dis),
            lidar_scan_dis.clamp_max(self.lidar_range),
            torch.full_like(lidar_scan_dis, self.lidar_range),
        )

        # --- Optional virtual risk memory for vertical FOV edges (risk-only)
        if self.lidar_risk_enable and self.risk_virtual_depth is None:
            self.risk_virtual_depth = torch.full_like(lidar_scan_dis, self.lidar_range + self.lidar_risk_clear_margin)
            self.risk_virtual_age = torch.zeros_like(lidar_scan_dis)

        if self.lidar_risk_enable:
            decay_target = self.lidar_range + self.lidar_risk_clear_margin
            decay_tau = float(self.cfg.task.get("lidar_risk_virtual_decay_tau", self.lidar_risk_decay_tau))
            clear_time = float(self.cfg.task.get("lidar_risk_virtual_clear_time", self.lidar_risk_clear_time))
            dt = float(self.lidar_radial_dt)  # keep consistent with training env dt

            # vertical edge band margin (deg). If not set, use ~1 vertical bin.
            v_margin = self.lidar_v_edge_margin_deg
            if v_margin is None:
                if self.lidar_v_res > 1:
                    v_step = (float(self.lidar_vfov[1]) - float(self.lidar_vfov[0])) / float(self.lidar_v_res - 1)
                    v_margin = float(v_step) * 1.1
                else:
                    v_margin = 0.0
            v_margin = float(v_margin)

            v_angles = torch.linspace(
                float(self.lidar_vfov[0]), float(self.lidar_vfov[1]),
                self.lidar_v_res, device=self.device
            )
            edge_v = (v_angles >= (float(self.lidar_vfov[1]) - v_margin)) | (v_angles <= (float(self.lidar_vfov[0]) + v_margin))
            edge_mask = edge_v.view(1, 1, 1, self.lidar_v_res).expand(1, 1, self.lidar_h_res, self.lidar_v_res)

            hit_mask = lidar_scan_dis < self.lidar_range
            nohit_mask = ~hit_mask
            edge_hit = hit_mask & edge_mask
            edge_nohit = nohit_mask & edge_mask

            # update virtual memory on edge hits
            self.risk_virtual_depth = torch.where(edge_hit, lidar_scan_dis, self.risk_virtual_depth)
            self.risk_virtual_age = torch.where(edge_hit, torch.zeros_like(self.risk_virtual_age), self.risk_virtual_age)

            # keep only for edge bands; clear outside edges
            self.risk_virtual_depth = torch.where(
                edge_mask,
                self.risk_virtual_depth,
                torch.full_like(self.risk_virtual_depth, decay_target),
            )
            self.risk_virtual_age = torch.where(edge_mask, self.risk_virtual_age, torch.zeros_like(self.risk_virtual_age))

            has_risk = self.risk_virtual_depth < self.lidar_range
            keep_mask = edge_nohit & has_risk
            clear_now_mask = edge_nohit & (~keep_mask)

            self.risk_virtual_age = torch.where(keep_mask, self.risk_virtual_age + dt, self.risk_virtual_age)
            self.risk_virtual_age = torch.where(clear_now_mask, torch.zeros_like(self.risk_virtual_age), self.risk_virtual_age)

            decay_factor = torch.exp(- (self.risk_virtual_age / max(decay_tau, 1e-6)) ** 2)
            risk_depth_decay = decay_target - (decay_target - self.risk_virtual_depth) * decay_factor
            clear_decay_mask = (self.risk_virtual_age >= clear_time) | (risk_depth_decay >= decay_target)

            self.risk_virtual_depth = torch.where(
                keep_mask,
                torch.where(clear_decay_mask, torch.full_like(self.risk_virtual_depth, decay_target), risk_depth_decay),
                self.risk_virtual_depth,
            )
            self.risk_virtual_depth = torch.where(
                clear_now_mask,
                torch.full_like(self.risk_virtual_depth, decay_target),
                self.risk_virtual_depth,
            )

            # risk-only fused distance (not used for policy observation)
            self.lidar_scan_dis_risk = torch.minimum(lidar_scan_dis, self.risk_virtual_depth)

        if self.depth_image_queue is None:
            window = max(1, self.lidar_radial_window)
            self.depth_image_queue = deque([lidar_scan_dis] * window, maxlen=window)
        else:
            self.depth_image_queue.append(lidar_scan_dis)
        depth_stack = torch.stack(list(self.depth_image_queue))
        depth_smoothed = torch.mean(depth_stack, dim=0)
        depth_denoised = torch.median(depth_stack, dim=0).values

        radial_channel = torch.full_like(lidar_scan_dis, self.lidar_radial_invalid_value)
        radial_speed_channel = torch.zeros_like(lidar_scan_dis)
        radial_valid_channel = torch.zeros_like(lidar_scan_dis, dtype=torch.bool)
        if (self.prev_depth_smoothed is not None) and (self.prev_pos is not None) and (self.prev_rot is not None):
            dir_flat = self.lidar_dirs.reshape(1, -1, 3)
            prev_depth_flat = self.prev_depth_smoothed.reshape(1, -1)
            curr_depth_flat = depth_smoothed.reshape(1, -1)

            curr_pos = drone_state[..., :3].squeeze(1)
            ori = self.odom.pose.pose.orientation
            curr_rot = torch.tensor([[ori.w, ori.x, ori.y, ori.z]], dtype=torch.float32, device=self.device)
            prev_pos = self.prev_pos
            prev_rot = self.prev_rot

            prev_rot_expand = prev_rot.unsqueeze(1).expand(-1, dir_flat.shape[1], -1)
            curr_rot_expand = curr_rot.unsqueeze(1).expand(-1, dir_flat.shape[1], -1)
            prev_pos_expand = prev_pos.unsqueeze(1)
            curr_pos_expand = curr_pos.unsqueeze(1)

            p_prev_body = dir_flat * prev_depth_flat.unsqueeze(-1)
            p_prev_world = quat_rotate(prev_rot_expand, p_prev_body) + prev_pos_expand
            p_prev_curr = quat_rotate_inverse(curr_rot_expand, p_prev_world - curr_pos_expand)

            pred_depth = (p_prev_curr * dir_flat).sum(-1)
            dt = float(self.lidar_radial_dt)
            if bool(self.cfg.task.get("mgdp_v2_use_effective_dt", False)):
                dt = float(self.cfg.task.get("sim_dt", self.lidar_radial_dt))
            residual = curr_depth_flat - pred_depth
            # 保留靠近为正的语义，突出障碍物接近的危险性
            radial_speed = - residual / max(dt, 1e-6) 

            speed_abs = radial_speed.abs()
            valid = (
                (curr_depth_flat >= self.lidar_radial_min_depth) &
                (curr_depth_flat <= self.lidar_radial_max_depth) &
                (pred_depth > 0.0) &
                (speed_abs >= self.lidar_radial_min_speed) &
                (speed_abs <= self.lidar_radial_max_speed)
            )
            radial_speed = torch.clamp(radial_speed, -self.lidar_radial_max_speed, self.lidar_radial_max_speed)
            radial_norm = radial_speed / max(self.lidar_radial_max_speed, 1e-6)
            radial_norm = torch.clamp(radial_norm, -1.0, 1.0)
            radial_flat = torch.where(
                valid,
                radial_norm,
                torch.full_like(radial_norm, self.lidar_radial_invalid_value),
            )
            radial_channel = radial_flat.reshape(1, 1, *self.lidar_resolution)
            radial_speed_channel = torch.where(valid, radial_speed, torch.zeros_like(radial_speed)).reshape(
                1, 1, *self.lidar_resolution
            )
            radial_valid_channel = valid.reshape(1, 1, *self.lidar_resolution)

        rpos = self.target_pos - drone_state[..., :3]
        target_dir = rpos / rpos.norm(dim=-1, keepdim=True).clamp(1e-6)
        vel_fb = (drone_state[..., 3:])
        vel_input = vel_fb/self.vel_ref

        if self.last_target_acc is not None:
            acc_fb = self.last_target_acc.unsqueeze(1)
        else:
            acc_fb = torch.zeros_like(vel_fb)
        acc_input = acc_fb/self.acc_ref

        # training uses proximity channel = (range - distance) / range
        scan_prox = self.lidar_range - lidar_scan_dis
        scan_normalized = torch.clamp(torch.nan_to_num(scan_prox / max(self.lidar_range, 1e-6), nan=0.0), 0.0, 1.0)

        radial_channel = torch.clamp(
            torch.nan_to_num(radial_channel, nan=self.lidar_radial_invalid_value,
                             posinf=self.lidar_radial_invalid_value, neginf=self.lidar_radial_invalid_value),
            -1.0,
            1.0,
        )
        input_mode = str(self.cfg.task.get("input_mode", "p2m")).lower()
        p2m_radial_channel = radial_channel
        if input_mode == "p2m":
            p2m_radial_channel = torch.zeros_like(radial_channel)

        flow_normalized = torch.zeros((1, 2, *self.lidar_resolution), device=self.device, dtype=lidar_scan_dis.dtype)
        if input_mode not in ("mgdp", "mgdp_lite", "mgdp_lite_v2"):
            lidar_dis_for_flow = torch.where(
                torch.isfinite(lidar_dis),
                lidar_dis,
                torch.full_like(lidar_dis, self.lidar_range),
            )
            scan4flow = (self.lidar_range - lidar_dis_for_flow.unsqueeze(1)).reshape(
                                1, 1,
                                self.lidar_h_res * self.lidar_h_sample,
                                self.lidar_v_res * self.lidar_v_sample
                                )/self.lidar_range

            scan4flow_scaled = torch.nn.functional.interpolate(scan4flow.half() * 255.,
                                                                self.dismap_flow_size,
                                                                mode='bilinear',
                                                                align_corners=False)

            if self.dismap_image_queue is None:
                self.dismap_image_queue = deque([scan4flow_scaled] * int(self.flow_gap + 3), maxlen=int(self.flow_gap + 3))
            else:
                self.dismap_image_queue.append(scan4flow_scaled)

            dismap_tensor = list(self.dismap_image_queue)
            dismap_image0 = torch.cat(dismap_tensor[:3], dim=1)
            dismap_image1 = torch.cat(dismap_tensor[-3:], dim=1)

            with torch.no_grad():
                dismap_image0 = dismap_image0.half()
                dismap_image1 = dismap_image1.half()
                dismap_flow = self.flow_est_model(dismap_image0, dismap_image1)[-1]

            if self.dismap_flow_queue is None:
                self.flow_slide_window = int(self.flow_slide_window)
                self.dismap_flow_queue = deque([dismap_flow] * self.flow_slide_window, maxlen=self.flow_slide_window)
            else:
                self.dismap_flow_queue.append(dismap_flow)
            dismap_flow_mean = torch.mean(torch.stack(list(self.dismap_flow_queue)), dim=0)
            dismap_flow_scaled = torch.nn.functional.interpolate(dismap_flow_mean.float(),
                                                                        self.lidar_resolution,
                                                                        mode='bilinear',
                                                                        align_corners=False)

            flow_zoom = torch.nan_to_num(dismap_flow_scaled.float(), nan=0.0, posinf=0.0, neginf=0.0)
            flow_normalized = torch.cat(
                [
                    (flow_zoom[:, 0:1, :, :] / 3.6),
                    (flow_zoom[:, 1:2, :, :] / 0.6),
                ],
                dim=1,
            )
            flow_normalized = torch.clamp(flow_normalized, -1.0, 1.0)

        if input_mode in ("mgdp", "mgdp_lite"):
            denoised_scan = self.lidar_range - depth_denoised
            denoised_scan = torch.clamp(
                torch.nan_to_num(denoised_scan / max(self.lidar_range, 1e-6), nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )

            dir_z = self.lidar_dirs[..., 2].reshape(1, 1, *self.lidar_resolution)
            pos_z = drone_state[..., 2].reshape(1, 1, 1, 1)
            target_z = self.target_pos[..., 2].reshape(1, 1, 1, 1)
            hit_z = pos_z + dir_z * depth_denoised
            corridor_half_height = float(self.cfg.task.get("mgdp_corridor_half_height", 1.0))
            height_risk = 1.0 - torch.clamp((hit_z - target_z).abs() / max(corridor_half_height, 1e-6), 0.0, 1.0)
            corridor_risk = torch.clamp(height_risk * denoised_scan, 0.0, 1.0)

            approach_risk = torch.clamp(radial_channel, 0.0, 1.0)
            ttc_risk = torch.clamp(approach_risk * denoised_scan, 0.0, 1.0)
            dismap_stack = torch.cat([scan_normalized, denoised_scan, corridor_risk, ttc_risk], dim=1)
        elif input_mode == "mgdp_lite_v2":
            proximity = torch.clamp(
                torch.nan_to_num((self.lidar_range - depth_denoised) / max(self.lidar_range, 1e-6), nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
            speed_scale = float(self.cfg.task.get("mgdp_v2_radial_speed_scale", 6.0))
            radial_signed = torch.clamp(radial_speed_channel / max(speed_scale, 1e-6), -1.0, 1.0)
            radial_signed = torch.where(radial_valid_channel, radial_signed, torch.zeros_like(radial_signed))

            target_vec = (self.target_pos - drone_state[..., :3]).squeeze(1)
            target_vec = target_vec / target_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            ori = self.odom.pose.pose.orientation
            curr_rot = torch.tensor([[ori.w, ori.x, ori.y, ori.z]], dtype=torch.float32, device=self.device)
            yaw = quaternion_to_euler(curr_rot)[..., 2]
            zeros = torch.zeros_like(yaw)
            q_yaw = euler_to_quaternion(torch.stack([zeros, zeros, yaw], dim=-1))
            target_sensor = quat_rotate_inverse(q_yaw, target_vec)
            target_sensor = target_sensor / target_sensor.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

            dir_flat = self.lidar_dirs.reshape(1, -1, 3)
            depth_flat = depth_denoised.reshape(1, -1)
            p_hit = dir_flat * depth_flat.unsqueeze(-1)
            s = (p_hit * target_sensor.unsqueeze(1)).sum(dim=-1)
            p_perp = p_hit - s.unsqueeze(-1) * target_sensor.unsqueeze(1)
            d_perp = p_perp.norm(dim=-1).reshape(1, 1, *self.lidar_resolution)
            sigma = float(self.cfg.task.get("mgdp_v2_corridor_sigma", 1.0))
            if bool(self.cfg.task.get("mgdp_v2_corridor_speed_adaptive", False)):
                speed = drone_state[..., 3:].norm(dim=-1).view(1, 1, 1, 1)
                sigma = sigma + float(self.cfg.task.get("mgdp_v2_corridor_speed_gain", 0.12)) * speed
            sigma_t = torch.as_tensor(sigma, device=self.device, dtype=d_perp.dtype).clamp_min(1.0e-6)
            corridor_weight = torch.exp(-0.5 * (d_perp / sigma_t) ** 2)
            if bool(self.cfg.task.get("mgdp_v2_corridor_forward_only", True)):
                front_gate = (s.reshape(1, 1, *self.lidar_resolution) > 0.0).float()
            else:
                front_gate = torch.ones_like(corridor_weight)
            corridor_risk = torch.clamp(proximity * corridor_weight * front_gate, 0.0, 1.0)

            closing_speed = torch.clamp(radial_speed_channel, min=0.0)
            min_closing = float(self.cfg.task.get("mgdp_v2_ttc_min_closing_speed", 0.15))
            horizon = float(self.cfg.task.get("mgdp_v2_ttc_horizon", 4.0))
            tau = float(self.cfg.task.get("mgdp_v2_ttc_tau", 1.5))
            ttc = lidar_scan_dis / closing_speed.clamp_min(1.0e-6)
            valid_ttc = (lidar_scan_dis > self.lidar_radial_min_depth) & radial_valid_channel & (closing_speed > min_closing) & (ttc < horizon)
            ttc_risk = torch.exp(-ttc / max(tau, 1.0e-6))
            ttc_risk = torch.where(valid_ttc, ttc_risk, torch.zeros_like(ttc_risk))
            ttc_risk = torch.clamp(ttc_risk, 0.0, 1.0)

            dismap_stack = torch.cat([proximity, radial_signed, corridor_risk, ttc_risk], dim=1)
        else:
            dismap_stack = torch.cat([scan_normalized, flow_normalized, p2m_radial_channel], dim=1)

        obs = {
            "state": torch.cat([target_dir, vel_input, acc_input], dim=-1).squeeze(1),
            "lidar": dismap_stack
        }

        ori = self.odom.pose.pose.orientation
        self.prev_depth_smoothed = depth_smoothed
        self.prev_pos = drone_state[..., :3].squeeze(1)
        self.prev_rot = torch.tensor([[ori.w, ori.x, ori.y, ori.z]], dtype=torch.float32, device=self.device)
        # Keep dt fixed to match env.py; timestamps can be used later if needed.

        self.tensordict = TensorDict(
                {
                    "agents": TensorDict(
                        {
                            "observation": obs,
                            "intrinsics": self.drone_intrinsics_spec_,
                        },
                        [self.num_envs],
                    ),
                    "stats": self.stats.clone(),
                },
                1,
            )
        return self.tensordict

    def _publish_odom(self):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = float(self.velox)
        odom.twist.twist.linear.y = float(self.veloy)
        odom.twist.twist.linear.z = float(self.veloz)
        odom.pose.pose.position.x = float(self.posx)
        odom.pose.pose.position.y = float(self.posy)
        odom.pose.pose.position.z = float(self.posz)
        # Publish yaw so perception modules (e.g. raycast) can align rays with the UAV heading.
        # We keep roll/pitch at 0 in the kinematic sim.
        half = 0.5 * float(self.yaw)
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(half)
        odom.pose.pose.orientation.w = math.cos(half)
        self.odom = odom
        self.odom_pub.publish(odom)

    def acccmd_2_odom(self, actions):
        """Kinematic integration of acceleration -> velocity -> position.

        NOTE: This is NOT a physics simulator. There is no real collision response,
        so we add optional virtual constraints (ground/ceiling + XY walls) to keep the
        state distribution closer to training and prevent flying through the point-cloud walls.
        """
        # freeze (goal reached)
        if self.freeze_motion:
            self.velox = 0.0
            self.veloy = 0.0
            self.veloz = 0.0
            self._publish_odom()
            return

        # safety stop when too close
        if self.collision_stop and (self.min_depth is not None) and (float(self.min_depth) < float(self.safety_dis)):
            self.velox = 0.0
            self.veloy = 0.0
            self.veloz = 0.0
            self._publish_odom()
            return

        dt = float(self.sim_dt)
        ax = float(actions[0, 0].item())
        ay = float(actions[0, 1].item())
        az = float(actions[0, 2].item())

        # integrate velocity
        self.velox += ax * dt
        self.veloy += ay * dt
        self.veloz += az * dt

        # global velocity limit (helps avoid blow-ups in the pure-kinematic sim)
        vmag = math.sqrt(self.velox * self.velox + self.veloy * self.veloy + self.veloz * self.veloz)
        if (vmag > float(self.vel_limit)) and (vmag > 1e-6):
            scale = float(self.vel_limit) / vmag
            self.velox *= scale
            self.veloy *= scale
            self.veloz *= scale

        # simple damping
        damp = max(0.0, 1.0 - float(self.vel_damping) * dt)
        self.velox *= damp
        self.veloy *= damp
        self.veloz *= damp

        # integrate position
        self.posx += self.velox * dt
        self.posy += self.veloy * dt
        self.posz += self.veloz * dt

        # Z clamp (virtual floor/ceiling)
        z_min = float(self.virtual_ground)
        z_max = float(self.virtual_ceiling)
        if self.posz < z_min:
            self.posz = z_min
            if self.veloz < 0.0:
                self.veloz = 0.0
        if self.posz > z_max:
            self.posz = z_max
            if self.veloz > 0.0:
                self.veloz = 0.0

        # XY clamp (virtual walls)
        if self.enable_xy_bound and (self.x_bound is not None) and (self.y_bound is not None):
            margin = float(self.bound_margin)
            xb = float(self.x_bound)
            yb = float(self.y_bound)
            # X
            if self.posx < (-xb + margin):
                self.posx = -xb + margin
                self.velox = abs(self.velox) * float(self.bound_restitution)
            elif self.posx > (xb - margin):
                self.posx = xb - margin
                self.velox = -abs(self.velox) * float(self.bound_restitution)
            # Y
            if self.posy < (-yb + margin):
                self.posy = -yb + margin
                self.veloy = abs(self.veloy) * float(self.bound_restitution)
            elif self.posy > (yb - margin):
                self.posy = yb - margin
                self.veloy = -abs(self.veloy) * float(self.bound_restitution)

        self._publish_odom()


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(cfg):
    # Keep behaviour consistent with train/infer_ros.
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except Exception:
        pass
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    rospy.init_node("infer")
    _ = Infer(cfg)
    rospy.spin()


if __name__ == "__main__":
    main()
