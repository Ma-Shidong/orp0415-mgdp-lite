import os
import time
import math
from collections import deque
from typing import Optional, Tuple

import hydra
import numpy as np
import rospy
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import AttitudeTarget

from nav_msgs.msg import Odometry
from omegaconf import OmegaConf

from hydra.utils import to_absolute_path
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

from resources.NeuFlow_v2.infer_lidar import init_neuflow
from resources.learning.ppo.ppo import Actor, IndependentNormal, PPOConfig, make_mlp
from resources.utils.torch import quat_rotate, quat_rotate_inverse

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

def _resolve_device() -> torch.device:
    env_device = os.getenv("SIM_DEVICE")
    if env_device:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BatchConv2dWrapper(nn.Module):
    """Keep the same CNN wrapper as training to match checkpoint state_dict keys."""

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


class PPOPolicy(TensorDictModuleBase):
    """Inference-only policy (encoder + actor) matching scripts/train.py architecture."""

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device: torch.device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.action_dim = 3

        fake_input = observation_spec.zero()

        cnn = nn.Sequential(
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
        mlp = make_mlp([256, 256])

        self.encoder = TensorDictSequential(
            TensorDictModule(BatchConv2dWrapper(cnn), [("agents", "observation", "lidar")], ["_cnn_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
            TensorDictModule(mlp, ["_feature"], ["_feature"]),
        ).to(self.device)

        self.actor = ProbabilisticActor(
            TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=False,
        ).to(self.device)

        # materialize lazy modules
        self.encoder(fake_input)
        self.actor(fake_input)

    @torch.no_grad()
    def act_mean(self, tensordict: TensorDict) -> torch.Tensor:
        """Return mean action (not sampled) for stable closed-loop control."""
        self.encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        return dist.mean


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    return q


def _quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix (body axes in world if q is body->world)."""
    w, x, y, z = [float(v) for v in q_wxyz]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def _attitude_from_accel_and_yaw(acc_cmd_w: np.ndarray, yaw_sp: float, g: float) -> Tuple[np.ndarray, float]:
    """PX4-style attitude generation in ENU (ROS world).

    - Treat desired body Z axis as direction of (g*e_z + acc_cmd_w).
    - Use yaw_sp to define a reference heading in XY.
    - Orthonormalize axes (same cross-product order as PX4 ControlMath::thrustToAttitude).

    Returns:
        q_wxyz: quaternion (w,x,y,z) of body (base_link FLU) w.r.t world ENU.
        thrust_scale: |g*e_z + acc_cmd_w| / g (dimensionless), useful for mapping to throttle.
    """
    a_total = np.array([0.0, 0.0, g], dtype=np.float64) + acc_cmd_w.astype(np.float64)
    a_norm = float(np.linalg.norm(a_total))
    if a_norm < 1e-6:
        body_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        thrust_scale = 0.0
    else:
        body_z = a_total / a_norm
        thrust_scale = a_norm / g

    # y_C is the desired yaw direction in XY plane, rotated by +90deg (matches PX4 ControlMath)
    y_C = np.array([-np.sin(yaw_sp), np.cos(yaw_sp), 0.0], dtype=np.float64)

    if abs(body_z[2]) > 1e-6:
        body_x = np.cross(y_C, body_z)
        # keep nose to front while inverted upside down
        if body_z[2] < 0.0:
            body_x = -body_x
        nx = float(np.linalg.norm(body_x))
        if nx < 1e-6:
            body_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            body_x = body_x / nx
    else:
        # desired thrust is in XY plane, set X downwards to construct a valid matrix
        body_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    body_y = np.cross(body_z, body_x)
    ny = float(np.linalg.norm(body_y))
    if ny < 1e-6:
        body_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        body_y = body_y / ny

    # Recompute body_x to ensure orthonormality
    body_x = np.cross(body_y, body_z)
    nx = float(np.linalg.norm(body_x))
    if nx < 1e-6:
        body_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        body_x = body_x / nx

    R = np.column_stack([body_x, body_y, body_z])  # columns are body axes in world
    q_wxyz = _rotmat_to_quat_wxyz(R)
    return q_wxyz, thrust_scale


def _yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw from quaternion (x,y,z,w)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

class InferROS:
    """Gazebo + MAVROS inference node.

    Inputs:
      - /livox/lidar (sensor_msgs/PointCloud2)
      - /sim_odom    (nav_msgs/Odometry)
      - /move_base_simple/goal (geometry_msgs/PoseStamped)
    Outputs:
      - /mavros/setpoint_raw/attitude (mavros_msgs/AttitudeTarget)


    Notes:
      * The policy outputs world-frame 3D acceleration (m/s^2) that matches training.
      * Yaw is not learned; we set yaw_sp = heading(current -> goal).
      * Thrust is mapped from acceleration using a hover_thrust estimate (tune later).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = _resolve_device()

        self.odom_msg: Optional[Odometry] = None
        self.goal_msg: Optional[PoseStamped] = None

        # command hold
        self._last_q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_thrust = 0.5
        self._last_cmd_time = 0.0
        self._z_hold = None  # lock altitude when goal_z_mode == "hold"


        # perception temporal state
        self.dismap_image_queue = None
        self.dismap_flow_queue = None
        self.depth_image_queue = None
        self.prev_depth_smoothed = None
        self.prev_pos = None
        self.prev_rot = None
        self.risk_depth = None
        self.risk_age = None
        self.last_target_acc = torch.zeros(1, 3, device=self.device)

        # defaults (match env)
        self.vel_ref = float(getattr(cfg.task, "vel_max", 5.0)) if hasattr(cfg, "task") else 5.0
        self.acc_ref = float(getattr(cfg.task, "acc_max", 10.0)) if hasattr(cfg, "task") else 10.0

        self._init_params()
        self._init_policy()
        self._init_neuflow()

        # ROS I/O
        # self.pub_att = rospy.Publisher("/mavros/setpoint_attitude/attitude", PoseStamped, queue_size=10)
        # self.pub_thrust = rospy.Publisher("/mavros/setpoint_attitude/thrust", Thrust, queue_size=10)
        # ROS I/O (UESTC style): publish AttitudeTarget only
        self.pub_att_raw = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)



        rospy.Subscriber("/sim_odom", Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb, queue_size=1)
        rospy.Subscriber("/livox/lidar", PointCloud2, self._pcd_cb, queue_size=1)

        # publish at a stable rate (PX4 offboard needs continuous setpoints)
        rate_hz = float(rospy.get_param("~publish_rate_hz", 30.0))
        rospy.Timer(rospy.Duration(1.0 / max(rate_hz, 1.0)), self._publish_timer)

    def _init_params(self):
        # we run a single "env" in ROS
        self.num_envs = 1

        self.lidar_range = float(self.cfg.task.lidar_range)
        self.lidar_h_res = int(self.cfg.task.lidar_h_res)
        self.lidar_v_res = int(self.cfg.task.lidar_v_res)
        self.lidar_h_sample = int(self.cfg.task.lidar_h_sample)
        self.lidar_v_sample = int(self.cfg.task.lidar_v_sample)
        self.lidar_hfov = float(self.cfg.task.lidar_hfov)
        self.lidar_vfov = self.cfg.task.lidar_vfov
        self.lidar_use_height_filter = bool(self.cfg.task.get("lidar_use_height_filter", False))
        self.bound_h = float(self.cfg.task.get("bound_h", 1.0))

        self.lidar_radial_window = int(self.cfg.task.get("lidar_radial_window", 3))
        self.lidar_radial_min_depth = float(self.cfg.task.get("lidar_radial_min_depth", 0.1))
        self.lidar_radial_max_depth = float(self.cfg.task.get("lidar_radial_max_depth", self.lidar_range))
        self.lidar_radial_min_speed = float(self.cfg.task.get("lidar_radial_min_speed", 0.01))
        self.lidar_radial_max_speed = float(self.cfg.task.get("lidar_radial_max_speed", 10.0))
        self.lidar_radial_invalid_value = float(self.cfg.task.get("lidar_radial_invalid_value", 0.0))
        self.lidar_radial_dt = float(self.cfg.task.get("lidar_radial_dt", 0.02))

        self.lidar_risk_decay_tau = float(self.cfg.task.get("lidar_risk_decay_tau", 1.0))
        self.lidar_risk_clear_time = float(self.cfg.task.get("lidar_risk_clear_time", 3.0))
        self.lidar_risk_clear_margin = float(self.cfg.task.get("lidar_risk_clear_margin", 1.0))
        self.lidar_risk_enable = bool(self.cfg.task.get("lidar_risk_enable", True))

        # Match env.py FOV-related keep/clear rule
        self.lidar_effective_range = float(self.cfg.task.get("lidar_effective_range", 5.0))
        self.lidar_v_edge_margin_deg = self.cfg.task.get("lidar_v_edge_margin_deg", None)

        self.flow_gap = int(self.cfg.task.flow_gap)
        self.flow_slide_window = int(self.cfg.task.flow_slide_window)
        self.lidar_resolution = (self.lidar_h_res, self.lidar_v_res)

        self.lidar_dirs = compute_rayhitsdir(
            self.device, 1, self.lidar_hfov, self.lidar_vfov, self.lidar_h_res, self.lidar_v_res
        )

        # accel->attitude/thrust mapping
        self.g = float(rospy.get_param("~gravity", 9.81))
        self.hover_thrust = float(rospy.get_param("~hover_thrust", 0.5))
        self.thrust_min = float(rospy.get_param("~thrust_min", 0.05))
        self.thrust_max = float(rospy.get_param("~thrust_max", 0.95))

        self.goal_z_mode = rospy.get_param("~goal_z_mode", "hold")  # hold | fixed
        self.goal_z_fixed = float(rospy.get_param("~goal_z_fixed", 2.0))

        # Yaw is not learned by the policy; we choose a yaw setpoint strategy.
        #   - goal:  face the goal direction (original behaviour)
        #   - fixed: hold a constant yaw (matches default training controller yaw)
        #   - hold:  hold the current yaw from odometry
        self.yaw_mode = str(
            rospy.get_param("~yaw_mode", getattr(self.cfg.task, "yaw_mode", "fixed"))
        ).lower()
        self.fixed_yaw = float(
            rospy.get_param("~fixed_yaw", getattr(self.cfg.task, "fixed_yaw", math.pi / 4))
        )

        self.max_points = int(rospy.get_param("~max_points", 200000))
        self.use_ros_numpy = bool(rospy.get_param("~use_ros_numpy", True))
        # 倾角限幅
        self.max_tilt_deg = float(rospy.get_param("~max_tilt_deg", 35.0))
        # 积分高度
        self.hover_i_gain = float(rospy.get_param("~hover_i_gain", 0.0))   # 0 disables
        self.hover_i_limit = float(rospy.get_param("~hover_i_limit", 0.15))
        self._hover_i = 0.0



    def _init_policy(self):
        # observation spec: [state(9), lidar(4,H,W)]
        batch_size = 1
        device = self.device

        observation_spec_ = CompositeSpec(
            state=UnboundedContinuousTensorSpec(shape=(batch_size, 9), device=device, dtype=torch.float32),
            lidar=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4, *self.lidar_resolution), device=device, dtype=torch.float32
            ),
            device=device,
        )

        intrinsics_spec = CompositeSpec(
            mass=UnboundedContinuousTensorSpec(shape=(batch_size, 1), device=device, dtype=torch.float32),
            inertia=UnboundedContinuousTensorSpec(shape=(batch_size, 3), device=device, dtype=torch.float32),
            com=UnboundedContinuousTensorSpec(shape=(batch_size, 3), device=device, dtype=torch.float32),
            KF=UnboundedContinuousTensorSpec(shape=(batch_size, 4), device=device, dtype=torch.float32),
            KM=UnboundedContinuousTensorSpec(shape=(batch_size, 4), device=device, dtype=torch.float32),
            tau_up=UnboundedContinuousTensorSpec(shape=(batch_size, 4), device=device, dtype=torch.float32),
            tau_down=UnboundedContinuousTensorSpec(shape=(batch_size, 4), device=device, dtype=torch.float32),
            drag_coef=UnboundedContinuousTensorSpec(shape=(batch_size, 1), device=device, dtype=torch.float32),
            device=device,
        )

        stats_spec = CompositeSpec(
            **{
                "return": UnboundedContinuousTensorSpec(shape=(batch_size, 1), device=device, dtype=torch.float32),
                "episode_len": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device=device, dtype=torch.float32
                ),
                "action_smoothness": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device=device, dtype=torch.float32
                ),
                "safety": UnboundedContinuousTensorSpec(shape=(batch_size, 1), device=device, dtype=torch.float32),
            },
            device=device,
        )

        self.drone_intrinsics_spec_ = intrinsics_spec.zero().to(device)
        self.stats = stats_spec.zero()
        self.observation_spec = CompositeSpec(
            agents=CompositeSpec(observation=observation_spec_, intrinsics=intrinsics_spec, device=device),
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
            domain="continuous",
        )
        reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([128, 1]), device=device, dtype=torch.float32)

        self.policy = PPOPolicy(self.cfg.algo, self.observation_spec, action_spec, reward_spec, device=device)

        # Load checkpoint
        # Priority: ROS param (~ckpt_path / ~checkpoint) > cfg.task.ckpt_path > repo default
        cfg_ckpt = None
        try:
            cfg_ckpt = getattr(self.cfg.task, "ckpt_path", None)
        except Exception:
            cfg_ckpt = None
        default_ckpt = cfg_ckpt if cfg_ckpt else os.path.join(os.path.dirname(__file__), "..", "models", "test2.pt")

        ckpt_path = rospy.get_param("~ckpt_path", None)
        if ckpt_path is None:
            ckpt_path = rospy.get_param("~checkpoint", None)
        if ckpt_path is None:
            ckpt_path = default_ckpt

        ckpt_path = str(ckpt_path)
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
        rospy.loginfo(f"[infer_ros] Loaded checkpoint: {ckpt_path}")
        filtered = {k: v for k, v in checkpoint.items() if not any(p in k for p in ["critic.", "gae.", "value_norm."])}
        self.policy.load_state_dict(filtered, strict=False)
        self.policy.eval()

        # make sure lazy layers are materialized
        td = self.observation_spec.zero()
        self.policy.encoder(td)
        _ = self.policy.actor.get_dist(td)

    def _init_neuflow(self):
        self.dismap_flow_size = (96, 16)
        self.flow_est_model = init_neuflow(1, self.dismap_flow_size, device=self.device)

    def _odom_cb(self, msg: Odometry):
        self.odom_msg = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_msg = msg
        if self.goal_z_mode == "hold":
            # re-lock altitude when a new goal arrives
            self._z_hold = None


    def _publish_timer(self, _evt):
        # If we haven't computed a new command for a while, still keep publishing last setpoint.
        now = rospy.Time.now()

        msg = AttitudeTarget()
        msg.header.stamp = now
        # Keep consistent with the UESTC px4ctrl / mavros raw setpoint convention
        msg.header.frame_id = "FCU"

        # We command attitude (orientation) + thrust. Body rates are ignored.
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )

        # Not used when the corresponding IGNORE_*_RATE bits are set, but fill for completeness.
        msg.body_rate.x = 0.0
        msg.body_rate.y = 0.0
        msg.body_rate.z = 0.0

        # self._last_q_wxyz is [w, x, y, z]
        w, x, y, z = self._last_q_wxyz
        msg.orientation.w = float(w)
        msg.orientation.x = float(x)
        msg.orientation.y = float(y)
        msg.orientation.z = float(z)

        # PX4 expects normalized thrust [0, 1]
        msg.thrust = float(self._last_thrust)

        self.pub_att_raw.publish(msg)


    def _pcd_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 -> Nx3 numpy array."""
        # Optionally use ros_numpy for speed.
        if self.use_ros_numpy:
            try:
                import ros_numpy  # type: ignore

                pc = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
                xyz = np.zeros((pc.shape[0], 3), dtype=np.float32)
                xyz[:, 0] = pc["x"].astype(np.float32)
                xyz[:, 1] = pc["y"].astype(np.float32)
                xyz[:, 2] = pc["z"].astype(np.float32)
                return xyz
            except Exception:
                pass

        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append((p[0], p[1], p[2]))
            if self.max_points > 0 and len(pts) >= self.max_points:
                break
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)

    def _depth_map_from_pointcloud(self, pts_body: np.ndarray, pos_w: np.ndarray, q_wxyz: np.ndarray) -> torch.Tensor:
        """Bin points into the same (H,V) sector depth map used in training.

        Returns distance map [1,1,H,V] with missing filled as +inf.
        """
        if pts_body.shape[0] == 0:
            dist_map = np.full((self.lidar_h_res, self.lidar_v_res), np.inf, dtype=np.float32)
            return torch.from_numpy(dist_map).to(self.device).view(1, 1, self.lidar_h_res, self.lidar_v_res)

        x = pts_body[:, 0].astype(np.float64)
        y = pts_body[:, 1].astype(np.float64)
        z = pts_body[:, 2].astype(np.float64)

        # Optional height band filter in world frame (matches env.py)
        if self.lidar_use_height_filter:
            R_bw = _quat_wxyz_to_rotmat(q_wxyz)  # body -> world
            z_w = R_bw[2, 0] * x + R_bw[2, 1] * y + R_bw[2, 2] * z + float(pos_w[2])
            z_in_range = (z_w >= (float(pos_w[2]) - self.bound_h)) & (z_w <= (float(pos_w[2]) + 2.0 * self.bound_h))
            x, y, z = x[z_in_range], y[z_in_range], z[z_in_range]
            if x.size == 0:
                dist_map = np.full((self.lidar_h_res, self.lidar_v_res), np.inf, dtype=np.float32)
                return torch.from_numpy(dist_map).to(self.device).view(1, 1, self.lidar_h_res, self.lidar_v_res)

        r = np.sqrt(x * x + y * y + z * z)
        mask = (r > 1e-6) & (r <= float(self.lidar_range))
        if not np.any(mask):
            dist_map = np.full((self.lidar_h_res, self.lidar_v_res), np.inf, dtype=np.float32)
            return torch.from_numpy(dist_map).to(self.device).view(1, 1, self.lidar_h_res, self.lidar_v_res)

        x = x[mask]
        y = y[mask]
        z = z[mask]
        r = r[mask]

        az = np.degrees(np.arctan2(y, x))
        az = np.mod(az, 360.0)
        el = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))

        v0 = float(self.lidar_vfov[0])
        v1 = float(self.lidar_vfov[1])
        mask_fov = (el >= v0) & (el <= v1)
        if not np.any(mask_fov):
            dist_map = np.full((self.lidar_h_res, self.lidar_v_res), np.inf, dtype=np.float32)
            return torch.from_numpy(dist_map).to(self.device).view(1, 1, self.lidar_h_res, self.lidar_v_res)

        az = az[mask_fov]
        el = el[mask_fov]
        r = r[mask_fov]

        h_step = float(self.lidar_hfov) / float(self.lidar_h_res)
        v_step = (v1 - v0) / float(max(self.lidar_v_res - 1, 1))

        h_idx = np.round(az / h_step).astype(np.int64) % self.lidar_h_res
        v_idx = np.round((el - v0) / v_step).astype(np.int64)
        v_idx = np.clip(v_idx, 0, self.lidar_v_res - 1)

        dist_map = np.full((self.lidar_h_res, self.lidar_v_res), np.inf, dtype=np.float32)
        np.minimum.at(dist_map, (h_idx, v_idx), r.astype(np.float32))

        return torch.from_numpy(dist_map).to(self.device).view(1, 1, self.lidar_h_res, self.lidar_v_res)

    def _update_risk_layer(self, lidar_scan_dis: torch.Tensor) -> torch.Tensor:
        """Apply the same FOV-aware keep+decay layer as env.py (distance space)."""
        if not self.lidar_risk_enable:
            return lidar_scan_dis

        if self.risk_depth is None:
            self.risk_depth = torch.full_like(lidar_scan_dis, self.lidar_range + self.lidar_risk_clear_margin)
            self.risk_age = torch.zeros_like(lidar_scan_dis)

        decay_target = self.lidar_range + self.lidar_risk_clear_margin
        dt = float(self.lidar_radial_dt)
        eff_range = min(float(self.lidar_effective_range), float(self.lidar_range))

        v_margin = self.lidar_v_edge_margin_deg
        if v_margin is None:
            if self.lidar_v_res > 1:
                v_step = (float(self.lidar_vfov[1]) - float(self.lidar_vfov[0])) / float(self.lidar_v_res - 1)
                v_margin = float(v_step) * 1.1
            else:
                v_margin = 0.0
        v_margin = float(v_margin)

        v_angles = torch.linspace(float(self.lidar_vfov[0]), float(self.lidar_vfov[1]), self.lidar_v_res, device=self.device)
        edge_v = (v_angles >= (float(self.lidar_vfov[1]) - v_margin)) | (v_angles <= (float(self.lidar_vfov[0]) + v_margin))
        edge_mask = edge_v.view(1, 1, 1, self.lidar_v_res).expand(1, 1, self.lidar_h_res, self.lidar_v_res)

        hit_mask = lidar_scan_dis < self.lidar_range
        nohit_mask = ~hit_mask

        self.risk_depth = torch.where(hit_mask, lidar_scan_dis, self.risk_depth)
        self.risk_age = torch.where(hit_mask, torch.zeros_like(self.risk_age), self.risk_age)

        has_risk = self.risk_depth < self.lidar_range
        keep_due_edge = edge_mask
        keep_due_far = self.risk_depth > eff_range
        keep_mask = nohit_mask & has_risk & (keep_due_edge | keep_due_far)
        clear_now_mask = nohit_mask & (~keep_mask)

        self.risk_age = torch.where(keep_mask, self.risk_age + dt, self.risk_age)
        self.risk_age = torch.where(clear_now_mask, torch.zeros_like(self.risk_age), self.risk_age)

        decay_factor = torch.exp(- (self.risk_age / max(self.lidar_risk_decay_tau, 1e-6)) ** 2)
        risk_depth_decay = decay_target - (decay_target - self.risk_depth) * decay_factor
        clear_decay_mask = (self.risk_age >= self.lidar_risk_clear_time) | (risk_depth_decay >= decay_target)

        self.risk_depth = torch.where(
            keep_mask,
            torch.where(clear_decay_mask, torch.full_like(self.risk_depth, decay_target), risk_depth_decay),
            self.risk_depth,
        )
        self.risk_depth = torch.where(clear_now_mask, torch.full_like(self.risk_depth, decay_target), self.risk_depth)

        return torch.minimum(lidar_scan_dis, self.risk_depth)

    def _build_observation(self, lidar_dis: torch.Tensor, pos_w: torch.Tensor, vel_w: torch.Tensor, q_wxyz: torch.Tensor, goal_w: torch.Tensor) -> TensorDict:
        """Match env.py observation construction: lidar(4ch) + state(9)."""
        lidar_scan_dis = torch.where(
            torch.isfinite(lidar_dis),
            lidar_dis.clamp_max(self.lidar_range),
            torch.full_like(lidar_dis, self.lidar_range),
        )

        lidar_scan_dis = self._update_risk_layer(lidar_scan_dis)

        if self.depth_image_queue is None:
            window = max(1, self.lidar_radial_window)
            self.depth_image_queue = deque([lidar_scan_dis] * window, maxlen=window)
        else:
            self.depth_image_queue.append(lidar_scan_dis)
        depth_smoothed = torch.mean(torch.stack(list(self.depth_image_queue)), dim=0)

        radial_channel = torch.full_like(lidar_scan_dis, self.lidar_radial_invalid_value)
        if self.prev_depth_smoothed is not None and self.prev_pos is not None and self.prev_rot is not None:
            dir_flat = self.lidar_dirs.reshape(1, -1, 3)
            prev_depth_flat = self.prev_depth_smoothed.reshape(1, -1)
            curr_depth_flat = depth_smoothed.reshape(1, -1)

            prev_pos = self.prev_pos
            prev_rot = self.prev_rot
            curr_pos = pos_w
            curr_rot = q_wxyz

            prev_rot_expand = prev_rot.unsqueeze(1).expand(-1, dir_flat.shape[1], -1)
            curr_rot_expand = curr_rot.unsqueeze(1).expand(-1, dir_flat.shape[1], -1)
            prev_pos_expand = prev_pos.unsqueeze(1)
            curr_pos_expand = curr_pos.unsqueeze(1)

            p_prev_body = dir_flat * prev_depth_flat.unsqueeze(-1)
            p_prev_world = quat_rotate(prev_rot_expand, p_prev_body) + prev_pos_expand
            p_prev_curr = quat_rotate_inverse(curr_rot_expand, p_prev_world - curr_pos_expand)

            pred_depth = (p_prev_curr * dir_flat).sum(-1)
            residual = curr_depth_flat - pred_depth
            radial_speed = -residual / max(float(self.lidar_radial_dt), 1e-6)

            speed_abs = radial_speed.abs()
            valid = (
                (curr_depth_flat >= self.lidar_radial_min_depth)
                & (curr_depth_flat <= self.lidar_radial_max_depth)
                & (pred_depth > 0.0)
                & (speed_abs >= self.lidar_radial_min_speed)
                & (speed_abs <= self.lidar_radial_max_speed)
            )

            radial_speed = torch.clamp(radial_speed, -self.lidar_radial_max_speed, self.lidar_radial_max_speed)
            radial_norm = torch.clamp(radial_speed / max(self.lidar_radial_max_speed, 1e-6), -1.0, 1.0)
            radial_flat = torch.where(valid, radial_norm, torch.full_like(radial_norm, self.lidar_radial_invalid_value))
            radial_channel = radial_flat.reshape(1, 1, *self.lidar_resolution)

        self.prev_depth_smoothed = depth_smoothed
        self.prev_pos = pos_w
        self.prev_rot = q_wxyz

        lidar_dis_for_flow = torch.where(torch.isfinite(lidar_dis), lidar_dis, torch.full_like(lidar_dis, self.lidar_range))
        scan4flow = (self.lidar_range - lidar_dis_for_flow).clamp(0.0, self.lidar_range) / max(self.lidar_range, 1e-6)
        scan4flow_scaled = torch.nn.functional.interpolate(scan4flow.half() * 255.0, self.dismap_flow_size, mode="bilinear", align_corners=False)

        if self.dismap_image_queue is None:
            self.dismap_image_queue = deque([scan4flow_scaled] * int(self.flow_gap + 3), maxlen=int(self.flow_gap + 3))
        else:
            self.dismap_image_queue.append(scan4flow_scaled)

        dismap_tensor = list(self.dismap_image_queue)
        dismap_image0 = torch.cat(dismap_tensor[:3], dim=1)
        dismap_image1 = torch.cat(dismap_tensor[-3:], dim=1)
        with torch.no_grad():
            dismap_flow = self.flow_est_model(dismap_image0.half(), dismap_image1.half())[-1]

        if self.dismap_flow_queue is None:
            self.dismap_flow_queue = deque([dismap_flow] * int(self.flow_slide_window), maxlen=int(self.flow_slide_window))
        else:
            self.dismap_flow_queue.append(dismap_flow)
        dismap_flow_mean = torch.mean(torch.stack(list(self.dismap_flow_queue)), dim=0)
        flow_zoom = torch.nn.functional.interpolate(dismap_flow_mean.float(), self.lidar_resolution, mode="bilinear", align_corners=False)

        flow_zoom = torch.nan_to_num(flow_zoom.float(), nan=0.0, posinf=0.0, neginf=0.0)
        flow_scaled = torch.cat([(flow_zoom[:, 0:1] / 3.6), (flow_zoom[:, 1:2] / 0.6)], dim=1)
        flow_scaled = torch.clamp(flow_scaled, -1.0, 1.0)

        scan_prox = self.lidar_range - lidar_scan_dis
        scan_normalized = torch.clamp(torch.nan_to_num(scan_prox / max(self.lidar_range, 1e-6), nan=0.0), 0.0, 1.0)

        radial_channel = torch.clamp(
            torch.nan_to_num(
                radial_channel,
                nan=self.lidar_radial_invalid_value,
                posinf=self.lidar_radial_invalid_value,
                neginf=self.lidar_radial_invalid_value,
            ),
            -1.0,
            1.0,
        )

        dismap_stack = torch.cat([scan_normalized, flow_scaled, radial_channel], dim=1)

        rpos = goal_w - pos_w.unsqueeze(1)
        target_dir = rpos / rpos.norm(dim=-1, keepdim=True).clamp(1e-6)
        vel_input = vel_w.unsqueeze(1) / max(self.vel_ref, 1e-6)
        acc_input = self.last_target_acc.unsqueeze(1) / max(self.acc_ref, 1e-6)

        obs = {"state": torch.cat([target_dir, vel_input, acc_input], dim=-1).squeeze(1), "lidar": dismap_stack}

        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": obs, "intrinsics": self.drone_intrinsics_spec_},
                    [self.num_envs],
                ),
                "stats": self.stats.clone(),
            },
            [self.num_envs],
        )
        return td


    def _pcd_cb(self, msg: PointCloud2):
        if self.odom_msg is None or self.goal_msg is None:
            return

        p = self.odom_msg.pose.pose.position
        o = self.odom_msg.pose.pose.orientation
        pos_w = torch.tensor([[p.x, p.y, p.z]], dtype=torch.float32, device=self.device)
        vel = self.odom_msg.twist.twist.linear
        vel_w = torch.tensor([[vel.x, vel.y, vel.z]], dtype=torch.float32, device=self.device)
        q_wxyz = torch.tensor([[o.w, o.x, o.y, o.z]], dtype=torch.float32, device=self.device)

        gx = float(self.goal_msg.pose.position.x)
        gy = float(self.goal_msg.pose.position.y)
        if self.goal_z_mode == "fixed":
            gz = self.goal_z_fixed
        elif self.goal_z_mode == "follow":
            # old behavior (track current altitude)
            gz = float(pos_w[0, 2].item())
        else:
            # "hold": lock to the first observed altitude (or re-lock on new goal)
            if self._z_hold is None:
                self._z_hold = float(pos_w[0, 2].item())
            gz = float(self._z_hold)
        goal_w = torch.tensor([[gx, gy, gz]], dtype=torch.float32, device=self.device).unsqueeze(1)

        pts = self._pcd_to_xyz(msg)
        if pts.shape[0] > self.max_points > 0:
            pts = pts[: self.max_points]

        lidar_dis = self._depth_map_from_pointcloud(pts, pos_w.detach().cpu().numpy().reshape(-1), np.array([o.w, o.x, o.y, o.z], dtype=np.float64))

        td = self._build_observation(lidar_dis, pos_w, vel_w, q_wxyz, goal_w)

        raw_action = self.policy.act_mean(td)
        raw_action = torch.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        raw_action = torch.clamp(raw_action, -1.0, 1.0)
        target_acc = torch.clamp(raw_action * self.acc_ref, -self.acc_ref, self.acc_ref)
        self.last_target_acc = target_acc

        r = (goal_w.squeeze(1) - pos_w).squeeze(0).detach().cpu().numpy()

        # Yaw setpoint strategy (policy does not output yaw)
        if self.yaw_mode == "goal":
            yaw_sp = float(np.arctan2(r[1], r[0]))
        elif self.yaw_mode == "hold":
            yaw_sp = float(_yaw_from_quat_xyzw(o.x, o.y, o.z, o.w))
        elif self.yaw_mode == "fixed":
            yaw_sp = float(self.fixed_yaw)
        else:
            yaw_sp = float(np.arctan2(r[1], r[0]))

        acc_cmd_w = target_acc.squeeze(0).detach().cpu().numpy()
        
        acc_cmd_w = self._limit_tilt(acc_cmd_w)
        
        q_cmd_wxyz, thrust_scale = _attitude_from_accel_and_yaw(acc_cmd_w, yaw_sp, self.g)

        # thrust = float(self.hover_thrust * thrust_scale)
        # thrust = float(np.clip(thrust, self.thrust_min, self.thrust_max))

        now = time.time()
        dt = 0.02 if self._last_cmd_time <= 0.0 else max(1e-3, now - self._last_cmd_time)

        thrust_cmd = float(self.hover_thrust * thrust_scale)

        # small I-term on vertical speed to compensate hover mismatch (battery/weight)
        if self.hover_i_gain > 0.0:
            vz = float(vel_w[0, 2].item())
            self._hover_i += self.hover_i_gain * (-vz) * dt
            self._hover_i = float(np.clip(self._hover_i, -self.hover_i_limit, self.hover_i_limit))
            thrust_cmd += self._hover_i

        thrust = float(np.clip(thrust_cmd, self.thrust_min, self.thrust_max))


        self._last_q_wxyz = q_cmd_wxyz
        self._last_thrust = thrust
        self._last_cmd_time = time.time()
        
        
    def _limit_tilt(self, acc_cmd_w: np.ndarray) -> np.ndarray:
        # limit tilt by scaling horizontal accel so that tilt <= max_tilt_deg
        theta = np.deg2rad(self.max_tilt_deg)
        a_z_total = self.g + float(acc_cmd_w[2])
        a_z_total = max(a_z_total, 0.5)  # avoid near-zero / negative
        max_xy = a_z_total * np.tan(theta)

        xy = acc_cmd_w[:2].astype(np.float64)
        xy_norm = float(np.linalg.norm(xy))
        if xy_norm > max_xy:
            acc_cmd_w[:2] = (xy * (max_xy / (xy_norm + 1e-6))).astype(np.float64)
        return acc_cmd_w


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(cfg):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except Exception:
        pass
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    rospy.init_node("infer_ros")
    _ = InferROS(cfg)
    rospy.spin()


if __name__ == "__main__":
    main()
