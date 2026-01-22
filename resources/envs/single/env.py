import torch
import torch.nn.functional as F
import torch.distributions as D
import einops
from collections import deque
from . import env_utils

from resources.envs.isaac_env import AgentSpec, IsaacEnv
from resources.robots.drone import MultirotorBase
from resources.utils.torch import (
    euler_to_quaternion,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_euler,
)
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from resources.NeuFlow_v2.infer_lidar import init_neuflow
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.lab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg


class Env(IsaacEnv):
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.bound_h = cfg.task.bound_h

        super().__init__(cfg, headless)

        self.lidar._initialize_impl()

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        self.safety_dis = 0.3
        self.vel_min = 3.5
        self.vel_max = 5.
        self.acc_max = 10.
        self.virtual_ground = 0.5
        self.virtual_ceiling = 3.5
        self.height_bound = 0.5

        # allow overriding constraints from cfg.task (useful for curriculum / inference)
        try:
            task_cfg = getattr(self.cfg, "task", None)
            if task_cfg is not None and hasattr(task_cfg, "get"):
                self.safety_dis = float(task_cfg.get("safety_dis", self.safety_dis))
                self.vel_min = float(task_cfg.get("vel_min", self.vel_min))
                self.vel_max = float(task_cfg.get("vel_max", self.vel_max))
                self.acc_max = float(task_cfg.get("acc_max", self.acc_max))
                self.virtual_ground = float(task_cfg.get("virtual_ground", self.virtual_ground))
                self.virtual_ceiling = float(task_cfg.get("virtual_ceiling", self.virtual_ceiling))
                self.height_bound = float(task_cfg.get("height_bound", self.height_bound))
        except Exception:
            # keep defaults
            pass

        # derived


        self.start_pos = None
        self.target_pos = None
        self.dismap_flow_size = None
        self.actions = None
        self.last_dis2goal = None
        self.last_acc = None
        # Per-env validity mask for jerk baseline. We want to avoid penalizing jerk across episode
        # boundaries for envs that have just been reset.
        self.last_acc_valid = None
        self.dismap_image_queue = None
        self.dismap_flow_queue = None
        self.depth_image_queue = None
        self.prev_depth_smoothed = None
        self.prev_pos = None
        self.prev_rot = None
        self.risk_depth = None
        self.risk_age = None
        self.set_dobs_state = None
        self.set_wall_state = None
        self.ray_hits_dir = None
        self.input_dir = None
        self.error_tolerance = None
        self.flow_est_model = None
        self.dismap_flow = None
        self.radial_channel = None  # cached [N,1,H,W] normalized radial speed for risk reward
        self.trace_prob = None
        self.virtual_x_bound = None
        self.reward_dobs_max = None
        self.speed_sum = None
        self.acc_sum = None
        from types import SimpleNamespace

        # --- reward_cfg default / compatibility ---
        # Old P2M does not use reward_cfg, but some newer experiments add it.
        # Keep it always defined to avoid AttributeError when accessing self.reward_cfg.
        # Also support hydra "dotted-key" style like `reward_cfg.k_s: 5.0`.
        from types import SimpleNamespace

        # Default
        self.reward_cfg = SimpleNamespace(k_s=1.0)

        # Try to load from cfg.task.reward_cfg (nested style)
        rcfg = None
        try:
            rcfg = cfg.task.get("reward_cfg", None)
        except Exception:
            rcfg = getattr(getattr(cfg, "task", None), "reward_cfg", None)

        # If still None, try dotted key style
        if rcfg is None:
            try:
                k_s = cfg.task.get("reward_cfg.k_s", None)
            except Exception:
                k_s = None
                try:
                    task_obj = getattr(cfg, "task", None)
                    if task_obj is not None and hasattr(task_obj, "get"):
                        k_s = task_obj.get("reward_cfg.k_s", None)
                except Exception:
                    k_s = None
            if k_s is not None:
                rcfg = {"k_s": float(k_s)}

        # Apply
        if rcfg is not None:
            if isinstance(rcfg, dict):
                self.reward_cfg = SimpleNamespace(**rcfg)
            else:
                self.reward_cfg = rcfg


    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import AssetBaseCfg
        from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
        from omni.isaac.lab.terrains import (
            TerrainImporterCfg,
            TerrainImporter,
            TerrainGeneratorCfg,
            HfDiscreteObstaclesTerrainCfg,
        )
        # from omni.isaac.lab.utils.assets import NVIDIA_NUCLEUS_DIR

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        rot = euler_to_quaternion(torch.tensor([0.0, 0.1, 0.1]))
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos, rot)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        self.seed = 10

        # ---------------------------------------------------------------------
        # Static obstacles / terrain sizing
        #
        # IMPORTANT: HfDiscreteObstaclesTerrainCfg(num_obstacles=K) places K obstacles PER TILE.
        # The original code used num_rows=num_cols=6 (36 tiles), so total obstacles ~= 36*K,
        # easily creating an obstacle forest.
        #
        # Fix: generate ONE large tile (same total area) and interpret obstacles as TOTAL count:
        #   static_obs_num_total (capped by static_obs_max_total)
        #
        # Keep legacy key for backward compatibility:
        #   static_obs_num_per_grid (if static_obs_num_total is absent, treat it as TOTAL fallback)
        # ---------------------------------------------------------------------
        try:
            self.static_obs_num_per_grid = int(getattr(self.cfg.task, "static_obs_num_per_grid"))
        except Exception:
            # legacy typo key
            try:
                self.static_obs_num_per_grid = int(getattr(self.cfg.task, "static_obs_num_per_gird"))
            except Exception:
                self.static_obs_num_per_grid = 0

        self.static_obs_num_total = int(getattr(self.cfg.task, "static_obs_num_total", 0))
        self.static_obs_max_total = int(getattr(self.cfg.task, "static_obs_max_total", 120))

        _wr = getattr(self.cfg.task, "static_obs_width_range", [0.5, 0.9])
        try:
            self.static_obs_width_range = (float(_wr[0]), float(_wr[1]))
        except Exception:
            self.static_obs_width_range = (0.5, 0.9)

        # terrain total size: tile_size * (num_rows/num_cols) (defaults keep legacy 36m x 36m)
        self.terrain_tile_size = float(getattr(self.cfg.task, "terrain_tile_size", 6.0))
        self.terrain_num_rows = int(getattr(self.cfg.task, "terrain_num_rows", 6))
        self.terrain_num_cols = int(getattr(self.cfg.task, "terrain_num_cols", 6))
        self.terrain_border_width = float(getattr(self.cfg.task, "terrain_border_width", 2.0))

        self.terrain_num_rows = max(1, self.terrain_num_rows)
        self.terrain_num_cols = max(1, self.terrain_num_cols)
        self.terrain_total_size = (
            float(self.terrain_tile_size * self.terrain_num_rows),
            float(self.terrain_tile_size * self.terrain_num_cols),
        )

        # default reserved open area width (clamped later against total size)
        # self.static_obs_platform_width = float(getattr(self.cfg.task, "static_obs_platform_width", self.static_obs_platform_width))
        self.static_obs_platform_width = float(self.cfg.task.get("static_obs_platform_width", 14.0))
)

        # obstacle height configuration (from cfg.task)
        try:
            _hr = self.cfg.task.get("static_obs_height_range", [3.5, 4.0])
        except Exception:
            _hr = [3.5, 4.0]
        try:
            static_obs_height_range = (float(_hr[0]), float(_hr[1]))
        except Exception:
            static_obs_height_range = (3.5, 4.0)
        self.static_obs_height_range = static_obs_height_range

        try:
            self.dobs_height = float(self.cfg.task.get("dobs_height", 4.0))
        except Exception:
            self.dobs_height = 4.0

        # ---------------------------------------------------------------------
        # Terrain (static obstacles) - success curriculum (global)
        # We cannot change heightfield obstacle count at runtime. Instead, we
        # pre-generate one ground per curriculum level and switch by moving Z.
        # ---------------------------------------------------------------------
        sc_cfg = getattr(self.cfg.task, "success_curriculum", None)
        self._curriculum_enabled = bool(getattr(sc_cfg, "enable", False)) if sc_cfg is not None else False
        self._success_curriculum_levels = []
        if self._curriculum_enabled:
            self._success_curriculum_levels = list(getattr(sc_cfg, "levels", []))
            if len(self._success_curriculum_levels) == 0:
                self._curriculum_enabled = False

        self._curriculum_level = int(getattr(sc_cfg, "start_level", 0)) if self._curriculum_enabled else 0
        self._curriculum_level = max(
            0,
            min(
                self._curriculum_level,
                (len(self._success_curriculum_levels) - 1) if self._curriculum_enabled else 0,
            ),
        )

        self._ground_prim_paths = []
        self._terrains = []
        num_levels = len(self._success_curriculum_levels) if self._curriculum_enabled else 1

        for lvl in range(num_levels):
            level_cfg = self._success_curriculum_levels[lvl] if self._curriculum_enabled else {}
            # Decide TOTAL number of static obstacles for this level.
            # Priority: per-level -> global -> legacy fallback
            try:
                num_obs_total = int(level_cfg.get("static_obs_num_total", self.static_obs_num_total)) if isinstance(level_cfg, dict) else int(self.static_obs_num_total)
            except Exception:
                num_obs_total = int(self.static_obs_num_total)

            if num_obs_total <= 0:
                # legacy fallback: treat per_grid as TOTAL if provided (to avoid breaking older configs)
                num_obs_total = int(self.static_obs_num_per_grid)

            # Safety cap (TOTAL)
            try:
                max_total = int(level_cfg.get("static_obs_max_total", self.static_obs_max_total)) if isinstance(level_cfg, dict) else int(self.static_obs_max_total)
            except Exception:
                max_total = int(self.static_obs_max_total)
            max_total = max(0, int(max_total))
            num_obs = max(0, min(int(num_obs_total), int(max_total)))

            # Optional: allow per-level platform width to reserve a free corridor
# Optional: allow per-level platform width to reserve a free corridor
            try:
                platform_width = float(level_cfg.get("static_obs_platform_width",
                                      getattr(self.cfg.task, "static_obs_platform_width", self.static_obs_platform_width)))
            except Exception:
                platform_width = float(getattr(self.cfg.task, "static_obs_platform_width", self.static_obs_platform_width))
            platform_width = max(0.0, platform_width)
            # Clamp against available inner width to avoid invalid terrain generation.
            # inner_size = min(total_size) - 2*border_width
            bw = float(getattr(self.cfg.task, "terrain_border_width", self.terrain_border_width))
            bw = max(0.0, bw)
            inner = max(0.0, min(self.terrain_total_size[0], self.terrain_total_size[1]) - 2.0 * bw)
            if inner > 0.0:
                platform_width = min(platform_width, max(0.0, inner - 0.1))

            prim_path = "/World/ground" if lvl == 0 else f"/World/ground_l{lvl}"
            terrain_cfg = TerrainImporterCfg(
                prim_path=prim_path,
                terrain_type="generator",
                terrain_generator=TerrainGeneratorCfg(
                    seed=self.seed + lvl,
                    size=(self.terrain_total_size[0], self.terrain_total_size[1]),
                    border_width=min(max(self.terrain_border_width, 0.0), 0.5*min(self.terrain_total_size[0], self.terrain_total_size[1]) - 0.1),
                    num_rows=1,
                    num_cols=1,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    slope_threshold=0.75,
                    use_cache=False,
                    curriculum=False,
                    sub_terrains={
                        "obstacles": HfDiscreteObstaclesTerrainCfg(
                            size=(self.terrain_total_size[0], self.terrain_total_size[1]),
                            horizontal_scale=0.1,
                            vertical_scale=0.1,
                            border_width=0.0,
                            num_obstacles=num_obs,
                            obstacle_height_mode="fixed",
                            obstacle_width_range=self.static_obs_width_range,
                            obstacle_height_range=(self.static_obs_height_range[0], self.static_obs_height_range[1]),
                            platform_width=platform_width,
                        ),
                    },
                ),
                num_envs=self.num_envs,
                max_init_terrain_level=5,
                collision_group=-1,
                debug_vis=False,
            )
            terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)
            self._terrains.append(terrain)
            self._ground_prim_paths.append(prim_path)

        # curriculum switch handshake
        self._pending_curriculum_level = None
        self._force_curriculum_reset = False

        # Activate selected ground at z=0; park the rest far below.
        self._apply_ground_level(self._curriculum_level)

        # dynamic obstacles (support disabling with dynamic_obs_num=0)
        self.dynamic_obs_num = int(self.cfg.task.dynamic_obs_num)
        self.dobs_pos_x_range = (-18.0, 18.0)
        self.dobs_pos_y_range = (-18.0, 18.0)
        self.dobs_vel_range = (1.0, 5.0)
        self.dobs_rad_range = (0.25, 0.45)
        if self.dynamic_obs_num <= 0:
            self.dynamic_obs_num = 0
            self.dobs_states = torch.zeros((0, 3, 2), device=self.device)
            self.dobs_origins = torch.zeros((0, 2), device=self.device)
            self.dobs_rad = torch.zeros((0,), device=self.device)
            self.dobs = None
        else:
            _dobs_states_np = env_utils.generate_obstacle_tensor(
                self.dynamic_obs_num,
                self.dobs_pos_x_range,
                self.dobs_pos_y_range,
                self.dobs_vel_range,
                self.dobs_rad_range,
                self.seed,
            )
            self.dobs_states = torch.tensor(_dobs_states_np, device=self.device)
            self.dobs_origins = self.dobs_states[:, 0]
            self.dobs_rad = self.dobs_states[:, 2][:, 0]
            dobs_cfg_dict = {}
            for i, origin in enumerate(self.dobs_origins):
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/moving_obs{i}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius=float(self.dobs_rad[i].item()),
                        height=self.dobs_height,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=False
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dobs_cfg_dict[f"Cylinder_{i}"] = cylinder_cfg
            cylinder_collection_cfg = RigidObjectCollectionCfg(rigid_objects=dobs_cfg_dict)
            self.dobs = RigidObjectCollection(cfg=cylinder_collection_cfg)
        # Dynamic obstacles curriculum: pre-generate `dynamic_obs_num` and only
        # activate a prefix of them (global curriculum).
        self._dobs_active_num = int(self.dynamic_obs_num)
        if getattr(self, "_curriculum_enabled", False) and len(getattr(self, "_success_curriculum_levels", [])) > 0:
            try:
                level_cfg = self._success_curriculum_levels[int(getattr(self, "_curriculum_level", 0))]
                self._dobs_active_num = int(level_cfg.get("dynamic_obs_active", self.dynamic_obs_num))
            except Exception:
                self._dobs_active_num = int(self.dynamic_obs_num)
        self._dobs_active_num = max(0, min(int(self._dobs_active_num), int(self.dynamic_obs_num)))

        self.wall_num = 1
        self.wall_width = 20
        self.wall_height = 0.4
        self.fly_height = 2.0
        self.wall_states = env_utils.generate_wall_tensor(self.wall_num, self.wall_width, 
                                                          self.wall_height, self.fly_height)
        self.wall_origins = self.wall_states[..., :3]
        self.wall_sizes = self.wall_states[..., 3:]
        wall_cfg_dict = {}
        for i, origin in enumerate(self.wall_origins):
            for j in range(4):
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Wall{i}/Cuboid{j}",
                    spawn=sim_utils.CuboidCfg(
                        size=self.wall_sizes[i, j, :],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=False),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg()
                )
                wall_cfg_dict[f"Wall_{4*i+j}"] = cuboid_cfg
        wall_collection_cfg = RigidObjectCollectionCfg(rigid_objects = wall_cfg_dict)
        self.wall = RigidObjectCollection(cfg = wall_collection_cfg)
        self.wall_states = torch.tensor(self.wall_states, device=self.device)
        self.wall_origins = torch.tensor(self.wall_origins, device=self.device)
        self.wall_sizes = torch.tensor(self.wall_sizes, device=self.device)

        self.lidar_hfov = self.cfg.task.lidar_hfov
        self.lidar_vfov = (
            max(-89., self.cfg.task.lidar_vfov[0]),
            min(89., self.cfg.task.lidar_vfov[1])
        )
        self.lidar_range = self.cfg.task.lidar_range
        self.lidar_h_res = self.cfg.task.lidar_h_res
        self.lidar_v_res = self.cfg.task.lidar_v_res
        self.lidar_h_sample = self.cfg.task.lidar_h_sample
        self.lidar_v_sample = self.cfg.task.lidar_v_sample
        self.lidar_use_height_filter = self.cfg.task.get("lidar_use_height_filter", True)
        self.lidar_radial_window = int(self.cfg.task.get("lidar_radial_window", 3))
        self.lidar_radial_min_depth = float(self.cfg.task.get("lidar_radial_min_depth", 0.1))
        self.lidar_radial_max_depth = float(self.cfg.task.get("lidar_radial_max_depth", self.lidar_range))
        self.lidar_radial_min_speed = float(self.cfg.task.get("lidar_radial_min_speed", 0.0))
        self.lidar_radial_max_speed = float(self.cfg.task.get("lidar_radial_max_speed", 10.0))
        self.lidar_radial_invalid_value = float(self.cfg.task.get("lidar_radial_invalid_value", 0.0))
        self.lidar_radial_dt = float(self.cfg.task.get("lidar_radial_dt", self.dt))
        self.lidar_risk_decay_tau = float(self.cfg.task.get("lidar_risk_decay_tau", 1.0))
        self.lidar_risk_clear_time = float(self.cfg.task.get("lidar_risk_clear_time", 3.0))
        self.lidar_risk_clear_margin = float(self.cfg.task.get("lidar_risk_clear_margin", 1.0))
        self.lidar_risk_enable = bool(self.cfg.task.get("lidar_risk_enable", True))

        # NOTE(compat): IsaacLab RayCaster currently supports **exactly one** mesh prim.
        # In this repo we may generate multiple terrain variants (one per curriculum level).
        # We therefore bind the lidar to the *active* ground mesh only, and when the
        # curriculum switches level we re-create the RayCaster with the new mesh prim.
        active_ground_prim = getattr(self, "_active_ground_prim_path", None)
        if active_ground_prim is None:
            active_ground_prim = getattr(self, "_ground_prim_paths", ["/World/ground"])[0]

        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res = self.lidar_hfov / (self.lidar_h_res * self.lidar_h_sample),
                vertical_ray_angles = torch.linspace(*self.lidar_vfov, self.lidar_v_res * self.lidar_v_sample)
            ),
            debug_vis=False,
            mesh_prim_paths=[active_ground_prim]
        )
        # Keep a copy for curriculum-time re-init.
        self._lidar_cfg = ray_caster_cfg
        self.lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)
        return getattr(self, "_ground_prim_paths", ["/World/ground"])

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 9
        self.lidar_resolution = (self.lidar_h_res, self.lidar_v_res)

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
                    "lidar": UnboundedContinuousTensorSpec((4, self.lidar_resolution[0], self.lidar_resolution[1]), device=self.device),
                }),
                "intrinsics": self.drone.intrinsics_spec.to(self.device)
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec,
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "reward_velocity": UnboundedContinuousTensorSpec(1),
            "reward_acceleration": UnboundedContinuousTensorSpec(1),
            "reward_jerk": UnboundedContinuousTensorSpec(1),
            "reward_height": UnboundedContinuousTensorSpec(1),
            "reward_goal": UnboundedContinuousTensorSpec(1),
            "reward_safety": UnboundedContinuousTensorSpec(1),
            "reward_dobs": UnboundedContinuousTensorSpec(1),
            "reward_risk": UnboundedContinuousTensorSpec(1),
            "reward_collision": UnboundedContinuousTensorSpec(1),
            "risk_smax": UnboundedContinuousTensorSpec(1),
            "goal_gate": UnboundedContinuousTensorSpec(1),
            "plan_success": UnboundedContinuousTensorSpec(1),
            "flight_success": UnboundedContinuousTensorSpec(1),

            # Global success curriculum bookkeeping
            "curriculum_level": UnboundedContinuousTensorSpec(1),
            "curriculum_static_obs_num_per_grid": UnboundedContinuousTensorSpec(1),
            "curriculum_dobs_active": UnboundedContinuousTensorSpec(1),
            "curriculum_reset": UnboundedContinuousTensorSpec(1),
            "avg_speed": UnboundedContinuousTensorSpec(1),
            "max_speed": UnboundedContinuousTensorSpec(1),
            "avg_acc": UnboundedContinuousTensorSpec(1),
            "max_acc": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "terminated": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "done_any": UnboundedContinuousTensorSpec(1),
            "done_success": UnboundedContinuousTensorSpec(1),
            "done_timeout": UnboundedContinuousTensorSpec(1),
            "done_safety": UnboundedContinuousTensorSpec(1),
            "done_height_low": UnboundedContinuousTensorSpec(1),
            "done_height_high": UnboundedContinuousTensorSpec(1),
            "done_bound": UnboundedContinuousTensorSpec(1),
            "done_vel_limit": UnboundedContinuousTensorSpec(1),
            "done_acc_limit": UnboundedContinuousTensorSpec(1),
            "done_nan": UnboundedContinuousTensorSpec(1),
            "done_other": UnboundedContinuousTensorSpec(1),
            "done_rate": UnboundedContinuousTensorSpec(1),
            "done_ratio_success": UnboundedContinuousTensorSpec(1),
            "done_ratio_timeout": UnboundedContinuousTensorSpec(1),
            "done_ratio_safety": UnboundedContinuousTensorSpec(1),
            "done_ratio_height_low": UnboundedContinuousTensorSpec(1),
            "done_ratio_height_high": UnboundedContinuousTensorSpec(1),
            "done_ratio_bound": UnboundedContinuousTensorSpec(1),
            "done_ratio_vel_limit": UnboundedContinuousTensorSpec(1),
            "done_ratio_acc_limit": UnboundedContinuousTensorSpec(1),
            "done_ratio_nan": UnboundedContinuousTensorSpec(1),
            "done_ratio_other": UnboundedContinuousTensorSpec(1)

        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec 
        self.stats = stats_spec.zero()

    # ---------------------------------------------------------------------
    # Global success curriculum API
    # ---------------------------------------------------------------------
    def request_curriculum_level(self, new_level: int):
        """Request a global curriculum level switch.

        The env will force all env instances to `done` on the next step, and will
        only apply the terrain/dobs switch during the subsequent *global* reset.
        """
        if not getattr(self, "_curriculum_enabled", False):
            return
        max_lvl = max(0, len(getattr(self, "_success_curriculum_levels", [])) - 1)
        new_level = int(max(0, min(int(new_level), max_lvl)))
        if new_level == int(getattr(self, "_curriculum_level", 0)):
            return
        self._pending_curriculum_level = new_level
        self._force_curriculum_reset = True

    def _apply_curriculum_level(self, level: int):
        """Apply curriculum switch (called during global reset)."""
        if not getattr(self, "_curriculum_enabled", False):
            return
        level = int(level)
        self._curriculum_level = level

        # Switch terrain by moving active ground to z=0.
        prev_ground = getattr(self, "_active_ground_prim_path", None)
        self._apply_ground_level(level)
        new_ground = getattr(self, "_active_ground_prim_path", None)

        # Re-bind lidar to the new ground mesh.
        # (RayCaster supports only one mesh prim, so we must rebuild on switch.)
        if new_ground is not None and prev_ground != new_ground and hasattr(self, "_lidar_cfg"):
            try:
                self._lidar_cfg.mesh_prim_paths = [new_ground]
                # Drop old sensor so its weakref callbacks can GC safely.
                self.lidar = None
                self.lidar = self._lidar_cfg.class_type(self._lidar_cfg)
                # Force immediate init (same pattern as __init__).
                if hasattr(self.lidar, "_initialize_impl"):
                    self.lidar._initialize_impl()
            except Exception as e:
                print(f"[WARN] Failed to re-create lidar on curriculum switch: {e}")

        # Update active dynamic obstacles count.
        try:
            cfg = self._success_curriculum_levels[level]
            self._dobs_active_num = int(cfg.get("dynamic_obs_active", self.dynamic_obs_num))
        except Exception:
            self._dobs_active_num = int(getattr(self, "dynamic_obs_num", 0))
        self._dobs_active_num = max(0, min(int(self._dobs_active_num), int(getattr(self, "dynamic_obs_num", 0))))

        # Immediately park inactive obstacles so the curriculum switch is visually correct.
        if getattr(self, "dobs", None) is not None and getattr(self, "set_dobs_state", None) is not None:
            active = int(self._dobs_active_num)
            total = int(getattr(self, "dynamic_obs_num", 0))
            if active < total:
                self.set_dobs_state[active:, :2] = 1.0e6
                self.set_dobs_state[active:, 2] = self.dobs_height / 2
                try:
                    self.dobs.write_object_link_pose_to_sim(self.set_dobs_state[..., :7])
                except Exception:
                    pass
        self._dobs_active_num = max(0, min(int(self._dobs_active_num), int(getattr(self, "dynamic_obs_num", 0))))

        # Mirror static obs count for logging (terrain itself is already generated).
        try:
            cfg = self._success_curriculum_levels[level]
            self.static_obs_num_per_grid = int(cfg.get("static_obs_num_per_grid", self.static_obs_num_per_grid))
        except Exception:
            pass

    def _apply_ground_level(self, active_level: int):
        """Move the active terrain to z=0 and park the rest far below."""
        paths = getattr(self, "_ground_prim_paths", None)
        if not paths:
            return

        active_level = int(active_level)
        self._active_ground_idx = active_level
        self._active_ground_prim_path = paths[active_level]

        # Always use UsdGeom Xform ops (do not rely on set_prim_property), otherwise
        # Hydra/scene delegate may warn that xformOp:translate does not exist.
        try:
            from omni.isaac.core.utils.prims import get_prim_at_path
            from pxr import UsdGeom, Gf

            for i, p in enumerate(paths):
                z = 0.0 if i == active_level else -1000.0 * (i + 1)
                prim = get_prim_at_path(p)
                xf = UsdGeom.Xformable(prim)
                # Reuse translate op if present; otherwise add one.
                tr_op = None
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        tr_op = op
                        break
                if tr_op is None:
                    tr_op = xf.AddTranslateOp()
                tr_op.Set(Gf.Vec3d(0.0, 0.0, float(z)))
        except Exception:
            # Best-effort; if translate ops fail, we will just keep terrains as-is.
            pass

    def _reset_idx(self, env_ids: torch.Tensor):
        # Apply pending curriculum only when we are resetting *all* envs.
        if getattr(self, "_pending_curriculum_level", None) is not None:
            if int(env_ids.numel()) == int(self.num_envs):
                self._apply_curriculum_level(int(self._pending_curriculum_level))
                self._pending_curriculum_level = None
                self._force_curriculum_reset = False

        self.drone._reset_idx(env_ids, self.training)

        if self.speed_sum is None:
            self.speed_sum = torch.zeros(self.num_envs, 1, device=self.device)
        if self.acc_sum is None:
            self.acc_sum = torch.zeros(self.num_envs, 1, device=self.device)
        self.speed_sum[env_ids] = 0.0
        self.acc_sum[env_ids] = 0.0

        if (self.start_pos is None) & (self.target_pos is None):
            drones_per_side = self.cfg.env.num_envs // 4
            in_max = 20
            out_max = 44
            offset = 12
            left_vals = torch.linspace(-0.5, 0, int(drones_per_side/2), device=self.device) * in_max
            right_vals = torch.linspace(0, 0.5, int(drones_per_side/2), device=self.device) * in_max
            vals = torch.cat([right_vals, left_vals], dim=0)

            self.start_pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            self.start_pos[:drones_per_side, 0, 0] = vals
            self.start_pos[:drones_per_side, 0, 1] = - out_max/2
            self.start_pos[drones_per_side:2*drones_per_side, 0, 0] = out_max/2
            self.start_pos[drones_per_side:2*drones_per_side, 0, 1] = vals
            self.start_pos[2*drones_per_side:3*drones_per_side, 0, 0] = - out_max/2
            self.start_pos[2*drones_per_side:3*drones_per_side, 0, 1] = vals
            self.start_pos[3*drones_per_side:, 0, 0] = vals
            self.start_pos[3*drones_per_side:, 0, 1] = out_max/2
            self.start_pos[:, 0, 2] = self.fly_height

            self.target_pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            self.target_pos[:drones_per_side, 0, 0] = - vals  
            self.target_pos[:drones_per_side, 0, 1] = out_max/2 - offset
            self.target_pos[drones_per_side:2*drones_per_side, 0, 0] = - out_max/2 + offset  
            self.target_pos[drones_per_side:2*drones_per_side, 0, 1] = - vals
            self.target_pos[2*drones_per_side:3*drones_per_side, 0, 0] = out_max/2 - offset  
            self.target_pos[2*drones_per_side:3*drones_per_side, 0, 1] = - vals 
            self.target_pos[3*drones_per_side:, 0, 0] = - vals
            self.target_pos[3*drones_per_side:, 0, 1] = - out_max/2 + offset
            self.target_pos[:, 0, 2] = self.fly_height

        pos = self.start_pos[env_ids]
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos, rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.stats[env_ids] = 0.
        # --- IMPORTANT: clear per-episode sensor/history states for the reset envs.
        # If not cleared, "risk depth decay" + radial estimation can carry stale values across resets,
        # causing immediate unsafe/NaN and repeated resets.
        if self.risk_depth is not None:
            self.risk_depth[env_ids] = self.lidar_range + self.lidar_risk_clear_margin
        if self.risk_age is not None:
            self.risk_age[env_ids] = 0.0

        # Clear temporal buffers (they store tensors over *all* envs). Re-init next step.
        self.depth_image_queue = None
        self.prev_depth_smoothed = None
        self.prev_pos = None
        self.prev_rot = None

        # Flow history (optional; comment out if you want continuity across episodes)
        self.dismap_image_queue = None
        self.dismap_flow_queue = None

        # Reset last distance-to-goal baseline ONLY for envs that are being reset.
        # IMPORTANT: do NOT overwrite all envs here, otherwise non-reset envs can get a huge
        #            positive delta (last_dis2goal - dis2goal) and reward_goal will explode.
        if (self.start_pos is not None) and (self.target_pos is not None):
            if self.last_dis2goal is None:
                self.last_dis2goal = torch.zeros((self.num_envs, 1), device=self.device)
            elif self.last_dis2goal.ndim == 1:
                self.last_dis2goal = self.last_dis2goal.view(self.num_envs, 1)
            init_dis = (self.target_pos[env_ids] - self.start_pos[env_ids]).norm(dim=-1)
            if init_dis.ndim == 1:
                init_dis = init_dis.view(-1, 1)
            self.last_dis2goal[env_ids] = init_dis


        # Mark jerk baseline invalid for just-reset envs. This avoids penalizing jerk across
        # episode boundaries in vectorized training.
        if self.last_acc_valid is None:
            self.last_acc_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            self.last_acc_valid = self.last_acc_valid.view(-1)
        self.last_acc_valid[env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        # defensive: ensure acc_min exists (older configs / patches)
        if not hasattr(self, 'acc_min'):
            self.acc_min = -self.acc_max
        # cache drone state once per step to avoid repeated sim queries
        drone_state_pre = self.drone.get_state(env_frame=False)
        drone_pos_xy = drone_state_pre[..., :2]
        if drone_pos_xy.ndim == 3 and drone_pos_xy.shape[1] == 1:
            drone_pos_xy = drone_pos_xy.squeeze(1)
        if (getattr(self, "dynamic_obs_num", 0) > 0) and (self.dobs is not None):
            if self.set_dobs_state is None:
                self.set_dobs_state = self.dobs.data.default_object_state.clone()
                # Start by placing all obstacles at their generated origins...
                self.set_dobs_state[..., :2] = self.dobs_origins
                self.set_dobs_state[..., 2] = self.dobs_height / 2

                # ...then immediately park INACTIVE obstacles far away so the rendered scene
                # matches the curriculum setting from the very first frame.
                active = int(getattr(self, "_dobs_active_num", self.dynamic_obs_num))
                active = max(0, min(active, int(self.dynamic_obs_num)))
                if active < int(self.dynamic_obs_num):
                    self.set_dobs_state[active:, :2] = 1.0e6
                    self.set_dobs_state[active:, 2] = self.dobs_height / 2
                    self.dobs_states[active:, 0, 0] = 1.0e6
                    self.dobs_states[active:, 0, 1] = 1.0e6
                    self.dobs_states[active:, 1].zero_()
            else:
                if self.trace_prob is None:
                    self.trace_prob = self.cfg.task.trace_prob

                active = int(getattr(self, "_dobs_active_num", self.dynamic_obs_num))
                active = max(0, min(active, int(self.dynamic_obs_num)))
                self._dobs_active_num = active

                if active > 0:
                    self.dobs_states[:active, 1] = env_utils.update_dobs_vel(
                        self.device,
                        self.dobs_states[:active],
                        self.dobs_pos_x_range,
                        self.dobs_pos_y_range,
                        self.drone,
                        self.trace_prob,
                        drone_pos_xy=drone_pos_xy,
                    )
                    set_dobs_vel = self.dobs_states[:active, 1]
                    self.set_dobs_state[:active, :2] = self.dobs_states[:active, 0] + (set_dobs_vel * self.dt)
                    self.set_dobs_state[:active, 2] = self.dobs_height / 2
                    self.dobs_states[:active, 0] = self.set_dobs_state[:active, :2]

                # Park inactive obstacles far away
                if active < int(self.dynamic_obs_num):
                    self.dobs_states[active:, 1].zero_()
                    self.dobs_states[active:, 0, 0] = 1.0e6
                    self.dobs_states[active:, 0, 1] = 1.0e6
                    self.set_dobs_state[active:, :2] = 1.0e6
                    self.set_dobs_state[active:, 2] = self.dobs_height / 2

            self.dobs.write_object_link_pose_to_sim(self.set_dobs_state[..., :7])

        if self.set_wall_state is None:
            self.set_wall_state = self.wall.data.default_object_state.clone()
            self.set_wall_state[..., :3] = self.wall_origins.reshape(self.wall_num * 4, 3)
            self.wall.write_object_link_pose_to_sim(self.set_wall_state[..., :7])
        # P2M-style action mapping: use raw policy action directly as target_acc input (no scaling/clamp).
        raw_action = tensordict[("agents", "action")]  # shape: [num_envs, 3]
        raw_action = torch.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        target_acc = raw_action  # P2M behavior: no scaling
        # keep both if you need debug/reward
        self.actions = target_acc

        ego_drone_state = drone_state_pre[..., :13].squeeze(0)
        unit_thrust = self.controller(ego_drone_state, target_acc.unsqueeze(1), None, False)
        self.effort = self.drone.apply_action(unit_thrust)


    def _post_sim_step(self, tensordict: TensorDictBase):
        # NOTE: IsaacEnv already increments `self._global_frame_count` by `num_envs` each step.
        # Do not increment it here to avoid double-counting.
        # optional lidar update down-sampling (cfg: task.lidar_update_period, default=1)
        try:
            period = int(self.cfg.task.get("lidar_update_period", 1))
        except Exception:
            period = 1
        if period <= 1:
            self.lidar.update(self.dt)
        else:
            # update every N steps; pass accumulated dt to keep time-consistency
            if not hasattr(self, "_lidar_update_counter"):
                self._lidar_update_counter = 0
            self._lidar_update_counter += 1
            if (self._lidar_update_counter % period) == 0:
                self.lidar.update(self.dt * float(period))

    def _compute_state_and_obs(self):
        # Keep locals defined to avoid UnboundLocalError in optional branches / debug.
        dismap_image0 = None

        self.drone_state = self.drone.get_state(env_frame=False)
        self.rpos = self.target_pos - self.drone_state[..., :3]

        if (self.ray_hits_dir is None) & (self.input_dir is None):
            self.ray_hits_dir = env_utils.compute_rayhitsdir(self.device, self.num_envs, self.lidar_hfov, self.lidar_vfov,
                                                         self.lidar_h_res * self.lidar_h_sample,
                                                         self.lidar_v_res * self.lidar_v_sample)
            self.input_dir = env_utils.compute_rayhitsdir(self.device, self.num_envs, self.lidar_hfov, self.lidar_vfov,
                                                      self.lidar_h_res, self.lidar_v_res)

        # Rotate sensor-frame ray directions to world frame (yaw-only), to align with RayCaster
        # (attach_yaw_only=True). Obstacles (dobs/wall) are defined in world coordinates, therefore
        # their ray-intersection must be computed using world-frame ray directions.
        rot_wxyz = self.drone_state[..., 3:7]
        if rot_wxyz.ndim == 3 and rot_wxyz.shape[1] == 1:
            rot_wxyz = rot_wxyz.squeeze(1)
        yaw = quaternion_to_euler(rot_wxyz)[..., 2]  # [N]
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        # High-res directions for dobs/wall hit computation.
        ray_dir = self.ray_hits_dir  # [N, rays, 3] in sensor frame
        ray_dir_w = torch.empty_like(ray_dir)
        ray_dir_w[..., 0] = cy[:, None] * ray_dir[..., 0] - sy[:, None] * ray_dir[..., 1]
        ray_dir_w[..., 1] = sy[:, None] * ray_dir[..., 0] + cy[:, None] * ray_dir[..., 1]
        ray_dir_w[..., 2] = ray_dir[..., 2]

        # Low-res directions (used for debug visualization). Keep self.input_dir in sensor frame
        # for other computations that expect sensor-frame rays.
        input_dir = self.input_dir
        input_dir_w = torch.empty_like(input_dir)
        input_dir_w[..., 0] = cy[:, None] * input_dir[..., 0] - sy[:, None] * input_dir[..., 1]
        input_dir_w[..., 1] = sy[:, None] * input_dir[..., 0] + cy[:, None] * input_dir[..., 1]
        input_dir_w[..., 2] = input_dir[..., 2]

        active = int(getattr(self, "_dobs_active_num", getattr(self, "dynamic_obs_num", 0)))
        active = max(0, min(active, int(getattr(self, "dynamic_obs_num", 0))))
        if active > 0 and (self.dobs_states is not None) and (self.dobs_states.shape[0] >= active):
            self.dobs_hits_w = env_utils.dobs_lidar_hits(
                self.lidar_range,
                self.dobs_height,
                self.dobs_states[:active],
                self.lidar.data.pos_w,
                ray_dir_w,
                error_tolerance=0.33
            )
        else:
            pos_w = self.lidar.data.pos_w
            if pos_w.ndim == 3 and pos_w.shape[1] == 1:
                pos_w = pos_w.squeeze(1)
            self.dobs_hits_w = pos_w[:, None, :] + ray_dir_w * (self.lidar_range + 1.0)

        self.wall_hits_w = env_utils.wall_lidar_hits(
            self.lidar_range,
            self.fly_height,
            self.wall_sizes,
            self.lidar.data.pos_w,
            ray_dir_w
        )

        self.merged_hits = env_utils.merge_hit_points(
            self.dobs_hits_w,
            self.wall_hits_w, 
            self.lidar.data.ray_hits_w,
            self.lidar.data.pos_w
        )

        distances = (self.merged_hits - self.lidar.data.pos_w.unsqueeze(1)).norm(dim=-1)
        valid = (distances > 0) & (distances <= self.lidar_range)
        if self.lidar_use_height_filter:
            pos_w_z = self.lidar.data.pos_w[:, 2].unsqueeze(1)
            ray_hits_w_z = self.merged_hits[:, :, 2]
            z_in_range = (ray_hits_w_z >= (pos_w_z - self.bound_h)) & (ray_hits_w_z <= (pos_w_z + 2*self.bound_h))
            valid = valid & z_in_range
        lidar_dis = torch.where(valid, distances, torch.full_like(distances, float("inf")))

        lidar_dis_unfold = lidar_dis.reshape(
            self.num_envs, self.lidar_h_res * self.lidar_h_sample, self.lidar_v_res * self.lidar_v_sample
        ).unfold(1, self.lidar_h_sample, self.lidar_h_sample).unfold(2, self.lidar_v_sample, self.lidar_v_sample) 
        self.lidar_scan_raw = lidar_dis_unfold.reshape(
            self.num_envs, 1, self.lidar_h_res * self.lidar_v_res, self.lidar_h_sample * self.lidar_v_sample
        ).min(dim=-1)[0]

        # --- Sector distance map (meters): [N,1,H,W]
        lidar_scan_dis = self.lidar_scan_raw.reshape(self.num_envs, 1, *self.lidar_resolution)
        lidar_scan_dis = torch.where(
            torch.isfinite(lidar_scan_dis),
            lidar_scan_dis.clamp_max(self.lidar_range),
            torch.full_like(lidar_scan_dis, self.lidar_range),
        )

        # --- Optional "depth missing keep + decay" risk layer (works in distance space)
        # New rule:
        #   Keep+decay ONLY when obstacle can disappear through FOV boundaries (top/bottom elevation bands)
        #   or when previous obstacle is beyond effective range (> lidar_effective_range).
        #   Otherwise (interior + near), no-hit clears immediately.
        if self.lidar_risk_enable and self.risk_depth is None:
            self.risk_depth = torch.full_like(lidar_scan_dis, self.lidar_range + self.lidar_risk_clear_margin)
            self.risk_age = torch.zeros_like(lidar_scan_dis)

        if self.lidar_risk_enable:
            decay_target = self.lidar_range + self.lidar_risk_clear_margin

            # --- knobs
            eff_range = float(self.cfg.task.get("lidar_effective_range", 5.0))  # e.g. 5m
            eff_range = min(eff_range, float(self.lidar_range))

            # vertical edge band margin (deg). If not set, use ~1 vertical bin.
            v_margin = self.cfg.task.get("lidar_v_edge_margin_deg", None)
            if v_margin is None:
                if self.lidar_v_res > 1:
                    v_step = (float(self.lidar_vfov[1]) - float(self.lidar_vfov[0])) / float(self.lidar_v_res - 1)
                    v_margin = float(v_step) * 1.1   # cover roughly 1 edge row
                else:
                    v_margin = 0.0
            v_margin = float(v_margin)

            # --- build vertical edge mask: near top (52deg) or bottom (-7deg)
            v_angles = torch.linspace(
                float(self.lidar_vfov[0]), float(self.lidar_vfov[1]),
                self.lidar_v_res, device=self.device
            )
            edge_v = (v_angles >= (float(self.lidar_vfov[1]) - v_margin)) | (v_angles <= (float(self.lidar_vfov[0]) + v_margin))
            edge_mask = edge_v.view(1, 1, 1, self.lidar_v_res).expand(
                self.num_envs, 1, self.lidar_h_res, self.lidar_v_res
            )

            # --- measurement masks
            hit_mask = lidar_scan_dis < self.lidar_range
            nohit_mask = ~hit_mask

            # --- update memory on hits
            self.risk_depth = torch.where(hit_mask, lidar_scan_dis, self.risk_depth)
            self.risk_age = torch.where(hit_mask, torch.zeros_like(self.risk_age), self.risk_age)

            # --- decide keep vs clear when no-hit
            # risk exists iff stored depth is within lidar_range (cleared state is > lidar_range)
            has_risk = self.risk_depth < self.lidar_range

            keep_due_edge = edge_mask
            keep_far = bool(self.cfg.task.get("lidar_risk_keep_far", False))
            if keep_far:
                keep_due_far = self.risk_depth > eff_range  # last seen obstacle beyond effective range
            else:
                keep_due_far = torch.zeros_like(keep_due_edge, dtype=torch.bool)

            keep_mask = nohit_mask & has_risk & (keep_due_edge | keep_due_far)
            clear_now_mask = nohit_mask & (~keep_mask)

            # --- age update:
            # keep -> age grows; clear_now -> age reset
            self.risk_age = torch.where(keep_mask, self.risk_age + self.dt, self.risk_age)
            self.risk_age = torch.where(clear_now_mask, torch.zeros_like(self.risk_age), self.risk_age)

            # --- decay ONLY where keep_mask is true
            decay_factor = torch.exp(- (self.risk_age / max(self.lidar_risk_decay_tau, 1e-6)) ** 2)
            risk_depth_decay = decay_target - (decay_target - self.risk_depth) * decay_factor

            clear_decay_mask = (self.risk_age >= self.lidar_risk_clear_time) | (risk_depth_decay >= decay_target)

            # apply decay/clear for keep region
            self.risk_depth = torch.where(
                keep_mask,
                torch.where(clear_decay_mask, torch.full_like(self.risk_depth, decay_target), risk_depth_decay),
                self.risk_depth,
            )
            # immediate clear for interior+near no-hit
            self.risk_depth = torch.where(
                clear_now_mask,
                torch.full_like(self.risk_depth, decay_target),
                self.risk_depth,
            )

            # fuse memory back
            lidar_scan_dis = torch.minimum(lidar_scan_dis, self.risk_depth)




        # keep both conventions:
        # - lidar_scan_dis: distance (meters), used for safety/done and radial estimation
        # - lidar_scan: proximity (legacy P2M convention) = lidar_range - distance, used by older reward code / vis
        self.lidar_scan_dis = lidar_scan_dis
        self.lidar_scan = self.lidar_range - lidar_scan_dis

        distance = self.rpos.norm(dim=-1, keepdim=True)
        target_dir = self.rpos / distance.clamp(1e-6)
        vel_fb = (self.drone_state[..., 7:])[..., :3]
        vel_input = vel_fb/self.vel_max
        if self.actions is not None:
            acc_fb = self.actions.unsqueeze(1)
        else:
            acc_fb = torch.zeros_like(vel_fb)
        acc_input = acc_fb/self.acc_max

        if self.depth_image_queue is None:
            window = max(1, self.lidar_radial_window)
            self.depth_image_queue = deque([self.lidar_scan_dis] * window, maxlen=window)
        else:
            self.depth_image_queue.append(self.lidar_scan_dis)
        depth_smoothed = torch.mean(torch.stack(list(self.depth_image_queue)), dim=0)

        radial_channel = torch.full_like(self.lidar_scan_dis, self.lidar_radial_invalid_value)
        if (self.prev_depth_smoothed is not None) and (self.prev_pos is not None) and (self.prev_rot is not None):
            dir_flat = self.input_dir.reshape(self.num_envs, -1, 3)
            prev_depth_flat = self.prev_depth_smoothed.reshape(self.num_envs, -1)
            curr_depth_flat = depth_smoothed.reshape(self.num_envs, -1)

            prev_pos = self.prev_pos
            prev_rot = self.prev_rot
            curr_pos = self.drone_state[..., :3].squeeze(1)
            curr_rot = self.drone_state[..., 3:7].squeeze(1)

            # RayCaster is configured with attach_yaw_only=True, and self.input_dir is generated in the
            # sensor (yaw-only) frame. Use yaw-only quaternions for motion compensation; using full
            # body quaternion here introduces a frame mismatch when roll/pitch is non-zero.
            prev_yaw = quaternion_to_euler(prev_rot)[..., 2]
            curr_yaw = quaternion_to_euler(curr_rot)[..., 2]
            zeros_prev = torch.zeros_like(prev_yaw)
            zeros_curr = torch.zeros_like(curr_yaw)
            prev_rot_yaw = euler_to_quaternion(torch.stack([zeros_prev, zeros_prev, prev_yaw], dim=-1))
            curr_rot_yaw = euler_to_quaternion(torch.stack([zeros_curr, zeros_curr, curr_yaw], dim=-1))

            prev_rot_expand = prev_rot_yaw.unsqueeze(1).expand(-1, dir_flat.shape[1], -1)
            curr_rot_expand = curr_rot_yaw.unsqueeze(1).expand(-1, dir_flat.shape[1], -1)
            prev_pos_expand = prev_pos.unsqueeze(1)
            curr_pos_expand = curr_pos.unsqueeze(1)

            p_prev_body = dir_flat * prev_depth_flat.unsqueeze(-1)
            p_prev_world = quat_rotate(prev_rot_expand, p_prev_body) + prev_pos_expand
            p_prev_curr = quat_rotate_inverse(curr_rot_expand, p_prev_world - curr_pos_expand)

            pred_depth = (p_prev_curr * dir_flat).sum(-1)
            residual = curr_depth_flat - pred_depth
            # 保留靠近为正的语义，突出障碍物接近的危险性
            radial_speed = - residual / max(self.lidar_radial_dt, 1e-6)

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
            radial_channel = radial_flat.reshape(self.num_envs, 1, *self.lidar_resolution)

        self.prev_depth_smoothed = depth_smoothed
        self.prev_pos = self.drone_state[..., :3].squeeze(1)
        self.prev_rot = self.drone_state[..., 3:7].squeeze(1)

        # --- NeuFlow init (ONLY ONCE) ---
        if self.dismap_flow_size is None:
            self.dismap_flow_size = (96, 16)  # (H, W)

        if self.flow_est_model is None:
            self.flow_est_model = init_neuflow(
                self.num_envs,
                self.dismap_flow_size,
                device=self.device,  # 确保和环境 device 一致，比如 cuda:1
            )
            # Print the model device only once when debugging.
            if self.cfg.task.get("debug_flow", False) and not getattr(self, "_dbg_flow_printed", False):
                p = next(self.flow_est_model.parameters())
                print("[DEBUG] flow model param device =", p.device)
                self._dbg_flow_printed = True

        lidar_dis_for_flow = torch.where(
            torch.isfinite(lidar_dis),
            lidar_dis,
            torch.full_like(lidar_dis, self.lidar_range),
        )
        self.scan4flow = (
            self.lidar_range - lidar_dis_for_flow.unsqueeze(1)).reshape(
                self.num_envs, 1,
                self.lidar_h_res * self.lidar_h_sample,
                self.lidar_v_res * self.lidar_v_sample
            ) / self.lidar_range
        scan4flow_scaled = torch.nn.functional.interpolate(self.scan4flow.half() * 255.,
                                                            self.dismap_flow_size,
                                                            mode='bilinear',
                                                            align_corners=False)
        if self.dismap_image_queue is None:
            self.dismap_image_queue = deque([scan4flow_scaled] * int(self.cfg.task.flow_gap + 3),
                                            maxlen=int(self.cfg.task.flow_gap + 3))
        else:
            self.dismap_image_queue.append(scan4flow_scaled)
        dismap_tensor = list(self.dismap_image_queue)
        dismap_image0 = torch.cat(dismap_tensor[:3], dim=1)
        dismap_image1 = torch.cat(dismap_tensor[-3:], dim=1)
        with torch.no_grad():
            self.dismap_flow = self.flow_est_model(dismap_image0, dismap_image1)[-1]
        if self.dismap_flow_queue is None:
            flow_slide_window = int(self.cfg.task.flow_slide_window)
            self.dismap_flow_queue = deque([self.dismap_flow] * flow_slide_window, maxlen=flow_slide_window)
        else:
            self.dismap_flow_queue.append(self.dismap_flow)
        self.dismap_flow_mean = torch.mean(torch.stack(list(self.dismap_flow_queue)), dim=0)
        self.dismap_flow_zoom = torch.nn.functional.interpolate(self.dismap_flow_mean.float(),
                                                                  self.lidar_resolution,
                                                                  mode='bilinear',
                                                                  align_corners=False)
        # ---- depth (sector/scan) channel: [0, 1]
        scan_normalized = self.lidar_scan / max(self.lidar_range, 1e-6)
        scan_normalized = torch.nan_to_num(scan_normalized, nan=1.0, posinf=1.0, neginf=1.0)
        scan_normalized = torch.clamp(scan_normalized, 0.0, 1.0)

        # ---- optical flow channels: clamp to [-1, 1] to stabilize PPO
        flow_zoom = torch.nan_to_num(self.dismap_flow_zoom.float(), nan=0.0, posinf=0.0, neginf=0.0)
        flow_scaled = torch.cat(
            [
                (flow_zoom[:, 0:1, :, :] / 3.6),
                (flow_zoom[:, 1:2, :, :] / 0.6),
            ],
            dim=1,
        )
        flow_scaled = torch.clamp(flow_scaled, -1.0, 1.0)

        # ---- radial residual channel: already normalized, just sanitize
        radial_channel = torch.nan_to_num(radial_channel, nan=self.lidar_radial_invalid_value,
                                        posinf=self.lidar_radial_invalid_value, neginf=self.lidar_radial_invalid_value)
        radial_channel = torch.clamp(radial_channel, -1.0, 1.0)

        # cache radial channel for risk reward
        self.radial_channel = radial_channel

        # ---- 4-channel perception: [depth(1), flow(2), radial(1)]
        self.dismap_stack = torch.cat([scan_normalized, flow_scaled, radial_channel], dim=1)
        # debug shapes (enable only when needed)
        assert scan_normalized.shape[1] == 1
        assert flow_scaled.shape[1] == 2
        assert radial_channel.shape[1] == 1
        assert self.dismap_stack.shape[1] == 4, f"dismap_stack channels={self.dismap_stack.shape}"
        # assert torch.isfinite(self.dismap_stack).all(), "dismap_stack contains NaN/Inf"
        if self.cfg.task.get("debug_checks", False):
            assert torch.isfinite(self.dismap_stack).all()  



        obs = {
            "state": torch.cat([target_dir, vel_input, acc_input], dim=-1).squeeze(1),
            "lidar": self.dismap_stack
        }         

        if (self._should_render(0)) & (self.cfg.task.vis_lidar):
            ray_dis = self.lidar_scan_dis.reshape(self.num_envs, self.lidar_h_res * self.lidar_v_res)
            # self.input_dir is sensor-frame; visualize using the yaw-rotated world-frame directions.
            rayhits = self.lidar.data.pos_w.unsqueeze(1) + ray_dis.unsqueeze(-1) * input_dir_w
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            set_camera_view(
                eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )
            v = (rayhits[0] - x).reshape(*self.lidar_resolution, 3)
            self.debug_draw.vector(x.expand_as(v[:, 1]), v[:, 1])

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs,
                        "intrinsics": self.drone.intrinsics,
                    },
                    [self.num_envs],
                ),
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )


    def _compute_reward_and_done(self):
        def _shape_guard(tag, tensor, expect_last=None, expect_ndim=None):
            n_envs = int(self.num_envs) if not isinstance(self.num_envs, int) else self.num_envs
            if expect_ndim is not None and tensor.ndim != expect_ndim:
                raise RuntimeError(f"[shape_guard:{tag}] bad ndim={tensor.ndim}, shape={tuple(tensor.shape)}, n_envs={n_envs}")
            if tensor.shape[0] != n_envs:
                raise RuntimeError(f"[shape_guard:{tag}] bad dim0={tensor.shape[0]}, shape={tuple(tensor.shape)}, n_envs={n_envs}")
            if expect_last is not None and tensor.shape[-1] != expect_last:
                raise RuntimeError(f"[shape_guard:{tag}] bad last={tensor.shape[-1]}, shape={tuple(tensor.shape)}, n_envs={n_envs}")

        def _to_col(tensor):
            n_envs = int(self.num_envs) if not isinstance(self.num_envs, int) else self.num_envs
            if tensor.ndim == 1:
                return tensor.view(n_envs, 1)
            if tensor.ndim == 2 and tensor.shape[-1] == 1:
                return tensor
            return tensor.view(n_envs, 1)

        # -------------------------
        # task / geometry variables
        # -------------------------
        distance = self.rpos.norm(dim=-1, keepdim=True)
        dis2goal = distance.squeeze(-1)
        height = self.drone_state[..., 2]

        # goal thresholds
        touch_goal_dis = float(self.cfg.task.get("touch_goal_dis", 3.0))
        touch_goal_mask = dis2goal <= touch_goal_dis
        reach_goal_dis = float(self.cfg.task.get("reach_goal_dis", touch_goal_dis))
        success_mask = (dis2goal <= reach_goal_dis)

        # direction to goal (unit)
        vel_direction = self.rpos / distance.clamp_min(1e-6)

        # drone velocities
        vel_w = self.drone.vel_w[..., :3]
        if vel_w.ndim == 3 and vel_w.shape[1] == 1:
            vel_w = vel_w.squeeze(1)
        _shape_guard("vel_w", vel_w, expect_last=3, expect_ndim=2)
        vel_magnitude = vel_w.norm(dim=-1)
        _shape_guard("vel_magnitude", vel_magnitude, expect_ndim=1)

        # action (acceleration command)
        acc = self.actions
        acc_magnitude = acc.norm(dim=-1, keepdim=True)

        # --- jerk baseline handling ---
        # `last_acc` is used to compute a jerk reward/penalty. In vectorized training, only a
        # subset of envs may reset on any given step. Resetting last_acc globally would affect
        # every env and create large training artifacts.
        if self.last_acc is None:
            self.last_acc = acc.clone()
            self.last_acc_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        elif self.last_acc_valid is None:
            self.last_acc_valid = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        if self.last_dis2goal is None:
            self.last_dis2goal = dis2goal

        # allow overriding some constraints from cfg.task (useful for inference)
        virtual_ground = float(self.cfg.task.get("virtual_ground", self.virtual_ground))
        virtual_ceiling = float(self.cfg.task.get("virtual_ceiling", self.virtual_ceiling))
        self.safety_dis = float(self.cfg.task.get("safety_dis", self.safety_dis))

        # -------------------------
        # reward weights
        # -------------------------
        rw = self.cfg.task.get("reward_weights", {})
        k_v = float(rw.get("k_v", 1.2))
        k_a = float(rw.get("k_a", 0.6))
        k_j = float(rw.get("k_j", 0.2))
        k_h = float(rw.get("k_h", 0.3))
        k_g = float(rw.get("k_g", 0.8))
        k_s = float(rw.get("k_s", 1.0))
        k_d = float(rw.get("k_d", 0.6))

        # -------------------------
        # state reward components
        # -------------------------
        beta_vel, beta_acc = 2.0, 5.0
        vel_limit = 1.2 * self.vel_max
        acc_limit = 1.5 * self.acc_max

        beta_hei = 2.0
        hei_set_min = self.fly_height - self.height_bound / 2
        hei_set_max = self.fly_height + self.height_bound / 2

        vel_set_min, vel_set_max = self.vel_min, self.vel_max
        acc_set_min, acc_set_max = self.acc_min, self.acc_max

        reward_vel, reward_acc, reward_jerk, reward_height = self._compute_state_reward(
            beta_vel, vel_set_min, vel_set_max, vel_magnitude,
            beta_acc, acc_set_min, acc_set_max, acc_magnitude,
            beta_hei, hei_set_min, hei_set_max, height,
            acc, self.last_acc, touch_goal_mask
        )
        reward_vel = reward_vel.view(-1, 1)
        reward_acc = reward_acc.view(-1, 1)
        reward_jerk = reward_jerk.view(-1, 1)
        # Avoid applying jerk penalty across episode boundaries.
        # Envs that were reset since the previous step have `last_acc_valid=False`.
        if self.last_acc_valid is not None:
            invalid = (~self.last_acc_valid).view(-1, 1)
            if invalid.any():
                reward_jerk = reward_jerk.clone()
                reward_jerk[invalid] = 1.0
        reward_height = reward_height.view(-1, 1)

        # goal reward (ungated)
        reward_goal = self._compute_goal_reward(
            self.drone.vel_w[..., :3], vel_direction,
            self.last_dis2goal, dis2goal,
            touch_goal_mask
        ).view(-1, 1)

        # legacy safety rewards (optional to keep for ablation)
        reward_safety = self._compute_safety_reward(self.lidar_scan)
        active = int(getattr(self, "_dobs_active_num", getattr(self, "dynamic_obs_num", 0)))
        active = max(0, min(active, int(getattr(self, "dynamic_obs_num", 0))))
        reward_dobs = self._compute_dobs_reward(
            self.dobs_states[:active] if (active > 0 and self.dobs_states is not None) else None,
            self.drone_state[..., :2].squeeze(1),
            self.drone.vel_w[..., :2].squeeze(1),
        )

        # -------------------------
        # risk scalar + gating / relax
        # -------------------------
        risk_gate_goal = bool(self.cfg.task.get("risk_gate_goal", True))
        risk_relax_enable = bool(self.cfg.task.get("risk_relax_enable", True))
        risk_reward_enable = bool(self.cfg.task.get("risk_reward_enable", True))

        if risk_gate_goal or risk_relax_enable or risk_reward_enable:
            radial = getattr(self, "radial_channel", None)
            risk_smax = self._compute_risk_smax(self.lidar_scan_dis, radial)
        else:
            risk_smax = torch.zeros(self.num_envs, device=self.device)
        risk_smax_col = risk_smax.view(-1, 1)

        # goal gate g(risk)
        if risk_gate_goal:
            g_min = float(self.cfg.task.get("risk_gate_g_min", 0.3))
            g_min = float(max(0.0, min(1.0, g_min)))
            goal_gate = g_min + (1.0 - g_min) * (1.0 - risk_smax_col)
            reward_goal = reward_goal * goal_gate
        else:
            goal_gate = torch.ones_like(reward_goal)

        # relax jerk & height constraints under high risk (reduce their weights, do not introduce drift)
        if risk_relax_enable:
            eta_j = float(self.cfg.task.get("risk_relax_eta_j", 0.5))
            eta_h = float(self.cfg.task.get("risk_relax_eta_h", 0.5))
            eta_j = float(max(0.0, min(1.0, eta_j)))
            eta_h = float(max(0.0, min(1.0, eta_h)))
            k_j_eff = (k_j * (1.0 - eta_j * risk_smax_col)).clamp_min(0.0)
            k_h_eff = (k_h * (1.0 - eta_h * risk_smax_col)).clamp_min(0.0)
        else:
            k_j_eff = torch.full_like(reward_jerk, k_j)
            k_h_eff = torch.full_like(reward_height, k_h)

        # -------------------------
        # risk avoidance reward (soft threshold + smooth growth)
        # -------------------------
        reward_risk = torch.zeros_like(reward_goal)
        if risk_reward_enable:
            rcfg = self.cfg.task.get("risk_cfg", {})
            alpha = float(rcfg.get("alpha", 10.0))
            rho0 = float(rcfg.get("rho0", 0.2))
            # base weight
            w_r = float(rcfg.get("w_r", 1.5))

            # optional curriculum: linearly interpolate w_r over global frames
            # risk_cfg: {w_r_start, w_r_end, w_r_steps}
            if ("w_r_start" in rcfg) and ("w_r_end" in rcfg) and ("w_r_steps" in rcfg):
                try:
                    w_r_start = float(rcfg.get("w_r_start"))
                    w_r_end = float(rcfg.get("w_r_end"))
                    w_r_steps = float(rcfg.get("w_r_steps"))
                    w_r_steps = max(w_r_steps, 1.0)
                    frames = float(getattr(self, "_global_frame_count", 0))
                    t = max(0.0, min(1.0, frames / w_r_steps))
                    w_r = w_r_start + (w_r_end - w_r_start) * t
                except Exception:
                    pass


            # baseline subtraction makes reward_risk=0 when risk_smax=0
            baseline = F.softplus(torch.tensor(-alpha * rho0, device=self.device))
            reward_risk = -w_r * (F.softplus(alpha * (risk_smax_col - rho0)) - baseline)

        # -------------------------
        # total reward (before collision penalty)
        # -------------------------
        reward = (
            k_v * reward_vel
            + k_a * reward_acc
            + k_j_eff * reward_jerk
            + k_h_eff * reward_height
            + k_g * reward_goal
            + k_s * reward_safety
            + k_d * reward_dobs
            + reward_risk
        ).view(-1, 1)

        # success bonus (optional)
        success_bonus = float(self.cfg.task.get("success_bonus", 0.0))
        if success_bonus != 0.0:
            reward = reward + success_bonus * success_mask.float().view(-1, 1)

        # time penalty (optional)
        time_penalty = float(self.cfg.task.get("time_penalty", 0.0))
        if time_penalty != 0.0:
            reward = reward - time_penalty

        # -------------------------
        # termination masks
        # -------------------------
        bound_misbehave = env_utils.get_bound_misbehave(
            self.drone_state[..., :2].squeeze(1),
            self.start_pos[..., :2].squeeze(1),
            self.target_pos[..., :2].squeeze(1),
        )
        height_low = _to_col(self.drone.pos[..., 2] < virtual_ground)
        height_high = _to_col(self.drone.pos[..., 2] > virtual_ceiling)
        bound_misbehave = _to_col(bound_misbehave)
        vel_limit_mask = _to_col(vel_magnitude > vel_limit)
        acc_limit_mask = _to_col(acc.abs().amax(dim=-1, keepdim=True) > acc_limit)

        # nearest obstacle distance
        min_depth_map = torch.where(
            self.lidar_scan_dis <= 1e-6,
            torch.full_like(self.lidar_scan_dis, self.lidar_range),
            self.lidar_scan_dis,
        )
        min_depth = einops.reduce(min_depth_map, "n 1 w h -> n 1", "min")
        safety_mask = _to_col(min_depth < self.safety_dis)

        # termination toggles (training can choose to penalize instead of terminate)
        try:
            terminate_on_height_high = bool(self.cfg.task.get("terminate_on_height_high", True))
            terminate_on_vel_limit = bool(self.cfg.task.get("terminate_on_vel_limit", True))
        except Exception:
            terminate_on_height_high = True
            terminate_on_vel_limit = True

        zero = torch.zeros_like(height_low)
        height_high_term = height_high if terminate_on_height_high else zero
        vel_limit_term = vel_limit_mask if terminate_on_vel_limit else zero

        # penalties when not terminating (optional)
        if not terminate_on_height_high:
            pen = float(self.cfg.task.get("height_high_penalty", 0.0))
            if pen != 0.0:
                reward = reward - pen * height_high.float()
        if not terminate_on_vel_limit:
            pen = float(self.cfg.task.get("vel_limit_penalty", 0.0))
            if pen != 0.0:
                reward = reward - pen * vel_limit_mask.float()

        misbehave = (
            height_low
            | height_high_term
            | bound_misbehave
            | vel_limit_term
            | acc_limit_mask
            | safety_mask
        )

        hasnan = _to_col(torch.isnan(self.drone_state).any(-1))
        terminated_base = misbehave | hasnan

        terminate_on_success = bool(self.cfg.task.get("terminate_on_success", False))
        terminated = terminated_base
        if terminate_on_success:
            terminated = terminated_base | success_mask

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # -------------------------
        # termination reason breakdown (step-level; exclusive by priority)
        # -------------------------
        done_any = terminated | truncated

        # success termination only if terminate_on_success is enabled
        done_success = success_mask if terminate_on_success else torch.zeros_like(done_any)
        used = done_success.clone()

        done_safety = done_any & safety_mask & (~used)
        used = used | done_safety

        done_height_low = done_any & height_low & (~used)
        used = used | done_height_low

        done_height_high = done_any & height_high & (~used)
        used = used | done_height_high

        done_bound = done_any & bound_misbehave & (~used)
        used = used | done_bound

        done_vel_limit = done_any & vel_limit_mask & (~used)
        used = used | done_vel_limit

        done_acc_limit = done_any & acc_limit_mask & (~used)
        used = used | done_acc_limit

        done_nan = done_any & hasnan & (~used)
        used = used | done_nan

        done_timeout = done_any & truncated & (~used)
        used = used | done_timeout

        done_other = done_any & (~used)

        # step-level ratios over done events (avoid .item() to prevent CPU sync)
        done_count = done_any.float().sum().clamp_min(1.0)
        done_rate = done_any.float().mean()

        ratio_success = done_success.float().sum() / done_count
        ratio_timeout = done_timeout.float().sum() / done_count
        ratio_safety = done_safety.float().sum() / done_count
        ratio_height_low = done_height_low.float().sum() / done_count
        ratio_height_high = done_height_high.float().sum() / done_count
        ratio_bound = done_bound.float().sum() / done_count
        ratio_vel_limit = done_vel_limit.float().sum() / done_count
        ratio_acc_limit = done_acc_limit.float().sum() / done_count
        ratio_nan = done_nan.float().sum() / done_count
        ratio_other = done_other.float().sum() / done_count

        # write stats (0/1 masks are per-env; ratios are global scalars broadcast to all env slots)
        self.stats["terminated"].copy_(terminated.float())
        self.stats["truncated"].copy_(truncated.float())
        self.stats["done_any"].copy_(done_any.float())
        self.stats["done_success"].copy_(done_success.float())
        self.stats["done_timeout"].copy_(done_timeout.float())
        self.stats["done_safety"].copy_(done_safety.float())
        self.stats["done_height_low"].copy_(done_height_low.float())
        self.stats["done_height_high"].copy_(done_height_high.float())
        self.stats["done_bound"].copy_(done_bound.float())
        self.stats["done_vel_limit"].copy_(done_vel_limit.float())
        self.stats["done_acc_limit"].copy_(done_acc_limit.float())
        self.stats["done_nan"].copy_(done_nan.float())
        self.stats["done_other"].copy_(done_other.float())

        self.stats["done_rate"][:] = done_rate
        self.stats["done_ratio_success"][:] = ratio_success
        self.stats["done_ratio_timeout"][:] = ratio_timeout
        self.stats["done_ratio_safety"][:] = ratio_safety
        self.stats["done_ratio_height_low"][:] = ratio_height_low
        self.stats["done_ratio_height_high"][:] = ratio_height_high
        self.stats["done_ratio_bound"][:] = ratio_bound
        self.stats["done_ratio_vel_limit"][:] = ratio_vel_limit
        self.stats["done_ratio_acc_limit"][:] = ratio_acc_limit
        self.stats["done_ratio_nan"][:] = ratio_nan
        self.stats["done_ratio_other"][:] = ratio_other


        # -------------------------
        # collision penalty (does not replace risk shaping)
        # -------------------------
        reward_collision = torch.zeros_like(reward)
        collision_penalty = float(self.cfg.task.get("collision_penalty", 0.0))
        if collision_penalty != 0.0:
            collision_include_height = bool(self.cfg.task.get("collision_include_height", True))
            collision_mask = safety_mask
            if collision_include_height:
                collision_mask = collision_mask | height_low | height_high
            reward_collision = -collision_penalty * collision_mask.float()
            reward = reward + reward_collision

        # -------------------------
        # optional debug prints (OFF by default; printing is expensive)
        # -------------------------
        if self.cfg.task.get("debug_print_reward", False) and (int(self.progress_buf[0].item()) % 50) == 0:
            print(
                f"\r vel: {(k_v * reward_vel[0]).item():.3f}, "
                f"acc: {(k_a * reward_acc[0]).item():.3f}, "
                f"jerk: {(k_j_eff[0] * reward_jerk[0]).item():.3f}, "
                f"height: {(k_h_eff[0] * reward_height[0]).item():.3f}, "
                f"goal: {(k_g * reward_goal[0]).item():.3f}, "
                f"gate: {goal_gate[0].item():.3f}, "
                f"risk: {reward_risk[0].item():.3f}, "
                f"safety: {(k_s * reward_safety[0]).item():.3f}, "
                f"dobs: {(k_d * reward_dobs[0]).item():.3f}, "
                f"coll: {reward_collision[0].item():.3f}, "
                f"total: {reward[0].item():.3f}\r",
                end="",
                flush=True,
            )

        # -------------------------
        # stats update
        # -------------------------
        self.stats["reward_velocity"].add_(reward_vel)
        self.stats["reward_acceleration"].add_(reward_acc)
        self.stats["reward_jerk"].add_(reward_jerk)
        self.stats["reward_height"].add_(reward_height)
        self.stats["reward_goal"].add_(reward_goal)
        self.stats["reward_safety"].add_(reward_safety)
        self.stats["reward_dobs"].add_(reward_dobs)
        self.stats["reward_risk"].add_(reward_risk)
        self.stats["reward_collision"].add_(reward_collision)

        # scalar diagnostics (store current-step values, not accumulated sums)
        self.stats["risk_smax"].copy_(risk_smax_col)
        self.stats["goal_gate"].copy_(goal_gate)

        vel_magnitude_col = vel_magnitude.view(self.num_envs, 1)
        step_count = self.progress_buf.clamp_min(1.0).unsqueeze(1)
        self.speed_sum.add_(vel_magnitude_col)
        self.acc_sum.add_(acc_magnitude)
        self.stats["avg_speed"] = self.speed_sum / step_count
        self.stats["avg_acc"] = self.acc_sum / step_count
        self.stats["max_speed"] = torch.maximum(self.stats["max_speed"], vel_magnitude_col)
        self.stats["max_acc"] = torch.maximum(self.stats["max_acc"], acc_magnitude)

        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        self.last_acc = acc
        if self.last_acc_valid is not None:
            self.last_acc_valid[:] = True
        self.last_dis2goal = dis2goal

        plan_success = success_mask
        flight_success = success_mask & (~terminated_base) & (~truncated)
        self.stats["plan_success"] = plan_success.float()
        self.stats["flight_success"] = flight_success.float()
        # Curriculum bookkeeping (global)
        lvl = int(getattr(self, "_curriculum_level", 0))
        self.stats["curriculum_level"] = torch.full((self.num_envs, 1), float(lvl), device=self.device)
        self.stats["curriculum_static_obs_num_per_grid"] = torch.full(
            (self.num_envs, 1), float(getattr(self, "static_obs_num_per_grid", 0)), device=self.device
        )
        self.stats["curriculum_dobs_active"] = torch.full(
            (self.num_envs, 1), float(int(getattr(self, "_dobs_active_num", getattr(self, "dynamic_obs_num", 0)))), device=self.device
        )

        # Safe global switch: force all envs to done on the next step; apply the
        # actual terrain/dobs switch in the subsequent global reset.
        if getattr(self, "_force_curriculum_reset", False):
            terminated = torch.ones_like(terminated, dtype=torch.bool)
            truncated = torch.zeros_like(truncated, dtype=torch.bool)
            self.stats["curriculum_reset"] = torch.ones((self.num_envs, 1), device=self.device)
        else:
            self.stats["curriculum_reset"] = torch.zeros((self.num_envs, 1), device=self.device)


        assert terminated.shape == (self.num_envs, 1), terminated.shape
        assert truncated.shape == (self.num_envs, 1), truncated.shape

        if self.cfg.task.get("debug_print_done", False) and int(self.progress_buf[0].item()) % 50 == 0:
            term_rate = float(terminated.float().mean().item())
            trunc_rate = float(truncated.float().mean().item())
            print(f"[DONE] term_rate={term_rate:.3f} trunc_rate={trunc_rate:.3f}")

        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                },
                "stats": self.stats.clone(),
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _compute_state_reward(self, beta_vel, vel_set_min, vel_set_max, vel_magnitude,
                            beta_acc, acc_set_min, acc_set_max, acc_magnitude,
                            beta_hei, hei_set_min, hei_set_max, height, 
                            acc, last_acc, touch_goal_mask):
        if vel_magnitude.ndim == 2 and vel_magnitude.shape[-1] == 1:
            vel_magnitude = vel_magnitude.squeeze(-1)
        if touch_goal_mask.ndim == 2 and touch_goal_mask.shape[-1] == 1:
            touch_goal_mask = touch_goal_mask.squeeze(-1)
        reward_vel = torch.log(torch.exp(- beta_vel * (torch.clamp(vel_set_min - vel_magnitude, min = 0.)
                                        + torch.clamp(vel_magnitude - vel_set_max, min = 0.))) + 1.)
        reward_vel[touch_goal_mask] = torch.log(torch.exp(- beta_vel * torch.clamp(
                                        vel_magnitude[touch_goal_mask] - vel_set_max, min = 0.)) + 1.)
        reward_acc = torch.log(torch.exp(- beta_acc * (torch.clamp(acc_set_min - acc_magnitude, min = 0.)
                                        + torch.clamp(acc_magnitude - acc_set_max, min = 0.))) + 1.)
        reward_jerk = 1. / (1. + torch.norm(acc - last_acc, dim=-1, keepdim=True))
        reward_height = torch.log(torch.exp(- beta_hei * (torch.clamp(hei_set_min - height, min = 0.)
                                        + torch.clamp(height - hei_set_max, min = 0.))) + 1.)

        return reward_vel, reward_acc, reward_jerk, reward_height

    def _compute_goal_reward(self, vel_vector, vel_direction, last_dis2goal, dis2goal, touch_goal_mask):
        """Goal progress reward.

        Controlled by cfg.task.goal_reward_type:
        - 'linear' (recommended): w_d*(d_prev - d_now) + w_theta*cos(theta)
        - 'exp' (legacy): direction + 10*(exp(Δd) - 1), where Δd = d_prev - d_now
        """
        goal_type = str(self.cfg.task.get("goal_reward_type", "exp")).lower()

        # squeeze common shapes
        if vel_vector.ndim == 3 and vel_vector.shape[1] == 1:
            vel_vec = vel_vector.squeeze(1)
        else:
            vel_vec = vel_vector
        if vel_direction.ndim == 3 and vel_direction.shape[1] == 1:
            vel_dir = vel_direction.squeeze(1)
        else:
            vel_dir = vel_direction

        last_d = last_dis2goal.squeeze(-1) if last_dis2goal.ndim == 2 else last_dis2goal
        d_now = dis2goal.squeeze(-1) if dis2goal.ndim == 2 else dis2goal
        touch_mask = touch_goal_mask.squeeze(-1) if (touch_goal_mask.ndim == 2 and touch_goal_mask.shape[-1] == 1) else touch_goal_mask

        if goal_type == "linear":
            w_d = float(self.cfg.task.get("goal_w_d", 10.0))
            w_theta = float(self.cfg.task.get("goal_w_theta", 1.0))

            # progress shaping (delta distance). Support optional scaling / dt normalization.
            progress = (last_d - d_now)
            try:
                progress_scale = float(self.cfg.task.get("goal_progress_scale", 1.0))
            except Exception:
                progress_scale = 1.0
            progress = progress * progress_scale

            # Plan A option: normalize by dt (turn into progress speed)
            try:
                scale_by_dt = bool(self.cfg.task.get("goal_progress_scale_by_dt", False))
            except Exception:
                scale_by_dt = False
            if scale_by_dt:
                dt = float(getattr(self, "dt", 0.0))
                if dt > 1e-6:
                    progress = progress / dt

            # clamp to avoid extreme spikes
            try:
                clamp_val = float(self.cfg.task.get("goal_progress_clamp", 1.0))
            except Exception:
                clamp_val = 1.0
            if clamp_val > 0:
                progress = progress.clamp(min=-clamp_val, max=clamp_val)


            speed = vel_vec.norm(dim=-1).clamp_min(1e-6)
            v_dir = vel_vec / speed.unsqueeze(-1)
            cos_theta = (v_dir * vel_dir).sum(-1).clamp(-1.0, 1.0)

            reward_goal = w_d * progress + w_theta * cos_theta
            reward_goal[touch_mask] = 0.0
            return reward_goal

        # legacy exp shaping (default)
        reward_goal_dir = (vel_vec * vel_dir).sum(-1).clip(max=2.0)
        delta_dis = (last_d - d_now).clamp(min=-3.0, max=3.0)
        reward_goal_dis = (torch.exp(delta_dis) - 1.0) * 10.0
        reward_goal_dis[touch_mask] = 0.0
        reward_goal = reward_goal_dir + reward_goal_dis
        return reward_goal

        

    def _compute_risk_smax(self, lidar_scan_dis: torch.Tensor, radial_channel: torch.Tensor) -> torch.Tensor:
        """Compute differentiable global risk scalar R^{smax} in [0, 1].

        Per-sector risk is defined as:
            TTC^{ij} = r^{ij} / max(v_c^{ij}, v_min)
            H^{ij}   = clip(T0 / TTC^{ij}, 0, 1)

        We aggregate H using a masked softmax to preserve max-like behavior while keeping
        the mapping smooth and differentiable.
        """
        if lidar_scan_dis.ndim == 3:
            lidar_scan_dis = lidar_scan_dis.unsqueeze(1)

        if radial_channel is None:
            radial_channel = torch.zeros_like(lidar_scan_dis)
        elif radial_channel.ndim == 3:
            radial_channel = radial_channel.unsqueeze(1)
        elif radial_channel.ndim == 4 and radial_channel.size(1) != 1:
            radial_channel = radial_channel[:, :1, ...]

        rcfg = self.cfg.task.get("risk_cfg", {})
        T0 = float(rcfg.get("T0", 3.0))
        v_min = float(rcfg.get("v_min", 0.1))
        beta = float(rcfg.get("beta", 10.0))

        max_range = rcfg.get("max_range", None)
        if max_range is None:
            max_range = float(self.cfg.task.get("lidar_effective_range", self.lidar_range))
        max_range = min(float(max_range), float(self.lidar_range))

        # closing speed (m/s)
        #   closing = ego_closing_along_ray + residual_closing_from_dynamic_obstacles
        # residual closing: radial channel is normalized by lidar_radial_max_speed
        radial_speed = radial_channel.float() * float(self.lidar_radial_max_speed)

        # --- ego velocity projection onto rays (covers static obstacles) ---
        ego_closing = torch.zeros_like(lidar_scan_dis)
        ray_dir = getattr(self, "input_dir", None)
        if ray_dir is not None:
            # ray_dir is in yaw-only lidar frame (attach_yaw_only)
            ray_dir_flat = ray_dir.reshape(ray_dir.shape[0], -1, 3).float()

            # drone velocity in yaw-only frame
            vel_w = self.drone.vel_w[..., :3]
            if vel_w.ndim == 3 and vel_w.shape[1] == 1:
                vel_w = vel_w.squeeze(1)

            rot = self.drone_state[..., 3:7]
            if rot.ndim == 3 and rot.shape[1] == 1:
                rot = rot.squeeze(1)
            yaw = quaternion_to_euler(rot)[..., 2]
            zeros = torch.zeros_like(yaw)
            q_yaw = euler_to_quaternion(torch.stack([zeros, zeros, yaw], dim=-1))
            vel_yaw = quat_rotate_inverse(q_yaw, vel_w)

            ego_proj = (vel_yaw.unsqueeze(1) * ray_dir_flat).sum(dim=-1)
            ego_proj = torch.clamp(ego_proj, min=0.0)
            ego_closing = ego_proj.reshape(lidar_scan_dis.shape[0], 1, *self.lidar_resolution)

        closing_raw = ego_closing + radial_speed
        closing_pos = torch.clamp(closing_raw, min=0.0)
        # only apply v_min when approaching; otherwise keep 0 to avoid false risk
        closing = torch.where(
            closing_pos > 0.0,
            torch.maximum(closing_pos, closing_pos.new_tensor(v_min)),
            torch.zeros_like(closing_pos),
        )

        r = lidar_scan_dis.clamp_min(1e-3)
        H = (T0 * closing / r).clamp(0.0, 1.0)

        # mask out very far cells (avoid gradient dilution)
        mask = lidar_scan_dis < max_range
        H = torch.where(mask, H, torch.zeros_like(H))

        H_flat = H.view(H.shape[0], -1)
        mask_flat = mask.view(mask.shape[0], -1)

        logits = beta * H_flat
        logits = logits.masked_fill(~mask_flat, -1e9)

        pi = torch.softmax(logits, dim=-1)
        pi = pi * mask_flat.float()
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        risk_smax = (pi * H_flat).sum(dim=-1)
        return risk_smax

    def _compute_safety_reward(self, lidar_scan: torch.Tensor) -> torch.Tensor:
        """Safety reward (P2M legacy logic).

        Input lidar_scan uses proximity convention: lidar_range - distance (meters),
        typically shaped [N,1,H,W]. This method is intentionally kept identical to
        the oldest P2M implementation to avoid changing reward behavior.
        """
        # Accept [N,H,W] or [N,C,H,W]; use channel0 if multi-channel
        if lidar_scan.ndim == 3:
            lidar_scan = lidar_scan.unsqueeze(1)
        elif lidar_scan.ndim == 4 and lidar_scan.size(1) != 1:
            lidar_scan = lidar_scan[:, :1, ...]

        # proximity -> distance
        lidar_values = self.lidar_range - lidar_scan
        lidar_values_merged = lidar_values.reshape(
            lidar_values.size(0), lidar_values.size(1), -1
        ).squeeze(1)  # [N, H*W]

        lidar_values_clip = torch.clamp(lidar_values_merged - self.safety_dis, min=0.0)

        obs_mask = lidar_values_merged <= (self.lidar_range / 10.0)
        obs_count = obs_mask.sum(dim=1)
        obs_dist_avg = (lidar_values_clip * obs_mask).sum(dim=1) / (obs_count + 1e-9)

        obs_dist_min = lidar_values_clip.min(dim=1)[0]
        obs_dist = torch.where(obs_count != 0, obs_dist_avg, obs_dist_min)

        reward_safety = torch.log(obs_dist).clamp_min(-5.0)
        # Optional: cap positive safety reward (set safety_reward_max=0.0 to make it penalty-only)
        safety_r_max = self.cfg.task.get("safety_reward_max", None)
        if safety_r_max is not None:
            reward_safety = reward_safety.clamp_max(float(safety_r_max))
        return reward_safety.reshape(-1, 1)
    def _compute_dobs_reward(self, obstacle_tensor, drone_pos, drone_vel):
        # NOTE: obstacle_tensor may be None/empty when dynamic obstacles are disabled
        # (e.g., curriculum sets dynamic_obs_active=0). In that case, return 0 reward.
        if obstacle_tensor is None or (hasattr(obstacle_tensor, "numel") and obstacle_tensor.numel() == 0):
            num_env = drone_pos.shape[0]
            return torch.zeros((num_env, 1), device=self.device)

        num_env = drone_pos.shape[0]
        n = obstacle_tensor.shape[0]
        if n == 0:
            # dynamic_obs_num=0 safety: no dynamic obstacles => no dobs reward/penalty
            return torch.zeros((num_env, 1), device=self.device)
        pos = obstacle_tensor[:, 0]
        vel = obstacle_tensor[:, 1]
        rad = obstacle_tensor[:, 2, 0]

        drone_pos_expanded = drone_pos.unsqueeze(1).expand(num_env, n, 2)
        drone_vel_expanded = drone_vel.unsqueeze(1).expand(num_env, n, 2)
        pos_expanded = pos.unsqueeze(0).expand(num_env, n, 2)

        obstacle_vel_drone_frame = vel - drone_vel_expanded
        if self.reward_dobs_max is None:
            self.reward_dobs_max = torch.full((num_env, 1), float("-inf"), device=self.device)

        r = pos - drone_pos_expanded
        dot_product = (r * obstacle_vel_drone_frame).sum(dim=2)
        r_norm = r.norm(dim=2)
        v_norm = obstacle_vel_drone_frame.norm(dim=2)

        cos_theta = dot_product / (r_norm * v_norm)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        coll_mask = theta < (torch.pi / 2)

        vel_magnitude = torch.norm(vel, dim=1)
        dist = torch.norm(pos_expanded - drone_pos_expanded, dim=2) - rad

        unit_velocity = vel / (vel_magnitude.unsqueeze(1) + 1e-6)
        unit_velocity_expanded = unit_velocity.unsqueeze(0).expand(num_env, n, 2)

        v_x = unit_velocity_expanded[..., 0]
        v_y = unit_velocity_expanded[..., 1]
        x = pos_expanded[..., 0]
        y = pos_expanded[..., 1]
        x_d = drone_pos_expanded[..., 0]
        y_d = drone_pos_expanded[..., 1]

        speed_line_distance = torch.abs((x_d - x) * v_y - (y_d - y) * v_x)
        fov_mask = dist <= (self.lidar_range * 0.75)
        obs_count = fov_mask.sum(dim=1).clamp(min=1)

        k_v = torch.norm(obstacle_vel_drone_frame, dim=2)
        k_theta = 1.0 - (2 * theta / torch.pi)
        k_d = torch.exp(1.0 / (1.0 + speed_line_distance))
        k_total = torch.where(
            coll_mask != 0,
            1 + k_v * k_theta * k_d,
            torch.ones_like(theta),
        )

        r_d_zoom = (dist - self.safety_dis).clamp_min(0.0) / (k_total + 1e-6)
        r_d = torch.log(r_d_zoom).clamp_min(-5.0)

        reward_dobs = ((r_d * fov_mask).sum(dim=1) / obs_count).reshape(num_env, 1)
        self.reward_dobs_max = torch.max(self.reward_dobs_max, reward_dobs)
        reward_dobs = torch.where(reward_dobs == 0, self.reward_dobs_max, reward_dobs)

        return reward_dobs
