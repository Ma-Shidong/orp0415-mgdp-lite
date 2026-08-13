# MGDP-lite v2 修改说明

本文记录本次新增的 `task.input_mode=mgdp_lite_v2`。本次只改输入前端，不改 PPO 网络主体，不改 P2M 奖励函数。

## 修改目标

保留原有三种使用方式：

```bash
task.input_mode=p2m
task.input_mode=mgdp_lite
task.input_mode=mgdp_lite_v2
```

其中 `p2m` 和旧 `mgdp_lite` 不覆盖、不删除。新的 `mgdp_lite_v2` 仍然输出 `4 x 36 x 6` 的 LiDAR 输入，所以可以继续使用当前项目里的 CNN + state 9 维 + PPO actor/critic。

## v2 四个输入通道

`mgdp_lite_v2` 的四个 LiDAR 通道为：

```text
ch0 = denoised proximity
ch1 = signed radial velocity
ch2 = 3D flight corridor risk
ch3 = radial TTC risk
```

含义：

- `ch0`：三帧 median 去噪后的接近度，表示障碍物离无人机多近，范围 `[0, 1]`。
- `ch1`：有符号径向速度，靠近为正、远离为负，按 `mgdp_v2_radial_speed_scale` 归一化到 `[-1, 1]`。
- `ch2`：三维飞行走廊风险，判断障碍是否挡在当前朝向目标的飞行走廊附近。
- `ch3`：径向 TTC 风险，只在障碍物正在靠近且 TTC 小于设定 horizon 时变大。

## 涉及文件

- `resources/envs/single/env.py`：训练环境新增 `mgdp_lite_v2` 输入构造和观测统计。
- `scripts/infer.py`：RViz/ROS 测试使用的推理端同步 `mgdp_lite_v2` 输入。
- `scripts/infer_ros.py`：ROS 推理端同步 `mgdp_lite_v2` 输入。
- `scripts/eval_p2m_ros_batch.py`：新增 `--input-mode` 参数，批量评估时可选择 `p2m`、`mgdp_lite` 或 `mgdp_lite_v2`。
- `cfg/task/train_env_goal.yaml`、`cfg/task/train_env.yaml`、`cfg/task/infer_ros_env.yaml`：新增 v2 相关默认参数。

## 关键参数

```yaml
mgdp_v2_radial_speed_scale: 6.0
mgdp_v2_use_effective_dt: true
mgdp_v2_corridor_sigma: 1.0
mgdp_v2_corridor_speed_adaptive: false
mgdp_v2_corridor_speed_gain: 0.12
mgdp_v2_corridor_forward_only: true
mgdp_v2_ttc_min_closing_speed: 0.15
mgdp_v2_ttc_horizon: 4.0
mgdp_v2_ttc_tau: 1.5
```

说明：

- `mgdp_v2_radial_speed_scale`：ch1 的速度归一化尺度，越大则 ch1 数值越保守。
- `mgdp_v2_use_effective_dt`：训练端按 `dt * lidar_update_period` 计算径向速度；推理端按 `sim_dt`。
- `mgdp_v2_corridor_sigma`：飞行走廊半径尺度，越大则 ch2 覆盖更宽。
- `mgdp_v2_corridor_forward_only`：只把目标方向前方的点计入走廊风险。
- `mgdp_v2_ttc_horizon`：超过该 TTC 的点不认为紧迫。
- `mgdp_v2_ttc_tau`：TTC 风险指数衰减时间常数。

## 训练命令

建议从头训练，不加载旧 checkpoint：

```bash
cd /home/csj/msd/orp0415/orp/scripts
source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

RUN_DIR=/home/csj/msd/orp0415/orp/runs/mgdp_lite_v2_train_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

SIM_DEVICE=cuda:0 \
WANDB_DIR="$RUN_DIR" \
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH \
python train.py \
  headless=true \
  wandb.mode=offline \
  task.input_mode=mgdp_lite_v2 \
  task.reward_mode=p2m \
  resume_checkpoint=null \
  task.env.num_envs=1024 \
  task.env.max_episode_length=1200 \
  algo.train_every=64 \
  total_frames=131072000 \
  max_iters=2000 \
  eval_interval=100 \
  save_interval=100 \
  record_video=false \
  task.success_curriculum.enable=true \
  task.flow_update_period=2 \
  2>&1 | tee "$RUN_DIR/train.log"
```

## 测试/评估

单次 RViz/ROS 推理时要显式指定：

```bash
task.input_mode=mgdp_lite_v2
```

批量评估可以使用：

```bash
python /home/csj/msd/orp0415/orp/scripts/eval_p2m_ros_batch.py \
  --input-mode mgdp_lite_v2 \
  --device cuda:0
```

## 2026-08-13 针对性修正

本轮修正了 v2 的三个观测一致性问题：

- `ch3` 的 TTC 风险不再只使用 ego-motion compensation 后的障碍物径向速度，而是使用 `无人机自身沿 ray 的 closing 分量 + 障碍物自身径向速度`。这样静态墙场景中，`ch1` 可以接近 0，但无人机主动飞向墙时 `ch3` 仍会升高。
- `infer.py` 和训练端一样使用 yaw-only 姿态做径向 motion compensation，避免训练/推理的 LiDAR frame 语义不一致。
- `infer_ros.py` 优先使用 PointCloud2 的 `msg.header.stamp` 计算真实 LiDAR 帧间隔；stamp 为空时退回 wall time，第一帧或时间戳异常时，障碍物自身 radial motion 按无效处理，避免除零和假大速度。

## 看哪些日志

训练时除了原来的 `train/stats.flight_success`、`train/stats.done_success`、`train/stats.reward_collision`、`train/stats.done_safety`，v2 还会记录这些观测检查量：

```text
train/stats.obs_ch0_mean
train/stats.obs_ch0_max
train/stats.obs_ch1_abs_mean
train/stats.obs_ch1_pos_ratio
train/stats.obs_ch1_neg_ratio
train/stats.obs_ch2_mean
train/stats.obs_ch2_max
train/stats.obs_ch3_mean
train/stats.obs_ch3_max
```

如果 `obs_ch1_abs_mean` 长期接近 0，说明径向速度通道几乎没有信息；如果 `obs_ch2_mean` 和 `obs_ch3_mean` 长期为 0，说明走廊风险或 TTC 风险可能过于保守，需要再调对应参数。
