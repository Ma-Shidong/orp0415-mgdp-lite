# ORP/P2M 复现笔记

这份笔记面向刚接触 Isaac Sim 和这个项目的人。目标是先跑通，再看懂，再改进。

## 1. 先把项目想成四块

- `scripts/train.py`
  - 训练入口。
  - 创建 Isaac 环境，创建 PPO 策略网络，采样 rollout，计算 advantage，然后更新 actor/critic。

- `resources/envs/single/env.py`
  - 任务环境核心。
  - 负责无人机初始点/目标点、静态障碍、动态障碍、LiDAR/flow 输入、reward 和 done 条件。

- `cfg/task/train_env_goal.yaml`
  - 任务配置。
  - 这里控制环境数量、障碍物数量、LiDAR 分辨率、`input_mode: p2m`、`reward_mode: p2m`、奖励权重等。

- `resources/learning/ppo/ppo.py` 和 `scripts/train.py` 里的 `PPOPolicy`
  - 强化学习算法和网络结构。
  - 当前策略输入是 `state: [N, 9]` 加 `lidar: [N, 4, 36, 6]`。

## 2. 当前 P2M 版本输入和奖励

当前代码已经切到 P2M 风格：

- `input_mode: p2m`
  - 使用 P2M 的 proximity/depth 风格 LiDAR 输入。
  - 官方 P2M 是 3 通道：proximity + 2 个 flow 通道。
  - 当前项目网络保留 4 通道，所以第 4 个 radial 通道在 P2M 模式下置零，避免改网络结构影响训练管线。

- `reward_mode: p2m`
  - 包含 velocity、acceleration、jerk、height、goal、safety、dynamic obstacle 等项。
  - 关键统计项包括 `reward_safety` 和 `reward_dobs`。

## 3. 路径隔离

这台机器上还有 `/home/csj/orp`，为了不导入旧项目，运行时固定加：

```bash
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH
```

建议所有输出也写到当前项目下：

```bash
WANDB_DIR=/home/csj/msd/orp0415/orp/runs/...
```

## 4. 最小 smoke test

用途：确认代码能跑、P2M observation/reward 正常、checkpoint 能保存。

```bash
cd /home/csj/msd/orp0415/orp/scripts
source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

WANDB_DIR=/home/csj/msd/orp0415/orp/runs/wandb_smoke \
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH \
python train.py \
  headless=true \
  wandb.mode=offline \
  task.env.num_envs=8 \
  task.env.max_episode_length=16 \
  algo.train_every=2 \
  total_frames=16 \
  max_iters=1 \
  eval_interval=-1 \
  save_interval=-1 \
  record_video=false \
  task.success_curriculum.enable=false \
  task.dynamic_obs_num=2 \
  task.static_obs_num_total=0 \
  task.static_obs_max_total=2 \
  task.flow_update_period=1 \
  +task.debug_checks=true
```

期望看到：

- `lidar: torch.Size([8, 4, 36, 6])`
- `state: torch.Size([8, 9])`
- stats 里有 `reward_safety` 和 `reward_dobs`
- 结尾保存 `checkpoint_final.pt`

## 5. 稍复杂的 headless 测试

用途：确认障碍物更多时训练循环仍然稳定。

```bash
cd /home/csj/msd/orp0415/orp/scripts
source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

WANDB_DIR=/home/csj/msd/orp0415/orp/runs/wandb_p2m_complex \
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH \
python train.py \
  headless=true \
  wandb.mode=offline \
  task.env.num_envs=64 \
  task.env.max_episode_length=64 \
  algo.train_every=2 \
  total_frames=640 \
  max_iters=5 \
  eval_interval=-1 \
  save_interval=-1 \
  record_video=false \
  task.success_curriculum.enable=false \
  task.dynamic_obs_num=6 \
  task.static_obs_num_total=12 \
  task.static_obs_max_total=16 \
  task.flow_update_period=1 \
  +task.debug_checks=true
```

## 6. Isaac GUI 可视化测试

GUI 模式建议先用小规模环境，并关闭 flatcache。今天验证过这个短命令能完整跑完：

```bash
cd /home/csj/msd/orp0415/orp/scripts
source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

WANDB_DIR=/home/csj/msd/orp0415/orp/runs/isaac_gui_probe \
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH \
HYDRA_FULL_ERROR=1 \
python train.py \
  headless=false \
  wandb.mode=offline \
  task.env.num_envs=8 \
  task.env.max_episode_length=128 \
  algo.train_every=4 \
  total_frames=96 \
  max_iters=3 \
  eval_interval=-1 \
  save_interval=-1 \
  record_video=false \
  task.success_curriculum.enable=false \
  task.dynamic_obs_num=2 \
  task.static_obs_num_total=4 \
  task.static_obs_max_total=8 \
  task.flow_update_period=1 \
  task.sim.use_flatcache=false
```

如果你想打开窗口看更久，可以把 `total_frames` 和 `max_iters` 调大，但建议不要连续频繁启动/关闭 Isaac GUI，否则 X/GLX 可能报资源不足。

## 7. 今天已经踩到的坑

- GUI + `task.sim.use_flatcache=true` 时，曾出现：
  - `Failed to get root link transforms from backend`
  - 后续 Isaac segfault
  - 规避方式：GUI 可视化时加 `task.sim.use_flatcache=false`

- `models/p2m_default.pt` 不能直接加载到当前 4 通道网络：
  - checkpoint 第一层是 `[4, 3, 5, 3]`
  - 当前网络第一层是 `[4, 4, 5, 3]`
  - 后面如果要复现官方 P2M 权重，需要做 3 通道网络版本，或者做 3->4 通道权重迁移。

## 8. 下一步改进路线

建议按这个顺序学和改：

1. 先能稳定运行 smoke test 和 GUI probe。
2. 看懂 `env.py` 里的 observation 和 reward。
3. 跑一段正式训练，观察 `done_success`、`done_collision`、`reward_goal`、`reward_safety`。
4. 再改奖励权重或输入通道。
5. 最后再动网络结构，比如 GRU、teacher-student、3 通道 P2M 网络等。
