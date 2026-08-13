# P2M 输入/奖励 + 当前网络训练命令

目标固定为：

```text
输入：P2M
奖励函数：P2M
网络：当前项目里的 PPOPolicy
仿真：Isaac Sim
项目路径：/home/csj/msd/orp0415/orp
```

所有命令都显式使用当前项目路径，避免误用旧路径 `/home/csj/orp`。

## 1. 先打开 GUI 看场景

这个命令只用来确认 Isaac GUI、无人机、障碍物、环境生成是否正常。  
看起来没问题后，关掉这个进程，再进行正式训练。

```bash
cd /home/csj/msd/orp0415/orp/scripts

source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

RUN_DIR=/home/csj/msd/orp0415/orp/runs/p2m_gui_probe_$(date +%Y%m%d_%H%M%S)
mkdir -p $RUN_DIR

SIM_DEVICE=cuda:0 \
WANDB_DIR=$RUN_DIR \
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH \
python train.py \
  headless=false \
  wandb.mode=offline \
  task.input_mode=p2m \
  task.reward_mode=p2m \
  task.sim.use_flatcache=false \
  task.env.num_envs=8 \
  task.env.max_episode_length=256 \
  algo.train_every=4 \
  total_frames=4096 \
  max_iters=16 \
  eval_interval=-1 \
  save_interval=8 \
  record_video=false \
  task.success_curriculum.enable=false \
  task.dynamic_obs_num=2 \
  task.static_obs_num_total=4 \
  task.static_obs_max_total=8 \
  task.flow_update_period=1
```

关键参数：

```text
headless=false                 打开 Isaac GUI
task.input_mode=p2m            使用 P2M 输入
task.reward_mode=p2m           使用 P2M 奖励函数
task.sim.use_flatcache=false   GUI 模式下更稳定
task.env.num_envs=8            GUI 检查时环境数不要太大
```

## 2. 中型训练

GUI 确认没问题后，先跑这个。它是正式训练，但规模不会太夸张，适合先看趋势。

```bash
cd /home/csj/msd/orp0415/orp/scripts

source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

RUN_DIR=/home/csj/msd/orp0415/orp/runs/p2m_train_mid_$(date +%Y%m%d_%H%M%S)
mkdir -p $RUN_DIR

SIM_DEVICE=cuda:0 \
WANDB_DIR=$RUN_DIR \
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH \
python train.py \
  headless=true \
  wandb.mode=offline \
  task.input_mode=p2m \
  task.reward_mode=p2m \
  resume_checkpoint=null \
  task.env.num_envs=256 \
  task.env.max_episode_length=768 \
  algo.train_every=32 \
  total_frames=2457600 \
  max_iters=300 \
  eval_interval=50 \
  save_interval=50 \
  record_video=false \
  task.success_curriculum.enable=false \
  task.dynamic_obs_num=4 \
  task.static_obs_num_total=12 \
  task.static_obs_max_total=20 \
  task.flow_update_period=1 \
  2>&1 | tee $RUN_DIR/train.log
```

规模：

```text
每轮采样量 = task.env.num_envs * algo.train_every
          = 256 * 32
          = 8192 frames

总采样量约为：
8192 * 300 = 2457600 frames
```

实时看日志：

```bash
tail -f /home/csj/msd/orp0415/orp/runs/p2m_train_mid_*/train.log
```

## 3. 大型训练

中型训练确认没有明显问题后，再跑这个。这个更接近正式训练，时间会长很多。

```bash
cd /home/csj/msd/orp0415/orp/scripts

source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

RUN_DIR=/home/csj/msd/orp0415/orp/runs/p2m_train_large_$(date +%Y%m%d_%H%M%S)
mkdir -p $RUN_DIR

SIM_DEVICE=cuda:0 \
WANDB_DIR=$RUN_DIR \
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH \
python train.py \
  headless=true \
  wandb.mode=offline \
  task.input_mode=p2m \
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
  2>&1 | tee $RUN_DIR/train.log
```

规模：

```text
每轮采样量 = 1024 * 64 = 65536 frames
总采样量约为：65536 * 2000 = 131072000 frames
```

实时看日志：

```bash
tail -f /home/csj/msd/orp0415/orp/runs/p2m_train_large_*/train.log
```

## 4. 从中型训练继续到大型训练

如果中型训练已经保存了 checkpoint，例如：

```text
/home/csj/msd/orp0415/orp/runs/p2m_train_mid_<时间戳>/wandb/<run-id>/files/checkpoint_final.pt
```

大型训练时把：

```bash
resume_checkpoint=null
```

改成：

```bash
resume_checkpoint=/home/csj/msd/orp0415/orp/runs/p2m_train_mid_<时间戳>/wandb/<run-id>/files/checkpoint_final.pt
```

## 5. 命令逐行解释

下面用中型训练命令解释。大型训练只是把训练规模调大，含义一样。

```bash
cd /home/csj/msd/orp0415/orp/scripts
```

进入训练脚本目录。`train.py` 在这里，所以后面运行 `python train.py`。

```bash
source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp
```

激活 `orp` conda 环境。训练用的 Python、PyTorch、Isaac 相关依赖都来自这个环境。

```bash
RUN_DIR=/home/csj/msd/orp0415/orp/runs/p2m_train_mid_$(date +%Y%m%d_%H%M%S)
mkdir -p $RUN_DIR
```

创建本次训练的保存目录。`$(date +%Y%m%d_%H%M%S)` 会自动加入时间戳，所以每次训练都会生成新目录，不会覆盖上一次。

```bash
SIM_DEVICE=cuda:0
```

指定 Isaac 仿真使用第 0 张 GPU。通常单卡机器就用 `cuda:0`。

```bash
WANDB_DIR=$RUN_DIR
```

指定日志和 checkpoint 保存到本次 `RUN_DIR` 下面。`.pt` 文件一般会在：

```text
$RUN_DIR/wandb/offline-run-xxxx/files/
```

```bash
PYTHONPATH=/home/csj/msd/orp0415/orp:$PYTHONPATH
```

让 Python 优先使用当前项目代码，避免误用旧路径 `/home/csj/orp`。

```bash
python train.py
```

启动训练脚本。后面的 `headless=true`、`task.input_mode=p2m` 等都是传给 `train.py` 的配置。

```bash
headless=true
```

不打开 Isaac GUI，在后台训练。正式训练建议用这个。  
如果只是想看场景，把它改成 `headless=false`，但不要用 GUI 跑大规模训练。

```bash
wandb.mode=offline
```

wandb 离线模式，只在本地记录日志，不上传云端。

```bash
task.input_mode=p2m
task.reward_mode=p2m
```

这两行最关键：

```text
输入 = P2M
奖励函数 = P2M
网络 = 当前项目里的 PPOPolicy
```

也就是你现在要做的组合。

```bash
resume_checkpoint=null
```

不加载旧模型，从头开始训练。  
如果要接着某个模型训练，就把 `null` 换成真实 `.pt` 路径。

```bash
task.env.num_envs=256
```

并行仿真环境数量。数值越大，采样越快，但显存和仿真压力越大。

```bash
task.env.max_episode_length=768
```

每次飞行最多持续 768 个仿真 step。成功、碰撞、飞太低、飞太高时会提前结束。

```bash
algo.train_every=32
```

每个环境采样 32 步后做一次 PPO 更新。  
所以中型训练每轮采样量是：

```text
256 * 32 = 8192 frames
```

```bash
total_frames=2457600
max_iters=300
```

`max_iters=300` 表示最多训练 300 轮。  
中型训练的总采样量约为：

```text
256 * 32 * 300 = 2457600 frames
```

```bash
eval_interval=50
```

每 50 个 iteration 做一次评估，用来看当前模型效果。

```bash
save_interval=50
```

每 50 个 iteration 保存一次 checkpoint。训练正常结束后还会保存 `checkpoint_final.pt`。

```bash
record_video=false
```

训练时不录视频，避免拖慢速度和占空间。

```bash
task.success_curriculum.enable=false
```

中型训练先关闭课程学习，让环境更固定，方便观察基础趋势。大型训练里可以打开。

```bash
task.dynamic_obs_num=4
task.static_obs_num_total=12
task.static_obs_max_total=20
```

障碍物数量设置：动态障碍物 4 个，静态障碍物 12 个，静态障碍物最大容量 20。

```bash
task.flow_update_period=1
```

环境中动态信息的更新周期。`1` 表示每一步都更新，计算更细。

```bash
2>&1 | tee $RUN_DIR/train.log
```

把终端输出和报错都保存到：

```text
$RUN_DIR/train.log
```

同时屏幕上也能实时看到输出。

最需要记住的是：

```text
训练规模主要由 num_envs * train_every * max_iters 决定
checkpoint 保存频率由 save_interval 决定
日志和 .pt 保存位置由 RUN_DIR / WANDB_DIR 决定
```

## 6. 看训练有没有变好

重点看这些指标：

```text
done_success              成功率，越高越好
return                    总回报，整体越高越好
reward_goal               接近目标的奖励，越高越好
reward_collision          碰撞惩罚，负值越少越好
done_safety               安全终止比例，越低越好
done_height_low/high      高度异常终止，越低越好
episode_len               飞行步数，不要只看越长越好，要结合成功率
```

PPO 训练会波动，不要因为几轮变差就马上停。更应该看几十轮到几百轮的趋势。

## 7. checkpoint 保存规则

当前代码会保存：

```text
每 save_interval 个 iteration 保存一次 checkpoint_帧数.pt
训练正常结束后保存 checkpoint_final.pt
```

中途强行停止时，`checkpoint_final.pt` 不一定会保存。  
所以中型训练建议 `save_interval=50`，大型训练建议 `save_interval=100`。

当前代码还没有自动保存 `checkpoint_best.pt`。如果后面你想自动保留最优模型，需要再加一段“按成功率保存最佳 checkpoint”的逻辑。

## 8. 推荐流程

```text
1. 先运行 GUI 检查命令
2. 看到 Isaac 里场景正常后，关掉 GUI
3. 运行中型训练
4. 看日志里的 success、return、collision 等趋势
5. 趋势正常后，运行大型训练
6. 用保存的 checkpoint 做可视化测试
```
