# P2M 论文方法复现实验与对比说明

这份文档记录如何按 P2M 论文/官方代码的测试流程评估当前 ORP/P2M checkpoint，并把结果和论文中的 P2M、FAPP、NavRL、Obsnet 做对比。

## 1. 论文测试方法

论文信息：

```text
Flow-Aided Flight Through Dynamic Clutters From Point to Motion
Xu et al., IEEE RA-L, 2025
官方代码：https://github.com/arclab-hku/P2M
```

官方 ROS 测试流程是三部分：

```bash
# 1. 地图/障碍物仿真
source devel/setup.bash
roslaunch map_generator sim_test.launch

# 2. LiDAR raycast
source devel/setup.bash
roslaunch lidar scanner.launch

# 3. 策略推理
conda activate p2m
cd scripts
python infer.py
```

障碍物数量由官方 launch 文件里的两个参数控制：

```text
src/uav_simulator/map_generator/launch/sim_test.launch
```

```xml
<param name="map/obs_num" value=""/>
<param name="map/moving_obs_num" value=""/>
```

本文档里的批量脚本不直接修改 launch 文件，而是启动 `dynamic_env` 时通过 ROS 私有参数传入障碍物数量和随机种子。这样不会污染旧项目，也不会反复改源码。

## 2. 论文评价指标

P2M 论文 Table I 使用这些指标：

| 指标 | 含义 | 这里的统计方式 |
|---|---|---|
| `eta (%)` | 成功率 | `到达目标 && 未触发碰撞阈值` 的比例 |
| `va` | 平均飞行速度 | 成功样本的轨迹长度 / 耗时 |
| `tp` | 规划延迟 | 论文为感知到动作的延迟；当前批量脚本暂不插桩 `infer.py`，所以不做公平对比 |
| `Rl` | 路径效率 | 实际轨迹长度 / 显式起点到终点直线距离 |
| `ds` | 安全距离 | 从 `/ray2array_hits` 估计的最小障碍物距离 |

当前脚本采用严格安全判定：

```text
reach_goal_dis = 1.0 m
collision_dis = 0.3 m
连续 5 帧 LiDAR 最小距离低于 collision_dis 才算碰撞
小于 0.05 m 的极小 raycast 值只用于 raw 记录，不用于 filtered safety distance
正式批量实验中，触发碰撞后该 trial 立即结束并记为失败
```

## 3. 论文 Table I 基线

| 场景 | 方法 | 成功率 eta (%) | 速度 va (m/s) | 延迟 tp (ms) | 路径效率 Rl | 安全距离 ds (m) |
|---|---:|---:|---:|---:|---:|---:|
| 7 dynamic + 7 static | P2M | 95 | 3.14 | 9.58 | 1.13 | 2.80 |
| 7 dynamic + 7 static | FAPP | 90 | 2.20 | 9.31 | 1.08 | 2.43 |
| 7 dynamic + 7 static | NavRL | 45 | 0.88 | 24.30 | 1.04 | 1.71 |
| 7 dynamic + 7 static | Obsnet | 20 | 5.73 | 4.05 | 1.05 | 0.97 |
| 13 dynamic + 13 static | P2M | 65 | 2.55 | 10.38 | 1.43 | 2.05 |
| 13 dynamic + 13 static | FAPP | 30 | 1.88 | 12.75 | 1.22 | 1.93 |
| 13 dynamic + 13 static | NavRL | 30 | 0.89 | 26.19 | 1.06 | 1.62 |
| 13 dynamic + 13 static | Obsnet | 30 | 5.61 | 5.06 | 1.15 | 0.90 |
| 25 dynamic + 19 static | P2M | 40 | 2.29 | 12.29 | 1.20 | 1.83 |
| 25 dynamic + 19 static | FAPP | 25 | 2.09 | 21.36 | 1.21 | 1.74 |
| 25 dynamic + 19 static | NavRL | 20 | 0.90 | 33.85 | 1.20 | 1.39 |
| 25 dynamic + 19 static | Obsnet | 20 | 5.35 | 6.92 | 1.08 | 0.93 |
| 25 dynamic only | P2M | 50 | 2.92 | 10.26 | 1.27 | 2.30 |
| 25 dynamic only | FAPP | 20 | 1.94 | 17.01 | 1.31 | 2.05 |
| 25 dynamic only | NavRL | 15 | 0.87 | 28.28 | 1.25 | 1.48 |
| 25 dynamic only | Obsnet | 25 | 5.33 | 4.32 | 1.07 | 0.96 |
| 44 static only | P2M | 60 | 2.15 | 8.88 | 1.24 | 1.52 |
| 44 static only | FAPP | 75 | 1.84 | 22.89 | 1.07 | 1.36 |
| 44 static only | NavRL | 80 | 0.92 | 27.09 | 1.13 | 1.32 |
| 44 static only | Obsnet | / | / | / | / | / |

## 4. 当前要比较的两个 checkpoint

来自 8000 轮训练：

```text
# 均衡/安全候选
/media/share/csj/msd/orp_runs/p2m_train_8000_from_52494336_20260715_111956/wandb/offline-run-20260715_112007-c3400gih/files/checkpoint_367067136.pt

# 训练成功率更高候选
/media/share/csj/msd/orp_runs/p2m_train_8000_from_52494336_20260715_111956/wandb/offline-run-20260715_112007-c3400gih/files/checkpoint_170459136.pt
```

训练日志附近指标：

| Checkpoint | Curriculum level | 训练 success | 训练 done_safety | collision reward | return |
|---|---:|---:|---:|---:|---:|
| `checkpoint_367067136.pt` | 4 | 0.673 | 0.146 | -60.2 | 2933 |
| `checkpoint_170459136.pt` | 4 | 0.686 | 0.196 | -80.5 | 2784 |

## 5. 场景映射

批量脚本使用与论文 Table I 对齐的 5 个场景：

| 脚本场景名 | 论文场景 | `map/obs_num` 静态障碍 | `map/moving_obs_num` 动态障碍 |
|---|---|---:|---:|
| `dyn7_static7` | 7 dynamic + 7 static | 7 | 7 |
| `dyn13_static13` | 13 dynamic + 13 static | 13 | 13 |
| `dyn25_static19` | 25 dynamic + 19 static | 19 | 25 |
| `dyn25_static0` | 25 dynamic only | 0 | 25 |
| `dyn0_static44` | 44 static only | 44 | 0 |

统一测试空间：

```text
map/x_size = 10
map/y_size = 25
map/z_size = 5
start = [0, -15, 2]
goal = [0, 15, 2]
timeout = 60 s
LiDAR v sample = 2
```

## 6. 先前单 seed 冒烟结果

这只是验证测试管线能跑通，不是论文级 benchmark。

```text
场景：13 dynamic + 13 static
seed：24
timeout：60 s
reach_goal_dis：1.0 m
collision_dis：0.3 m
```

| Checkpoint | 到达目标 | 安全成功 | 平均速度 (m/s) | 路径效率 | filtered 最小距离 (m) | raw 最小距离 (m) | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `checkpoint_367067136.pt` | yes | no | 2.01 | 1.46 | 0.050 | 0.00055 | 能到终点，但按 0.3m 安全阈值不算成功 |
| `checkpoint_170459136.pt` | no | no | 1.79 | 2.27 | 0.050 | 0.00020 | 接近终点但高度偏低，未进入 1m 目标半径 |

单 seed 下，`checkpoint_367067136.pt` 更值得继续测；但两个 checkpoint 都还不能说超过论文 P2M。

## 7. 严谨批量实验命令

新增脚本：

```text
/home/csj/msd/orp0415/orp/scripts/eval_p2m_ros_batch.py
```

推荐先做 20 seeds 的正式小表：

```bash
cd /home/csj/msd/orp0415/orp

source /opt/ros/noetic/setup.bash
source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

export TMPDIR=/media/share/csj/msd/orp_tmp
mkdir -p "$TMPDIR" /media/share/csj/msd/orp_eval

python scripts/eval_p2m_ros_batch.py \
  --run-name p2m_paper_eval_20seeds_$(date +%Y%m%d_%H%M%S) \
  --out-root /media/share/csj/msd/orp_eval \
  --checkpoints balanced_367067136,high_success_170459136 \
  --scenarios dyn7_static7,dyn13_static13,dyn25_static19,dyn25_static0,dyn0_static44 \
  --seeds 0:19 \
  --timeout 60 \
  --reach-goal-dis 1.0 \
  --collision-dis 0.3 \
  --ros-port 11312 \
  --device cuda:0 \
  --skip-existing
```

如果时间充足，建议跑 100 seeds，更接近论文级统计：

```bash
cd /home/csj/msd/orp0415/orp

source /opt/ros/noetic/setup.bash
source /home/csj/anaconda3/etc/profile.d/conda.sh
conda activate orp

export TMPDIR=/media/share/csj/msd/orp_tmp
mkdir -p "$TMPDIR" /media/share/csj/msd/orp_eval

python scripts/eval_p2m_ros_batch.py \
  --run-name p2m_paper_eval_100seeds_$(date +%Y%m%d_%H%M%S) \
  --out-root /media/share/csj/msd/orp_eval \
  --checkpoints balanced_367067136,high_success_170459136 \
  --scenarios dyn7_static7,dyn13_static13,dyn25_static19,dyn25_static0,dyn0_static44 \
  --seeds 0:99 \
  --timeout 60 \
  --reach-goal-dis 1.0 \
  --collision-dis 0.3 \
  --ros-port 11312 \
  --device cuda:0 \
  --skip-existing
```

只跑一个场景做调试：

```bash
python scripts/eval_p2m_ros_batch.py \
  --run-name debug_dyn13_seed0_$(date +%Y%m%d_%H%M%S) \
  --out-root /media/share/csj/msd/orp_eval \
  --checkpoints balanced_367067136 \
  --scenarios dyn13_static13 \
  --seeds 0 \
  --timeout 60 \
  --ros-port 11312 \
  --device cuda:0
```

只生成命令、不实际启动 ROS：

```bash
python scripts/eval_p2m_ros_batch.py \
  --run-name dryrun_$(date +%Y%m%d_%H%M%S) \
  --out-root /media/share/csj/msd/orp_eval \
  --checkpoints balanced_367067136,high_success_170459136 \
  --scenarios dyn13_static13 \
  --seeds 0:1 \
  --dry-run
```

## 8. 输出文件

每次批量实验都会输出到 16TB 盘：

```text
/media/share/csj/msd/orp_eval/<run-name>/
```

主要结果文件：

| 文件 | 内容 |
|---|---|
| `config.json` | 本次实验参数 |
| `results.csv` | 每个 seed 的原始统计 |
| `summary.csv` | 按 checkpoint + 场景聚合后的表格 |
| `summary.md` | 可直接读的中文 markdown 总结 |
| `trials/*/monitor.log` | 单次 trial 的 JSON 结果 |
| `trials/*/infer.log` | 对应推理日志 |
| `trials/*/map.log` | 地图/障碍物仿真日志 |
| `trials/*/lidar.log` | LiDAR raycast 日志 |

## 9. 怎么判断有没有超过论文

如果目标是超过论文 P2M，本项目在每个场景的成功率至少要大于：

| 场景 | P2M 论文成功率 | 需要超过 |
|---|---:|---:|
| 7 dynamic + 7 static | 95% | `>95%` |
| 13 dynamic + 13 static | 65% | `>65%` |
| 25 dynamic + 19 static | 40% | `>40%` |
| 25 dynamic only | 50% | `>50%` |
| 44 static only | 60% | `>60%` |

如果目标是只超过非 P2M 算法里最好的一个：

| 场景 | 最强非 P2M 基线 | 需要超过 |
|---|---:|---:|
| 7 dynamic + 7 static | FAPP 90% | `>90%` |
| 13 dynamic + 13 static | FAPP/NavRL/Obsnet 30% | `>30%` |
| 25 dynamic + 19 static | FAPP 25% | `>25%` |
| 25 dynamic only | Obsnet 25% | `>25%` |
| 44 static only | NavRL 80% | `>80%` |

严格比较时不要只看成功率，还要同时看：

```text
1. 成功率 eta 是否更高
2. 成功样本平均速度 va 是否接近或更高
3. 路径效率 Rl 是否接近或更小
4. 安全距离 ds 是否接近或更大
5. tp 延迟是否在同一插桩方式下统计
```

## 10. 当前注意事项

1. 目前批量脚本统计 `eta/va/Rl/ds`，但暂未公平统计论文里的 `tp` 延迟。要比较 `tp`，建议后续在 `infer.py::lidar_callback` 里对 `prepare_input + policy + safety_shield + acccmd_2_odom` 做计时并输出平均值。
2. 当前 `ds` 来自 `/ray2array_hits`，它是在线 LiDAR 估计值，不一定和论文完全相同；如果论文代码另有离线最近距离统计，应以离线统计为准。
3. 批量脚本使用独立 ROS master：`11312`。如果机器上已有同端口 ROS，请换 `--ros-port`。
4. 所有新实验输出都在 `/media/share/csj/msd/orp_eval`，临时文件在 `/media/share/csj/msd/orp_tmp`，不会写到旧项目 `/home/csj/orp`。
5. 如果中途中断，可以重新执行同一个命令并带 `--skip-existing`，脚本会跳过已经有 `result.json` 的 trial。

## 11. 2026-08-10 完整 20 seeds 实验结果

本轮实验已完成：

```text
run 目录：
/media/share/csj/msd/orp_eval/p2m_paper_eval_20seeds_final_20260810_174043

实验规模：
2 个 checkpoint x 5 个论文场景 x 20 个 seeds = 200 trials

错误 trial：
0 / 200

安全距离计算：
/ray2array_hits 按 xyz hit point 处理，用 hit 点到当前 odom 位置的欧氏距离作为 LiDAR safety distance。

失败终止：
正式批量实验中，一旦连续 5 帧距离低于 0.3 m，trial 立即记为 collision failure。
```

结果文件：

```text
/media/share/csj/msd/orp_eval/p2m_paper_eval_20seeds_final_20260810_174043/results.csv
/media/share/csj/msd/orp_eval/p2m_paper_eval_20seeds_final_20260810_174043/summary.csv
/media/share/csj/msd/orp_eval/p2m_paper_eval_20seeds_final_20260810_174043/summary.md
```

### 11.1 和 P2M 论文成功率对比

| Checkpoint | 场景 | N | 成功率 | P2M 论文 | 与 P2M 差距 | 最强非 P2M | 与最强非 P2M 差距 |
|---|---|---:|---:|---:|---:|---:|---:|
| `balanced_367067136` | 7 dynamic + 7 static | 20 | 65% | 95% | -30% | 90% | -25% |
| `balanced_367067136` | 13 dynamic + 13 static | 20 | 5% | 65% | -60% | 30% | -25% |
| `balanced_367067136` | 25 dynamic + 19 static | 20 | 0% | 40% | -40% | 25% | -25% |
| `balanced_367067136` | 25 dynamic only | 20 | 40% | 50% | -10% | 25% | +15% |
| `balanced_367067136` | 44 static only | 20 | 5% | 60% | -55% | 80% | -75% |
| `high_success_170459136` | 7 dynamic + 7 static | 20 | 0% | 95% | -95% | 90% | -90% |
| `high_success_170459136` | 13 dynamic + 13 static | 20 | 0% | 65% | -65% | 30% | -30% |
| `high_success_170459136` | 25 dynamic + 19 static | 20 | 0% | 40% | -40% | 25% | -25% |
| `high_success_170459136` | 25 dynamic only | 20 | 0% | 50% | -50% | 25% | -25% |
| `high_success_170459136` | 44 static only | 20 | 5% | 60% | -55% | 80% | -75% |

结论：

```text
1. 两个 checkpoint 都没有超过 P2M 论文结果。
2. balanced_367067136 明显强于 high_success_170459136。
3. balanced_367067136 只在 25 dynamic only 场景超过了最强非 P2M 基线：40% vs 25%。
4. balanced_367067136 在 7+7 场景能达到 65%，但仍低于 P2M 的 95% 和 FAPP 的 90%。
5. 高密混合场景 25 dynamic + 19 static 是当前最弱项，两个 checkpoint 都是 0%。
```

### 11.2 成功样本质量

| Checkpoint | 场景 | 成功率 | 成功样本速度 (m/s) | 成功样本路径效率 | 成功样本安全距离 (m) |
|---|---|---:|---:|---:|---:|
| `balanced_367067136` | 7 dynamic + 7 static | 65% | 2.47 | 1.31 | 0.64 |
| `balanced_367067136` | 13 dynamic + 13 static | 5% | 2.71 | 1.25 | 0.53 |
| `balanced_367067136` | 25 dynamic only | 40% | 2.53 | 1.42 | 0.38 |
| `balanced_367067136` | 44 static only | 5% | 3.06 | 1.15 | 0.40 |
| `high_success_170459136` | 44 static only | 5% | 2.99 | 1.13 | 0.22 |

没有成功样本的场景不统计速度、路径效率和安全距离。

### 11.3 到达率与碰撞率

| Checkpoint | 场景 | 到达率 | 碰撞率 |
|---|---|---:|---:|
| `balanced_367067136` | 7 dynamic + 7 static | 65% | 25% |
| `balanced_367067136` | 13 dynamic + 13 static | 5% | 95% |
| `balanced_367067136` | 25 dynamic + 19 static | 0% | 100% |
| `balanced_367067136` | 25 dynamic only | 40% | 60% |
| `balanced_367067136` | 44 static only | 5% | 75% |
| `high_success_170459136` | 7 dynamic + 7 static | 0% | 45% |
| `high_success_170459136` | 13 dynamic + 13 static | 0% | 90% |
| `high_success_170459136` | 25 dynamic + 19 static | 0% | 100% |
| `high_success_170459136` | 25 dynamic only | 0% | 90% |
| `high_success_170459136` | 44 static only | 5% | 95% |

### 11.4 训练与测试不一致的现象

`checkpoint_170459136.pt` 在训练日志附近的 success 略高，但 ROS 论文式测试几乎全场景失败。这说明训练环境里的 success 曲线不能直接代表 P2M ROS benchmark 表现，尤其是安全距离、真实 raycast 输入、起终点/障碍物分布和失败终止条件都会放大 sim-to-test gap。

当前推荐继续以 `checkpoint_367067136.pt` 为主做后续改进，因为它在完整 20 seeds 测试中全面优于 `checkpoint_170459136.pt`。
