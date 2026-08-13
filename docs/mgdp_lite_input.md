# MGDP-lite 输入设计说明

本文档说明新增的 `task.input_mode=mgdp_lite` 输入模式。它的目标是：

```text
MGDP 风格前端观测 + ORP 网络 + P2M reward
```

也就是说：

```text
网络结构不换
reward 不换
只换前端输入的 4 个 LiDAR 通道
```

原来的 `task.input_mode=p2m` 没有去掉，可以随时切回。

## 1. 输入维度

当前 ORP 网络吃两部分观测：

```text
lidar: 4 × 36 × 6
state: 9
```

其中：

```text
36 = 水平方向分成 36 个角度格子
6  = 垂直方向分成 6 个角度格子
4  = 4 张风险图/特征图
```

所以 LiDAR 部分一共是：

```text
4 × 36 × 6 = 864 个数
```

但这些数不是直接拉平成 MLP 输入，而是先作为 `4 × 36 × 6` 的小图输入 CNN；CNN 提取特征后，再和 9 维 `state` 拼接。

## 2. p2m 输入和 mgdp_lite 输入的区别

原来的 P2M/legacy 输入仍然保留：

```text
ch0 = 当前距离接近度
ch1 = NeuFlow 光流 x
ch2 = NeuFlow 光流 y
ch3 = radial 通道
```

在 `task.input_mode=p2m` 下，为了和之前训练保持一致，代码仍然会把 radial 通道置零：

```text
ch3 = 0
```

新增的 MGDP-lite 输入是：

```text
ch0 = 当前接近风险
ch1 = 去噪接近风险
ch2 = 飞行高度走廊风险
ch3 = 动态碰撞/TTC 风险
```

这 4 个通道全部归一化到：

```text
[0, 1]

0 = 安全 / 不明显
1 = 危险 / 很近 / 很重要
```

## 3. 四个通道怎么得到

### 3.1 ch0：当前接近风险

每个 LiDAR 方向有一个距离：

```text
depth_t: 36 × 6
```

距离越近越危险，所以转成接近风险：

```text
risk_now = (lidar_range - depth_t) / lidar_range
```

再裁剪到：

```text
risk_now ∈ [0, 1]
```

例子：

```text
lidar_range = 10m
depth = 1m  -> risk_now = 0.9
depth = 8m  -> risk_now = 0.2
depth = 10m -> risk_now = 0.0
```

### 3.2 ch1：去噪接近风险

只看单帧 LiDAR 容易被噪声骗，所以 MGDP-lite 会保存最近几帧 depth。

默认使用 `lidar_radial_window=3`，也就是：

```text
depth_t
depth_t-1
depth_t-2
```

对这几帧做 median：

```text
depth_denoised = median(depth_t, depth_t-1, depth_t-2)
```

然后同样转成风险：

```text
risk_denoised = (lidar_range - depth_denoised) / lidar_range
```

作用：

```text
保留稳定障碍
压掉偶然出现的假近点
让策略输入更平滑
```

### 3.3 ch2：飞行高度走廊风险

MGDP 原始思想里有 height map。无人机不是贴地走，所以这里把 height map 改成“飞行走廊风险”。

直觉：

```text
如果障碍刚好挡在目标飞行高度附近，就危险
如果障碍明显在上方或下方，就没那么危险
```

先估计每个 LiDAR hit 的世界高度：

```text
hit_z = drone_z + ray_dir_z * depth_denoised
```

再看它和目标高度的差距：

```text
height_error = abs(hit_z - target_z)
```

把差距转成高度风险：

```text
height_risk = 1 - clamp(height_error / mgdp_corridor_half_height, 0, 1)
```

最后再乘以去噪接近风险：

```text
corridor_risk = height_risk * risk_denoised
```

这样只有“又近、又挡在飞行高度层”的障碍才会特别亮。

默认参数：

```text
mgdp_corridor_half_height = 1.0m
```

### 3.4 ch3：动态碰撞/TTC 风险

动态风险必须用多帧 depth 才能算出来。

先用当前去噪 depth 和上一时刻去噪 depth 做运动补偿后的径向速度估计。代码里沿用了已有 radial residual 计算逻辑：

```text
radial_speed = - residual / dt
```

含义：

```text
radial_speed > 0：障碍相对无人机正在靠近
radial_speed < 0：障碍相对无人机正在远离
```

转成靠近风险：

```text
approach_risk = clamp(radial_speed / lidar_radial_max_speed, 0, 1)
```

最后乘以去噪接近风险：

```text
ttc_risk = approach_risk * risk_denoised
```

这样：

```text
远处快速变化不会被过分夸大
近处且正在靠近的方向会明显变亮
```

## 4. 为什么 reward 先不改

这次实验想验证的是：

```text
只换输入，能不能提升 P2M 论文式测试表现
```

所以第一版保持：

```text
task.reward_mode=p2m
```

输入和 reward 的关系可以这样理解：

```text
输入 = 眼睛看到什么
reward = 老师怎么打分
```

reward 当然和输入有关系，因为 reward 惩罚碰撞时，输入里最好能看见碰撞风险。但它们不需要一一对应。

如果同时改输入和 reward，实验变量会变多，最后不好判断到底是哪一部分带来了变化。

## 5. 怎么切换

继续使用原 P2M 输入：

```bash
task.input_mode=p2m
task.reward_mode=p2m
```

使用新增 MGDP-lite 输入：

```bash
task.input_mode=mgdp_lite
task.reward_mode=p2m
```

可以调整飞行高度走廊宽度：

```bash
task.mgdp_corridor_half_height=1.0
```

如果想让高度风险更严格：

```bash
task.mgdp_corridor_half_height=0.6
```

如果想让高度风险更宽松：

```bash
task.mgdp_corridor_half_height=1.5
```

## 6. 训练建议

建议从当前更好的 checkpoint 开始：

```text
/media/share/csj/msd/orp_runs/p2m_train_8000_from_52494336_20260715_111956/wandb/offline-run-20260715_112007-c3400gih/files/checkpoint_367067136.pt
```

训练时只改输入：

```bash
task.input_mode=mgdp_lite
task.reward_mode=p2m
resume_checkpoint=/media/share/csj/msd/orp_runs/p2m_train_8000_from_52494336_20260715_111956/wandb/offline-run-20260715_112007-c3400gih/files/checkpoint_367067136.pt
```

因为 `mgdp_lite` 仍然保持 `4 × 36 × 6`，所以 ORP 网络结构不需要改，旧 checkpoint 可以作为初始化。

但要注意：虽然 shape 一样，通道语义已经变了，所以第一段训练可能会有一段重新适应期。

## 7. 代码位置

训练环境输入构造：

```text
resources/envs/single/env.py
```

仿真 ROS 推理：

```text
scripts/infer.py
```

真实/PointCloud2 ROS 推理：

```text
scripts/infer_ros.py
```

默认配置仍然是：

```text
input_mode: p2m
```

所以不会影响正在运行的旧训练；只有新启动命令显式写：

```bash
task.input_mode=mgdp_lite
```

才会启用新输入。
