# MGDP-lite v2（第二阶段）代码修改指导规范

> 目标：在**不修改 PPO 后端网络主体、不修改 P2M reward、不加载旧 checkpoint**的前提下，将当前 `mgdp_lite` 四通道前端升级为一个信息互补、物理意义更明确的 `mgdp_lite_v2`。
>
> 本文档用于指导 Codex 修改当前工程。优先保证：**旧 `p2m` 模式完全不受影响、现有 `mgdp_lite` v1 可继续复现、新 v2 的训练端与推理端严格一致。**

---

## 0. 本次修改边界

### 0.1 只新增一种输入模式

```text
task.input_mode=mgdp_lite_v2
```

LiDAR 输入尺寸继续保持：

```text
4 × 36 × 6
```

四个通道：

```text
ch0 = denoised proximity        # 去噪后的接近度
ch1 = signed radial velocity    # 有符号径向相对速度
ch2 = 3D flight corridor risk   # 三维飞行走廊风险
ch3 = radial TTC risk           # 径向 TTC / 碰撞紧迫风险
```

后端保持：

```text
原 CNN
+ 原 state 9维
+ 原 PPO actor/critic
+ task.reward_mode=p2m
```

### 0.2 本阶段不要同时修改

- PPO 网络主体；
- actor / critic MLP；
- reward 权重；
- curriculum；
- episode 终止条件；
- action 定义；
- 原 `p2m` 输入；
- 原 `mgdp_lite` v1；
- `p2m` 模式下 NeuFlow 的旧逻辑。

原因：本阶段要做干净的输入消融，只验证观测表示。

### 0.3 从头训练

建议：

```text
resume_checkpoint=null
```

因此：

- 不需要特殊重置第一层 CNN；
- 不需要做旧通道到新通道的权重映射；
- 整个网络从随机初始化开始学习新语义。

---

# 1. 三种模式必须同时保留

```text
task.input_mode=p2m
task.input_mode=mgdp_lite
task.input_mode=mgdp_lite_v2
```

### p2m

保持修改前行为：

```text
ch0 = 当前距离接近度
ch1 = NeuFlow x
ch2 = NeuFlow y
ch3 = 0
```

### mgdp_lite v1

保持当前行为：

```text
ch0 = 当前接近风险
ch1 = median 去噪接近风险
ch2 = 原高度走廊风险
ch3 = 原 radial/动态风险
```

### mgdp_lite_v2

新增：

```text
ch0 = 去噪接近度
ch1 = 有符号径向速度
ch2 = 三维飞行走廊风险
ch3 = 径向 TTC 风险
```

**不要覆盖旧 `mgdp_lite`。**

---

# 2. 四通道设计原则

| 通道 | 回答的问题 |
|---|---|
| ch0 | 障碍在哪里、离我多近？ |
| ch1 | 它正在朝我靠近还是远离？ |
| ch2 | 它是否真正挡在我当前三维飞行路线附近？ |
| ch3 | 如果继续当前趋势，碰撞有多紧迫？ |

目标是避免 v1 中 `ch0` 与 `ch1` 都主要描述距离的重复。

---

# 3. 坐标、shape 与数值约定

## 3.1 LiDAR 图

保持：

```text
36 = 水平方向
6  = 垂直方向
```

实际张量顺序必须沿用当前工程，禁止根据变量名猜维度。

## 3.2 depth

统一视为：

```text
depth ∈ [lidar_min_depth, lidar_range]
```

以下视为 invalid：

- NaN；
- Inf；
- 小于等于最小有效深度；
- 超出最大量程；
- 当前工程已有 invalid 标志。

invalid 方向统一按：

```text
depth = lidar_range
proximity = 0
```

处理，不能把无效点误认为近障碍。

## 3.3 径向速度正负号

全工程统一：

```text
radial_velocity > 0  => 障碍相对无人机正在靠近
radial_velocity = 0  => 径向距离基本不变
radial_velocity < 0  => 障碍相对无人机正在远离
```

训练、推理、可视化、日志必须一致。

---

# 4. 时间缓存：几何和运动必须分开

至少保留：

```text
depth_t
depth_t_minus_1
depth_t_minus_2
```

如果当前已有 radial history，应优先复用。

每个 environment reset 时必须同时清空：

- depth history；
- 上一帧位姿/姿态；
- ego-motion compensation 中间状态；
- TTC/radial 历史；
- 任何滤波缓存。

reset 后最初没有历史时：

```text
ch1 = 0
ch3 = 0
```

---

# 5. ch0：去噪后的接近度

## 5.1 只负责稳定几何

继续使用三帧 median：

```python
depth_stable = median(
    depth_t,
    depth_t_minus_1,
    depth_t_minus_2
)
```

但注意：

> `depth_stable` 只用于几何类信息，不应作为高速动态速度估计的唯一输入。

因为三帧 median 可能带来动态延迟。

## 5.2 接近度

推荐：

```python
proximity = (lidar_range - depth_stable) / lidar_range
proximity = clamp(proximity, 0.0, 1.0)
```

输出：

```text
ch0 ∈ [0, 1]
```

解释：

```text
0 = 远 / 无明显障碍
1 = 非常近
```

---

# 6. ch1：有符号径向相对速度

## 6.1 为什么保留符号

必须区分：

```text
高速靠近
静止
高速远离
```

不能像 v1 一样把负值 clamp 成 0。

## 6.2 运动估计不要直接使用三帧 median 后的差分

推荐使用：

```text
当前 depth
+ 上一帧 depth
+ ego-motion compensation
```

当前设计说明提到 v1 已沿用已有 radial residual 逻辑，因此 Codex 应先定位这部分代码，优先复用它的：

```text
时序残差
运动补偿
```

部分，而不是复用最后的风险裁剪。

如果已有约定：

```python
radial_speed = -residual / dt
```

且满足：

```text
radial_speed > 0 => 靠近
```

则继续使用。

## 6.3 归一化

新增配置：

```yaml
mgdp_v2_radial_speed_scale: 6.0
```

计算：

```python
radial_norm = radial_speed / mgdp_v2_radial_speed_scale
radial_norm = clamp(radial_norm, -1.0, 1.0)
```

输出：

```text
ch1 ∈ [-1, 1]
```

解释：

```text
+1 = 高速靠近
 0 = 径向基本静止
-1 = 高速远离
```

**不要强制映射到 [0,1]。**

---

# 7. ch2：三维飞行走廊风险

## 7.1 目标

v1 主要判断：

```text
障碍是否接近目标飞行高度
```

v2 改成：

```text
障碍是否真正靠近当前未来飞行路线
```

同时考虑：

- 左右偏离；
- 上下偏离；
- 目标方向；
- 障碍距离。

## 7.2 核心思想

不要对所有“同高度”的障碍都敏感。

重点关注：

```text
又近
+
位于目标方向前方
+
靠近未来飞行走廊中心
```

这样减少：

- 上方无关障碍导致过度下降；
- 下方无关障碍导致过度上升；
- 左右远离航线的障碍导致无谓绕行；
- 上下振荡。

## 7.3 推荐最简三维几何

令：

```text
p_hit = LiDAR 命中点相对无人机的位置
g_hat = 当前目标方向单位向量
```

纵向投影：

```python
s = dot(p_hit, g_hat)
```

只重点考虑：

```text
s > 0
```

再算离目标方向轴线的垂直距离：

```python
p_perp = p_hit - s * g_hat
d_perp = norm(p_perp)
```

推荐高斯走廊：

```python
corridor_weight = exp(
    -0.5 * (d_perp / sigma_corridor) ** 2
)
```

前向门控：

```python
front_gate = (s > 0).float()
```

最终：

```python
corridor_risk = (
    proximity
    * corridor_weight
    * front_gate
)
```

输出：

```text
ch2 ∈ [0, 1]
```

## 7.4 配置

```yaml
mgdp_v2_corridor_sigma: 1.0
mgdp_v2_corridor_speed_adaptive: false
mgdp_v2_corridor_speed_gain: 0.12
mgdp_v2_corridor_forward_only: true
```

第一轮消融建议：

```text
speed_adaptive=false
```

后续再单独尝试高速时扩大走廊。

---

# 8. ch3：径向 TTC 风险

## 8.1 第二阶段只做径向 TTC

不要在本阶段直接加入 CPA / 横穿预测。

本阶段只回答：

```text
当前这个方向的障碍，
按照当前距离和靠近速度，
碰撞有多紧迫？
```

## 8.2 TTC 计算

使用 ch1 归一化前的：

```text
radial_speed_mps
```

定义：

```python
closing_speed = clamp(radial_speed_mps, min=0.0)
```

新增配置：

```yaml
mgdp_v2_ttc_min_closing_speed: 0.15
mgdp_v2_ttc_horizon: 4.0
mgdp_v2_ttc_tau: 1.5
```

计算：

```python
ttc = depth_motion / max(closing_speed, eps)
```

其中 `depth_motion` 推荐使用当前有效 `depth_t`，不要用三帧 median 后的 `depth_stable`。

## 8.3 转为风险

推荐：

```python
ttc_risk = exp(-ttc / mgdp_v2_ttc_tau)
```

门控：

```python
valid_ttc = (
    valid_depth
    & (closing_speed > min_closing_speed)
    & (ttc < ttc_horizon)
)

ttc_risk = where(valid_ttc, ttc_risk, 0.0)
```

第一版**不要额外乘 proximity**，避免再次重复 ch0 距离信息。

输出：

```text
ch3 ∈ [0, 1]
```

---

# 9. 推荐数据流

```text
                     LiDAR depth_t
                          │
          ┌───────────────┴────────────────┐
          │                                │
      几何分支                         运动分支
          │                                │
depth_t,t-1,t-2                    depth_t + depth_t-1
          │                                │
      median 去噪                    ego-motion compensation
          │                                │
     depth_stable                      radial_speed
          │                             /        \
          ↓                            ↓          ↓
 ch0 proximity                    ch1 signed     TTC
          │                         radial        │
          │                                      ↓
          └──结合目标方向/点位置──→ ch2        ch3
                                corridor
```

禁止让：

```text
median depth
→ radial speed
```

成为唯一动态路径。

---

# 10. NeuFlow 的处理

## p2m

完全保留 NeuFlow。

## mgdp_lite v1

保持当前逻辑。

## mgdp_lite_v2

本阶段不调用 NeuFlow：

- 不加载 NeuFlow 模型；
- 不做 NeuFlow forward；
- 不构造 pseudo-RGB；
- 不做为 NeuFlow 服务的 resize；
- 推理端同样不调用。

这不是因为 NeuFlow “无用”，而是为了验证：

```text
连续 LiDAR 深度
+
轻量物理结构特征
```

能否替代重型光流前端，并继续提高高难度课程表现。

---

# 11. 建议新增配置项

```yaml
input_mode: p2m

# MGDP-lite v2
mgdp_v2_radial_speed_scale: 6.0

mgdp_v2_corridor_sigma: 1.0
mgdp_v2_corridor_speed_adaptive: false
mgdp_v2_corridor_speed_gain: 0.12
mgdp_v2_corridor_forward_only: true

mgdp_v2_ttc_min_closing_speed: 0.15
mgdp_v2_ttc_horizon: 4.0
mgdp_v2_ttc_tau: 1.5

mgdp_v2_debug: false
mgdp_v2_debug_every: 100
```

不要删除 v1 原配置。

---

# 12. 训练端修改位置

当前设计说明指向：

```text
resources/envs/single/env.py
```

Codex 应：

1. 找到 `input_mode` 分支；
2. 保留 `p2m`；
3. 保留 `mgdp_lite`；
4. 新增 `mgdp_lite_v2`；
5. 尽量将 v2 构造封装成单独 helper，避免 `env.py` 主逻辑膨胀。

建议类似：

```python
def _build_mgdp_lite_v2_observation(...):
    ...
    return lidar_obs
```

具体函数名按当前工程风格调整，不要盲目创建重复模块。

---

# 13. 推理端同步

当前设计说明中还有：

```text
scripts/infer.py
scripts/infer_ros.py
```

两处必须支持：

```text
input_mode=mgdp_lite_v2
```

并与训练端严格一致：

- depth invalid 处理；
- history 长度；
- dt；
- radial 正负号；
- corridor 坐标系；
- TTC 参数；
- reset/首次帧逻辑；
- 四通道顺序。

如果工程允许，优先把 v2 前端抽成公共 helper，避免训练和 ROS 各复制一份公式后逐渐漂移。

---

# 14. dt 必须是真实传感更新周期

不要默认写死：

```python
dt = 0.02
```

优先根据：

```text
physics_dt
× lidar_update_period
```

或当前真实 LiDAR 更新频率得到运动估计的 `dt`。

如果 radial 只每隔若干 step 更新，必须使用真实间隔。

---

# 15. Ego-motion compensation 是最高优先级检查项

无人机自己向前飞时，静态墙的 depth 也会：

```text
5m -> 4m -> 3m
```

如果没有补偿，就会错误得到：

```text
radial_speed > 0
TTC 很危险
```

所以 Codex 必须确认当前 radial residual 是否真的考虑：

- 无人机平移；
- 无人机旋转；
- ray 方向变化；
- frame 对齐。

如果现有实现只是：

```python
depth_prev - depth_now
```

那它只能叫：

```text
apparent radial change
```

不能当作障碍真实相对速度。

本阶段最大潜在 bug 就是：

```text
把无人机自己的前进
误判成
动态障碍迎面冲来
```

因此请优先复用或完善已有 ego-motion compensation。

---

# 16. ch2 的稳定性要求

三维走廊风险必须：

- 使用平滑高斯；
- 不用硬阈值；
- 使用 `depth_stable`；
- 不新增独立 reward；
- 不直接写规则“ch2 高就向上/向下”。

ch2 只是观测，让 PPO 学动作。

这样减少左右/上下振荡风险。

---

# 17. 建议新增日志

## 四通道

```text
obs/ch0_mean
obs/ch0_max

obs/ch1_abs_mean
obs/ch1_pos_ratio
obs/ch1_neg_ratio
obs/ch1_max
obs/ch1_min

obs/ch2_mean
obs/ch2_max
obs/ch2_active_ratio

obs/ch3_mean
obs/ch3_max
obs/ch3_active_ratio
```

## 垂直稳定性

```text
train/stats/abs_vz_mean
train/stats/vz_std
train/stats/abs_az_mean
train/stats/az_std
train/stats/height_error_abs_mean
```

如果容易实现，再加：

```text
train/stats/az_sign_switch_rate
```

用来判断是否出现：

```text
上 -> 下 -> 上 -> 下
```

的高频振荡。

---

# 18. 调试场景

至少验证：

### A. 静态墙 + 无人机悬停

预期：

```text
ch0 高
ch1 ≈ 0
ch2 取决于是否在飞行走廊
ch3 ≈ 0
```

### B. 障碍正面靠近

预期：

```text
ch0 越来越高
ch1 > 0
ch3 越来越高
```

### C. 障碍远离

预期：

```text
ch0 可较高
ch1 < 0
ch3 = 0
```

### D. 上方/侧方但不挡目标方向

预期：

```text
ch0 可能高
ch2 应明显低于正前方同距离障碍
```

### E. 无人机主动前进靠近静态墙

重点检查 ego-motion compensation。

---

# 19. Debug 断言

在 debug 模式至少检查：

```python
assert lidar_obs.shape[-3:] == (4, 36, 6)  # 按工程实际顺序调整
```

范围：

```text
0 <= ch0 <= 1
-1 <= ch1 <= 1
0 <= ch2 <= 1
0 <= ch3 <= 1
```

所有通道不得：

```text
NaN
Inf
```

reset 第一帧至少满足：

```text
ch1 = 0
ch3 = 0
```

---

# 20. 训练建议

从头训练：

```bash
cd scripts

python train.py \
  headless=true \
  wandb.mode=offline \
  task.input_mode=mgdp_lite_v2 \
  task.reward_mode=p2m \
  resume_checkpoint=null \
  task.success_curriculum.enable=true
```

其余参数尽量与之前 `mgdp_lite` v1 实验一致。

---

# 21. 公平对比要求

至少保证：

```text
相同 seed
相同 reward
相同 curriculum
相同 num_envs
相同 episode length
相同 PPO 参数
相同控制器
相同障碍随机化
相同训练总帧数
```

主要比较：

```text
P2M
MGDP-lite v1
MGDP-lite v2
```

---

# 22. 评价指标

不要只看 final success。

## 课程推进

```text
Frames to L1
Frames to L2
Frames to L3
Frames to L4
Frames to L5
Frames to L6
```

最重点：

```text
Frames to L5
```

## 同等级比较

重点看 L4/L5：

```text
mean success
last-50 success
10-point moving-average peak
EMA success
done_safety
done_timeout
return
```

## 飞行稳定性

```text
vz_std
az_std
height_error_abs_mean
```

避免出现：

```text
成功率提升
但上下抖动明显变大
```

---

# 23. 第二阶段成功标准

至少满足一个：

### A

同 L4：

```text
后期成功率 > v1
```

且：

```text
done_safety 不明显变差
```

### B

进入 L5 所需 frames 少于 v1。

### C

L5 稳定成功率高于 v1 当前平台。

### D

成功率相近，但：

```text
完全移除 NeuFlow
+
推理延迟下降
+
控制稳定性不下降
```

也有研究价值。

---

# 24. 本阶段暂时不要做

留到第三阶段：

- 横穿障碍二维角运动；
- CPA；
- 完整未来轨迹交叉；
- 轻量 optical flow；
- NeuFlow + v2 hybrid。

留到后续真正 MGDP 化阶段：

- teacher-student；
- 对比学习；
- privileged 3D teacher map；
- 3D feature alignment；
- reward 重设计。

第二阶段只把：

```text
距离
+
径向运动
+
三维走廊
+
径向碰撞紧迫性
```

做干净。

---

# 25. Codex 建议执行顺序

1. 定位 `p2m` / `mgdp_lite` / radial / history / NeuFlow / ray direction / goal direction 代码；
2. 不修改旧模式；
3. 增加 v2 配置；
4. 实现 ch0；
5. 实现 ch1，并验证正负号；
6. 实现 ch2；
7. 实现 ch3；
8. 拼成 `4×36×6`；
9. 同步 `infer.py`；
10. 同步 `infer_ros.py`；
11. 增加 debug 日志；
12. 先跑 1M～5M frames 做 sanity check；
13. 确认无 NaN、ch1/ch3 非全 0、ch2 非全 0/全 1；
14. 再跑完整 curriculum。

---

# 26. 最终代码审查 Checklist

- [ ] `p2m` 行为与修改前一致；
- [ ] `mgdp_lite` v1 行为与修改前一致；
- [ ] `mgdp_lite_v2` 不调用 NeuFlow；
- [ ] shape 仍为 `4×36×6`；
- [ ] reset 不串 episode 历史；
- [ ] ch1 靠近为正；
- [ ] ch1 保留负值；
- [ ] ch3 使用物理单位 radial speed，而不是用归一化 ch1 反推；
- [ ] ch3 除法有 epsilon；
- [ ] ch3 对远离障碍为 0；
- [ ] ch2 使用平滑权重；
- [ ] ch2 主要关注目标方向前方；
- [ ] invalid LiDAR 不会变成近障碍；
- [ ] 训练与推理 dt 一致；
- [ ] 训练、infer、ROS 通道顺序一致；
- [ ] 所有通道无 NaN/Inf；
- [ ] 新参数有默认值；
- [ ] 从头训练时没有第一层 CNN 特殊处理。

---

# 27. 最终结构

```text
                         LiDAR
                           │
                  current depth_t
                           │
              ┌────────────┴────────────┐
              │                         │
          geometry                  temporal
              │                         │
       3-frame median             t-1 → t residual
              │                         │
        stable depth             ego compensated
              │                         │
              ↓                         ↓
        ch0 proximity            radial_speed_mps
              │                    /          \
              │                   ↓            ↓
              │              ch1 signed       TTC
              │                radial           │
              │                                 ↓
       target direction                     ch3 risk
              │
              ↓
       3D corridor geometry
              │
              ↓
          ch2 corridor
              │
              └────────┬───────────────┘
                       ↓
                 4 × 36 × 6
                       ↓
                   原 CNN
                       ↓
                  + state 9
                       ↓
                     PPO
                       ↓
                 acceleration
```

---

# 28. 方法定位

`MGDP-lite v2` 应理解为：

```text
MGDP 启发的稳定/结构化感知
+
P2M 动态避障任务
+
轻量物理时序特征
```

它暂时没有实现 MGDP 原论文最核心的：

```text
深度特征 ↔ 高度/环境教师特征的对比学习对齐
```

真正的 MGDP 式 teacher-student / privileged 3D feature alignment 建议留到后续独立阶段。

本阶段要清楚回答：

```text
仅靠更合理的四通道观测，
是否就能让 P2M 后端在 L4/L5 上继续提升？
```
