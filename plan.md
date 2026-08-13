# L4 Done-First 改进计划

## Summary
- 用这份内容完整替换项目根目录 `plan.md`。
- 当前主问题按 2026-03-31 日志判定为：`L4 train 有一定成功率，但 eval 几乎 0 成功；eval 主要死于 done_height_low，train 主要卡在 done_safety + done_bound`。按 2026-03-25 日志判定：`done_acc_limit` 暴雷已经是历史问题，不再作为当前主矛盾。
- 阶段顺序保持为：`提速 A/B + 更小 batch` -> `GRU 优先的时序增强` -> `teacher-student 扩展 privileged 信息` -> `推理优先的轻量 safety shield`，但从现在开始所有阶段都以“改善当前 done 问题”为第一目标，而不是单看吞吐或 train 成功率。

## Key Changes
### Phase 0: Done 问题对齐与判据固定
- 把当前 L4 主判据固定为：
  - `eval/stats.done_height_low` 最低
  - `eval/stats.done_success` 最高
  - `train/stats.done_bound` 最低
  - `train/stats.done_safety` 最低
  - `rollout_fps` 只作为第五优先级
- 扩展训练监控，新增固定告警项：
  - `eval_gap_success = train/stats.done_success - eval/stats.done_success`
  - `eval/stats.done_height_low`
  - `train/stats.done_height_low`
  - `train/stats.done_bound`
- 调整 `scripts/monitor_training.py` 阈值，新增：
  - `eval_done_success == 0` 连续 3 次告警
  - `eval_done_height_low > 0.40` 告警
  - `eval_gap_success > 0.25` 告警
- 所有后续阶段都必须保留 train/eval 双视角 done 分解，不允许只汇总总成功率。

### Phase 1: 提速 A/B + 更小 batch，但以 done 改善为硬门槛
- 所有 Phase 1 试验都从“2026-03-31 这次 run 第一次进入 L4 后的第一个 checkpoint”恢复，不从头训。
- 先做吞吐 A/B，配置固定为 4 组：
  - `P0` 基线：当前配置原样。
  - `P1` 小 batch：`env.num_envs=1024`，`algo.train_every=192`。
  - `P2` 小 batch + 感知降频：`env.num_envs=1024`，`algo.train_every=192`，`lidar_update_period=2`，`flow_update_period=3`。
  - `P3` 更激进吞吐：`env.num_envs=896`，`algo.train_every=160`，`lidar_update_period=2`，`flow_update_period=4`。
- 吞吐 A/B 的淘汰规则固定为：
  - 任一配置若 `eval/stats.done_height_low` 比基线更差，直接淘汰。
  - 任一配置若 `train/stats.done_bound` 比基线更差超过 `0.03`，直接淘汰。
  - 在未淘汰配置中，优先选 `eval/stats.done_success` 更高者，再看 `rollout_fps`。
- 在吞吐优胜配置上立刻做 done 修复 A/B，配置固定为 4 组：
  - `D0`：吞吐优胜配置，不改 reward/done。
  - `D1` 高度优先：`goal_use_planar=false`，`reward_weights.w_h=0.65`，`height_gate_min=0.30`，`height_w_floor=4.0`。
  - `D2` 高度 + 降速：在 `D1` 基础上设 `v_ref=2.40`，`acc_ref=8.0`。
  - `D3` 高度 + 走廊：在 `D1` 基础上设 `bound_line_max_dist=15.0`，`bound_soft_margin_ratio=0.65`，`bound_soft_penalty_w=6.0`。
- Phase 1 增加两个轻量代码改动，优先于网络改动：
  - 在 reward 中加入显式 `height_low_penalty` 配置项，默认 `20.0`，在 `height_low` 命中时先罚再终止。
  - 在日志中单独记录 `eval_gap_success` 和 `done_height_low` 的 train/eval 对照。
- Phase 1 通过门槛固定为：
  - `eval/stats.done_height_low <= 0.25`
  - `eval/stats.done_success >= 0.05`
  - `train/stats.done_bound <= 0.14`
  - `rollout_fps` 不低于当前基线的 90%

### Phase 2: GRU 优先的时序增强
- 只做 GRU，不在这一阶段实现 SSM；但接口命名统一为 `TemporalCore`，后续可替换实现。
- 结构固定为：
  - `lidar -> CNN -> _cnn_feature`
  - `_cnn_feature -> GRU -> _cnn_feature_mem`
  - `concat(_cnn_feature_mem, state) -> _feature`
  - `_feature -> actor`
  - `concat(_feature, observation_central) -> critic`
- GRU 参数固定：
  - `hidden_size=128`
  - `num_layers=1`
  - `residual=true`
  - `layer_norm=true`
- rollout 与 PPO 更新必须改成序列式：
  - 每个 env 持有独立 hidden state
  - reset 时仅清空对应 env hidden
  - minibatch 先按 env 切，再按时间连续 chunk
  - `bptt_len=64`
  - 禁止把 `[T, N]` 直接拍平后随机打散
- 新增配置键固定为：
  - `model.temporal.enable`
  - `model.temporal.type=gru`
  - `model.temporal.hidden_size=128`
  - `algo.bptt_len=64`
  - `algo.sequence_num_minibatches=4`
- Phase 2 通过门槛固定为：
  - `eval/stats.done_height_low <= 0.15`
  - `eval/stats.done_success >= 0.10`
  - `train/stats.done_bound` 不高于 Phase 1 最优配置
  - `eval_gap_success` 相比 Phase 1 缩小至少 30%

### Phase 3: 复用现有 privileged 信息的 teacher-student
- 只复用当前 `observation_central`，不引入 planner teacher，不引入第二套 actor。
- teacher 仅在训练期存在，student 仍是唯一 acting policy。
- teacher 分支固定为：
  - 输入 `concat(student_feature_mem, observation_central)`
  - 输出 `teacher_feature`
  - `teacher_feature` 维度与 `student_feature_mem` 相同
- distillation 固定包含两项：
  - `feature_distill_loss = mse(student_feature_mem, teacher_feature.detach())`
  - `priv_recon_loss = smooth_l1(student_priv_pred, observation_central.detach())`
- 保留现有 critic auxiliary head，不新增 teacher actor head。
- loss 权重固定为：
  - 初始 `distill_feature_w=0.05`
  - 初始 `distill_priv_w=0.05`
  - 启用后前 1000 个 L4 iter 线性升到 `0.15 / 0.10`
- Phase 3 通过门槛固定为：
  - `eval/stats.done_success >= 0.15`
  - `eval/stats.done_height_low <= 0.10`
  - `train/stats.done_bound <= 0.10`
  - `eval_gap_success` 再缩小至少 20%

### Phase 4: 推理优先的轻量 safety shield
- shield 先只上推理链路，不进训练环境。
- 挂载位置固定为：`target_acc` 生成后、控制器执行前，同时覆盖本地推理和 ROS 推理。
- shield 只用现有推理量：`min_depth`、当前高度 `z`、`virtual_ground`、`safety_dis`、`target_acc`。
- 规则固定为：
  - 硬碰撞区：`min_depth < safety_dis`，平面加速度归零，`az >= 0`
  - 软谨慎区：`safety_dis <= min_depth < safety_dis + 0.8`，平面加速度线性缩放
  - 地板保护：`z < virtual_ground + 0.35`，禁止负向 `az`，并叠加向上偏置
- shield 输出后统一再裁剪回 `[-acc_ref, acc_ref]`。
- 新增推理日志固定为：
  - `shield_active`
  - `shield_reason`
  - `shield_scale_xy`
  - `shield_floor_bias`
  - `target_acc_before`
  - `target_acc_after`
- Phase 4 通过门槛固定为：
  - 部署评估中 `done_height_low <= 0.05`
  - `done_bound` 不得比 Phase 3 更差
  - `eval/stats.done_success` 下降不得超过 `0.02`

## Test Plan
- Phase 1:
  - 验证 4 组吞吐配置和 4 组 done 修复配置都能从同一 L4 checkpoint 恢复。
  - 对每组生成统一 scoreboard，包含 `train/eval done_success`、`done_height_low`、`done_safety`、`done_bound`、`rollout_fps`。
  - 检查 `height_low_penalty` 开关前后，reward 数值有限且 `done_height_low` 日志正常。
- Phase 2:
  - 验证 hidden state 在单 env reset、全局 reset、eval reset 都正确清零。
  - 验证 sequence minibatch 保持时间顺序，不再使用当前 flatten 随机采样。
  - 验证 `model.temporal.enable=false` 时可完全回退到当前前馈模型。
- Phase 3:
  - 验证推理图不依赖 `observation_central`。
  - 验证 distillation loss 全程有限值，关闭后可回退到 Phase 2。
- Phase 4:
  - 验证近障时平面动作被正确收缩。
  - 验证低高度时 `az` 不再为负。
  - 验证安全区内 shield 不改命令。
  - 验证 shield 触发日志可复盘。

## Assumptions
- 这份计划完整替换现有根目录 `plan.md`。
- 当前基线固定为 2026-03-31 这次 run，不再以 2026-03-25 的 `done_acc_limit` 暴雷配置作为主参考。
- 当前第一目标不是冲到 L5，而是先把 L4 的 `done_height_low`、`done_bound`、`train/eval gap` 压下去。
- Phase 1 允许轻量 reward/logging 改动，但不改网络结构。
- Phase 2 先做 GRU，SSM 只预留接口，不在本轮落地。
- Phase 4 先解决掉高和近障保底，不在这一轮实现完整 CBF/QP 版本 shield。
