# 四个 10000 轮对比实验

## 目的

这组实验用于比较 MGDP-lite v2 输入、原版 P2M 输入/网络/奖励、原版奖励以及去掉 ch3 后的效果。四个实验都从随机参数开始训练，并使用当前仓库里同一套 Isaac 训练环境和 curriculum。

## 实验设置

| 实验 | 输入 | 网络 | 奖励 |
| --- | --- | --- | --- |
| exp1 | `mgdp_lite_v2` | 当前 ORP 网络 | 当前修改后的 `p2m` 奖励 |
| exp2 | 原版 P2M 三通道输入 | 原版 P2M CNN+MLP actor/critic 风格 | 原版 P2M 奖励 |
| exp3 | `mgdp_lite_v2` | 当前 ORP 网络 | 原版 P2M 奖励 |
| exp4 | `mgdp_lite_v2_no_ch3` | 当前 ORP 网络 | 当前修改后的 `p2m` 奖励 |

说明：exp2 没有直接在原版 P2M 仓库里训练，因为那样训练环境就和另外三个实验不一致。当前仓库增加了 `p2m_original` 模式，用原版 P2M 的输入、PPO 超参数、critic 结构和奖励公式，同时保留当前训练环境。

## 启动命令

```bash
cd /home/csj/msd/orp0415/orp
bash scripts/run_four_10000_experiments.sh
```

脚本会自动使用四张卡：

| GPU | 实验 |
| --- | --- |
| 0 | exp1 |
| 1 | exp2 |
| 2 | exp3 |
| 3 | exp4 |

## 保存位置

所有运行文件都放在 16T 硬盘：

```bash
/media/share/csj/msd/orp_runs/four_exp_时间戳/
```

临时文件和缓存也放在 16T 硬盘：

```bash
/media/share/csj/msd/orp_tmp/four_exp_时间戳/
```

每个实验目录里有：

```text
train.log      # 终端完整日志
command.txt    # 本实验实际使用的参数
pid.txt        # 进程号
wandb/         # offline wandb 和 checkpoint
```

`.pt` checkpoint 会在对应实验目录的 `wandb/offline-run-*/files/` 下，每 500 iter 保存一次，最终保存 `checkpoint_final.pt`。

## 查看训练

先找到最新一组实验：

```bash
ls -td /media/share/csj/msd/orp_runs/four_exp_* | head -1
```

实时看某个实验：

```bash
tail -f /media/share/csj/msd/orp_runs/four_exp_时间戳/exp1_mgdpv2_orp_current_reward/train.log
```

看四个进程是否还在：

```bash
cat /media/share/csj/msd/orp_runs/four_exp_时间戳/exp*/pid.txt
ps -fp $(cat /media/share/csj/msd/orp_runs/four_exp_时间戳/exp*/pid.txt)
```

## 停止训练

```bash
kill $(cat /media/share/csj/msd/orp_runs/four_exp_时间戳/exp*/pid.txt)
```

如果只停一个实验：

```bash
kill $(cat /media/share/csj/msd/orp_runs/four_exp_时间戳/exp2_original_p2m_input_net_reward/pid.txt)
```

突然停止时，已经到达 `save_interval=500` 的 checkpoint 会保留；如果刚好还没到下一次保存点，中间这段训练不会额外保存。
