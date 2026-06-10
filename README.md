# 基于 UDP 的应用层可靠传输与 AI 拥塞控制

本项目对应大作业题目五，实现基于 UDP 的应用层可靠传输、虚拟瓶颈链路、AIMD 拥塞控制基线、Q-Learning 智能拥塞控制与 DQN 深度强化学习控制。

- Header 封装：在 UDP Payload 中自定义 `Sequence Number + Timestamp + Payload`。
- 定时重传：发送端维护未确认队列，超过 RTO 后自动重传。
- RTT/SRTT 采样：Sender 收到 ACK 后用当前时间减去包头 Timestamp 得到 RTT，并平滑计算 SRTT。
- 多线程控制：发送线程、ACK 接收线程、定时器线程协同工作。
- 拥塞控制：支持固定窗口、AIMD、Q-Learning 与 DQN 四种模式，动态调整 CWND。
- 数据记录与可视化：输出 CSV 指标；安装 `matplotlib` 后可生成 CWND/RTT 图。
- 快速重传与乱序处理：连续 3 个重复 ACK 立即重传缺失分组，接收端缓存乱序到达的数据包。

## 目录说明

- `protocol.py`：定义数据包和 ACK 的二进制封装/解析。
- `virtual_link.py`：模拟固定带宽和有限缓存的虚拟瓶颈链路。
- `sender.py`：可靠发送端，负责滑动窗口发送、ACK 处理、RTO 重传、AIMD/Q-Learning/DQN CWND 控制和指标输出。
- `receiver.py`：接收端，负责解析数据包、维护累计 ACK 并返回确认。
- `plot_metrics.py`：读取 `artifacts/metrics/metrics.csv` 和 `artifacts/metrics/history.csv`，生成 AIMD/Q-Learning 对比图。
- `artifacts/`：统一保存训练指标、checkpoint、绘图结果和一键演示产物。
- `题目五实验报告.md`：仅保留可靠传输底层架构说明。
- `答辩讲稿提纲.md`：仅保留可靠传输部分的答辩讲稿。

## 产物位置

| 类型 | 位置 |
| --- | --- |
| 当前 Q-Learning 权重 | `artifacts/models/active/q_table.json` |
| 当前 DQN 权重 | `artifacts/models/active/dqn_model.pt` |
| 待安装候选权重 | `artifacts/models/candidates/` |
| 权重备份 | `artifacts/models/backups/` |
| 训练指标 CSV | `artifacts/training/` |
| Q-Learning checkpoint | `artifacts/checkpoints/qlearning/` 或 `artifacts/checkpoints/q_curriculum_时间戳/` |
| DQN checkpoint | `artifacts/checkpoints/dqn/` |
| 单次绘图输出 | `artifacts/plots/` |
| 一键 demo 图片和 HTML | `artifacts/demo_results/时间戳/` |
| 报告/Poster 使用图片 | `report/figures/` |
| 最终报告 PDF 与 Poster PPTX | `report/output/` |
| 旧版零散输出归档 | `artifacts/legacy/` |

根目录下旧的 `q_table.json`、`q_table_good.json`、`q_table_curriculum.json` 和 `dqn_model.pt` 仅作为兼容副本保留；新训练和演示默认读写 `artifacts/models/active/`。

## 协议格式

数据包格式：

| 字段 | 大小 | 含义 |
| --- | --- | --- |
| Sequence Number | 4 Bytes | 分组序号 |
| Timestamp | 8 Bytes | 发送端写入的 Unix 时间戳 |
| Payload | 1024 Bytes | 固定长度负载 |

ACK 格式：

| 字段 | 大小 | 含义 |
| --- | --- | --- |
| ACK Number | 4 Bytes | 当前累计确认到的最大连续序号，`-1` 表示尚无连续分组 |

## 运行方式

### 一键演示

在 macOS 上可以直接双击：

```bash
Run_Demo.command
```

它会自动启动 Receiver，依次验证 AIMD、Q-Learning、可选 DQN 和带宽突变场景，并在 `artifacts/demo_results/时间戳/` 下生成：

- `index.html`：可视化总览页，包含 ACK 成功率、吞吐量、RTT、重传次数和日志入口。
- `comparison_main.png`：AIMD / Q-Learning / DQN 对比图。
- `comparison_drop.png`：带宽减半场景图。
- `logs/`：每个场景的 Sender / Receiver 日志。

也可以在终端运行：

```bash
python3 demo_runner.py
```

一键演示默认每个场景发送 300 个分组；如需快速试跑，可改为：

```bash
python3 demo_runner.py --packets 50
```

先启动接收端：

```bash
python3 receiver.py --host 127.0.0.1 --port 9001 --initial-seq 0
```

再启动发送端：

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001
```

可调参数：

```bash
python3 sender.py \
  --target-host 127.0.0.1 \
  --target-port 9001 \
  --packets 40 \
  --window-size 8 \
  --cc-mode aimd \
  --rto 0.2
```

四种拥塞控制模式：

```bash
# 固定窗口，便于验证可靠传输底层
python3 sender.py --cc-mode fixed --window-size 8

# AIMD 基线：ACK 加性增，RTO/快速重传乘性减
python3 sender.py --cc-mode aimd --window-size 1 --max-cwnd 80

# Q-Learning：按 RTT 趋势和丢包事件离散成 6 个状态，动作是保持、CWND+1、CWND/2
python3 sender.py --cc-mode qlearning --window-size 1 --max-cwnd 64 --q-eval

# DQN：输入 RTT、RTT 趋势、丢包/timeout、CWND 和 ACK 利用率，神经网络输出 CWND 调整动作
python3 sender.py --cc-mode dqn --window-size 8 --max-cwnd 80
```

Reward 可调参数：

```bash
python3 sender.py \
  --cc-mode qlearning \
  --reward-throughput-weight 1.0 \
  --reward-timeout-weight 10.0 \
  --reward-retx-weight 2.0 \
  --reward-rtt-weight 0.015
```

当前默认 reward 为：

```text
reward = throughput_weight * acked_packets
       - timeout_weight * timeout_events
       - retx_weight * retransmissions
       - rtt_weight * max(0, avg_rtt_ms - reward_target_rtt_ms)
```

默认训练环境与一键演示保持一致：300 个分组、`initial_cwnd=1`、`max_cwnd=64`、2% 随机丢包、10ms 延迟和 3ms 抖动。模型按最近 10 轮平均分挑选，避免训练环境与最终展示不一致，也减少单轮随机低丢包对选模的干扰。

推荐直接跑一次完整训练。脚本会自动备份旧 active Q-Table、重置为 6 状态初始表、进行 100 轮 epsilon 衰减训练、保存每 5 轮 checkpoint、按“吞吐高、RTT 低、重传少、timeout 少”的综合分数挑选最佳表，并把最佳表安装回 `artifacts/models/active/q_table.json`，最后用贪心策略评估 5 轮：

```bash
python3 train_q_learning.py
```

如需从当前 active Q-Table 继续训练而不是重新开始，加 `--continue-q-table`。如需快速调参，可使用：

```bash
python3 train_q_learning.py --fast
```

`--fast` 会在不覆盖显式参数的前提下，把默认轮数降到 20、包数降到 60、Receiver delay 降到 5 ms、虚拟链路服务间隔降到 2 ms，并改为最后评估 2 轮。

两阶段 curriculum 脚本仍可使用，适合做额外对照：

```bash
python3 train_q_curriculum.py --install
```

该命令会生成候选表 `artifacts/models/candidates/q_table_good.json`，并在 `--install` 时备份旧的 active Q-Table 后安装新策略到 `artifacts/models/active/q_table.json`。当前 Q-Learning 严格使用题目要求的 6 个状态：`RTT 趋势(变大/变小/平稳) × 是否发生丢包/重传`。

输出格式示例：

```text
Q-Learning training:   1%|...| 1/100 [..]
[TRAIN] round=1/100 epsilon=0.350 acked=180/180 duration=...s throughput=...Mbps avg_rtt=...ms srtt=...ms retx=... fast=... timeout=... score=... ckpt=...
[TRAIN] installed best Q-table from round=... score=... to artifacts/models/active/q_table.json
[EVAL] round=1/5 acked=180/180 duration=...s throughput=...Mbps avg_rtt=...ms retx=... timeout=... score=...
```

每轮都会将 round、epsilon、checkpoint、吞吐、RTT、重传等指标追加到 summary CSV；默认每 5 轮保存 checkpoint，并额外持续维护 `artifacts/models/candidates/q_table_best.json`。进度条使用 `tqdm`；若未安装，脚本会提示 `python3 -m pip install tqdm` 并退回普通逐轮输出。

如需调试每个 ACK、FAST 重传或 Q-Learning 动作，可额外加 `--verbose-sender`。

记录和绘图：

```bash
python3 sender.py \
  --cc-mode qlearning \
  --packets 200 \
  --q-eval \
  --metrics-file artifacts/metrics/metrics.csv \
  --history-file artifacts/metrics/history.csv \
  --plot-file artifacts/plots/qlearning_plot.png
```

分别运行 AIMD 与 Q-Learning 后，可生成对比图：

```bash
python3 plot_metrics.py \
  --metrics-file artifacts/metrics/metrics.csv \
  --history-file artifacts/metrics/history.csv \
  --output artifacts/plots/comparison.png \
  --smooth-window 5
```

`plot_metrics.py` 会保留 Q-Learning 的原始 CWND 曲线，并额外叠加一条移动平均曲线。原始曲线用于展示真实探索、丢包和 timeout 反馈，移动平均曲线用于观察整体控制趋势；若要关闭平滑，设置 `--smooth-window 1`。

## 虚拟瓶颈链路

发送端内置了一个虚拟漏斗模块，用来模拟“固定带宽 + 有限队列”的瓶颈链路：

- 默认带宽约为 100 包/秒，即每 10ms 漏出一个包。
- 默认队列容量为 20 个包，超过容量的分组会直接丢弃。
- 发送端的 `sendto()` 会先进入虚拟队列，再由后台线程按固定速率真正发送到网卡。
- 可选启用中途带宽突变，按已转发包数触发带宽下降，用于观察拥塞控制算法恢复过程。

运行示例：

```bash
python3 sender.py \
  --target-host 127.0.0.1 \
  --target-port 9001 \
  --window-size 30 \
  --link-queue-capacity 20 \
  --link-service-delay-ms 10
```

中途将带宽减半：

```bash
python3 sender.py \
  --cc-mode qlearning \
  --packets 200 \
  --q-eval \
  --link-bandwidth-drop-after-packets 100 \
  --link-bandwidth-drop-factor 0.5
```

如果想临时关闭虚拟链路，可加上：

```bash
python3 sender.py --disable-virtual-link
```

## 实现要点

1. `protocol.py` 使用 `struct.pack` / `struct.unpack` 完成 Header 与 ACK 的二进制封装。
2. `sender.py` 使用 `unacked` 字典保存所有未确认分组的 Payload、最近发送时间和发送次数。
3. 主线程按照当前 `CWND` 控制新分组发送，避免无限制注入 UDP 数据。
4. ACK 线程持续 `recvfrom()`，收到累计 ACK 后删除所有 `seq <= ack_number` 的未确认分组。
5. ACK 线程根据被确认分组保存的 `wire_timestamp` 计算 RTT，并使用 `SRTT = 0.875 * SRTT + 0.125 * RTT` 平滑。
6. Timer 线程周期扫描 `unacked`，若 `now - last_send >= rto`，则重新封包并发送。
7. ACK 线程统计重复累计 ACK，连续 3 次相同 ACK 时立即快速重传 `ack_number + 1`。
8. `receiver.py` 维护 `expected_seq`，按序推进累计确认号，乱序分组暂存在缓存集合中。
9. AIMD 模式在收到 ACK 时执行 `CWND += 1 / CWND`，发生 RTO 或快速重传时执行 `CWND = max(1, CWND / 2)`。
10. Q-Learning 模式用 `RTT 趋势 × 是否重传` 形成 6 个状态，每个控制周期用吞吐、timeout、重传和 RTT 计算奖励并更新 Q-Table。
11. DQN 模式废弃离散 Q-Table，使用 PyTorch 构建三层全连接网络；输入为连续浮点状态 `[RTT(ms), RTT趋势%, loss%, timeout%, CWND, ACK利用率]`，输出 5 个动作的 Q 值，动作分别对应 `CWND × 0.70 / 0.90 / 1.00 / 1.10 / 1.25`。增窗动作每个控制周期最多增加 1 个窗口单位，避免随机探索时快速打爆虚拟队列。

## DQN 深度强化学习模式

DQN 模式需要额外安装 PyTorch：

```bash
pip install torch
```

单轮运行：

```bash
python3 sender.py \
  --cc-mode dqn \
  --packets 300 \
  --window-size 8 \
  --max-cwnd 80 \
  --epsilon 0.20 \
  --dqn-model-file artifacts/models/active/dqn_model.pt \
  --link-bandwidth-drop-after-packets 150 \
  --link-bandwidth-drop-factor 0.5 \
  --metrics-file artifacts/metrics/dqn_metrics.csv \
  --history-file artifacts/metrics/dqn_history.csv
```

多轮训练：

```bash
python3 train_dqn.py \
  --reset-dqn-model \
  --rounds 100 \
  --packets 180 \
  --window-size 5 \
  --max-window 48 \
  --dqn-model artifacts/models/active/dqn_model.pt \
  --checkpoint-dir artifacts/checkpoints/dqn \
  --summary-file artifacts/training/dqn_summary.csv
```

`--reset-dqn-model` 会先把旧模型备份到 `artifacts/models/backups/`，再从新网络结构重新训练。重新训练 DQN 时建议加上它，避免沿用旧的保守模型。

快速调参时可直接使用：

```bash
python3 train_dqn.py --fast
```

DQN 的 `--fast` 会把默认包数降到 80、Receiver delay 降到 5 ms、虚拟链路服务间隔降到 2 ms，并使用较小 batch/replay 设置，适合先看训练方向是否正常；正式对比再改回默认或手动指定更大的 `--packets`。

当前 DQN 训练默认走低时延吞吐平衡配置：`--epsilon 0.16`、`--window-size 5`、`--max-window 48`、`--reward-throughput-weight 2.1`、`--reward-timeout-weight 16.0`、`--reward-retx-weight 2.5`、`--reward-rtt-weight 0.08`、`--reward-target-rtt-ms 30.0`。RTT 在 30ms 以下不扣分，超过 30ms 的部分才进入 RTT 惩罚；这样会鼓励模型提高吞吐，同时避免排队延迟明显失控。脚本还会检测异常轮次；如果某轮 `duration > 30s` 或 `retransmissions > 1000`，会自动降低后续探索率。

DQN 训练默认同样只输出每轮指标摘要，例如：

```text
DQN training:   1%|...| 1/100 [..]
[DQN-TRAIN] round=1/100 epsilon=0.160 acked=180/180 duration=...s throughput=...Mbps avg_rtt=...ms srtt=...ms retx=... fast=... timeout=... ckpt=artifacts/checkpoints/dqn/...
```

每轮结束后会保存一份模型 checkpoint，并将 round、epsilon、checkpoint、吞吐、RTT、重传等指标追加到 summary CSV。如需观察发送端日志中的 `[SENDER][DQN]` 连续状态、动作编号、CWND 乘数、新 CWND、经验池大小和 reward，可额外加 `--verbose-sender`。训练后的最新模型权重保存在 `artifacts/models/active/dqn_model.pt`。如果只想评估模型而不继续在线训练或覆盖权重，给 `sender.py` 加 `--dqn-eval`。

## 环境要求

核心可靠传输、AIMD 和 Q-Learning 功能只依赖 Python 标准库。若需要训练进度条，建议安装 `tqdm`；若需要生成 PNG 图表，需要安装 `matplotlib`；若需要启用 DQN 深度强化学习模式，需要安装 `torch`。
