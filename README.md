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
python3 sender.py --cc-mode qlearning --window-size 1 --max-cwnd 80 --qtable-file q_table.json

# DQN：输入连续状态 [精确 RTT, 丢包率百分比, 当前 CWND]，神经网络输出 CWND 乘数动作
python3 sender.py --cc-mode dqn --window-size 8 --max-cwnd 80 --dqn-model-file dqn_model.pt
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
       - rtt_weight * avg_rtt_ms
```

这组参数仍奖励吞吐，但会显式重罚 RTO timeout。默认下，fast retransmit 主要承担 `retx_weight` 惩罚，而 timeout 会同时承担 `timeout_weight + retx_weight`，更符合 timeout 比快速重传更伤性能的常识。若希望 AI 更激进，可把 `--reward-timeout-weight` 降到 6 或把 `--reward-rtt-weight` 降到 0.010；若希望更稳，可把 timeout 权重提高到 15。

多轮训练默认只在终端输出每轮指标摘要，不显示 ACK、重传和学习细节日志：

```bash
python3 train_q_learning.py \
  --rounds 50 \
  --packets 300 \
  --q-table q_table.json \
  --checkpoint-dir artifacts/checkpoints/qlearning \
  --summary-file artifacts/training/qlearning_summary.csv
```

输出格式示例：

```text
Q-Learning training:   2%|...| 1/50 [..]
[TRAIN] round=1/50 epsilon=0.300 acked=300/300 duration=...s throughput=...Mbps avg_rtt=...ms srtt=...ms retx=... fast=... timeout=... ckpt=artifacts/checkpoints/qlearning/...
```

每轮结束后会保存一份 Q-Table checkpoint，并将 round、epsilon、checkpoint、吞吐、RTT、重传等指标追加到 summary CSV。进度条使用 `tqdm`；若未安装，脚本会提示 `python3 -m pip install tqdm` 并退回普通逐轮输出。默认 50 轮足够观察趋势；如果时间允许，建议改成 `--rounds 100`。可用 `--checkpoint-every 2` 改为每 2 轮保存一次。

如需调试每个 ACK、FAST 重传或 Q-Learning 动作，可额外加 `--verbose-sender`。

记录和绘图：

```bash
python3 sender.py \
  --cc-mode qlearning \
  --packets 200 \
  --metrics-file artifacts/metrics/metrics.csv \
  --history-file artifacts/metrics/history.csv \
  --plot-file artifacts/plots/qlearning_plot.png
```

分别运行 AIMD 与 Q-Learning 后，可生成对比图：

```bash
python3 plot_metrics.py \
  --metrics-file artifacts/metrics/metrics.csv \
  --history-file artifacts/metrics/history.csv \
  --output artifacts/plots/comparison.png
```

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
11. DQN 模式废弃离散 Q-Table，使用 PyTorch 构建三层全连接网络；输入为连续浮点状态 `[RTT(ms), loss_percent, CWND]`，输出 5 个动作的 Q 值，动作分别对应 `CWND × 0.50 / 0.75 / 1.00 / 1.25 / 1.50`。

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
  --dqn-model-file dqn_model.pt \
  --link-bandwidth-drop-after-packets 150 \
  --link-bandwidth-drop-factor 0.5 \
  --metrics-file artifacts/metrics/dqn_metrics.csv \
  --history-file artifacts/metrics/dqn_history.csv
```

多轮训练：

```bash
python3 train_dqn.py \
  --rounds 50 \
  --packets 160 \
  --dqn-model dqn_model.pt \
  --checkpoint-dir artifacts/checkpoints/dqn \
  --summary-file artifacts/training/dqn_summary.csv
```

DQN 训练默认同样只输出每轮指标摘要，例如：

```text
DQN training:   2%|...| 1/50 [..]
[DQN-TRAIN] round=1/50 epsilon=0.350 acked=160/160 duration=...s throughput=...Mbps avg_rtt=...ms srtt=...ms retx=... fast=... timeout=... ckpt=artifacts/checkpoints/dqn/...
```

每轮结束后会保存一份模型 checkpoint，并将 round、epsilon、checkpoint、吞吐、RTT、重传等指标追加到 summary CSV。如需观察发送端日志中的 `[SENDER][DQN]` 连续状态、动作编号、CWND 乘数、新 CWND、经验池大小和 reward，可额外加 `--verbose-sender`。训练后的最新模型权重保存在 `dqn_model.pt`。

## 环境要求

核心可靠传输、AIMD 和 Q-Learning 功能只依赖 Python 标准库。若需要训练进度条，建议安装 `tqdm`；若需要生成 PNG 图表，需要安装 `matplotlib`；若需要启用 DQN 深度强化学习模式，需要安装 `torch`。
