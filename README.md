# 基于 UDP 的应用层可靠传输与 Q-Learning 智能拥塞控制

本项目保留题目五中的可靠传输底层实现，并扩展了 Q-Learning 智能拥塞控制器：

- Header 封装：在 UDP Payload 中自定义 `Sequence Number + Timestamp + Payload`。
- 定时重传：发送端维护未确认队列，超过 RTO 后自动重传。
- RTT/SRTT 采样：Sender 收到 ACK 后用当前时间减去包头 Timestamp 得到 RTT，并平滑计算 SRTT。
- 多线程控制：发送线程、ACK 接收线程、定时器线程协同工作。
- 快速重传与乱序处理：连续 3 个重复 ACK 立即重传缺失分组，接收端缓存乱序到达的数据包。
- Q-Learning 拥塞控制：发送端按 1 个 RTT 为周期统计 RTT 趋势和丢包事件，用 6 状态 Q-Table 选择 CWND 保持、加 1 或减半。

## 目录说明

- `protocol.py`：定义数据包和 ACK 的二进制封装/解析。
- `virtual_link.py`：模拟固定带宽和有限缓存的虚拟瓶颈链路。
- `congestion_control.py`：实现固定窗口控制器和 Q-Learning 智能拥塞控制器。
- `sender.py`：可靠发送端，负责滑动窗口发送、ACK 处理和 RTO 重传。
- `receiver.py`：接收端，负责解析数据包、维护累计 ACK 并返回确认。
- `train_q_learning.py`：多轮循环训练脚本，复用 Q-Table 并逐轮调整 epsilon。
- `题目五实验报告.md`：实验报告说明。
- `答辩讲稿提纲.md`：答辩讲稿提纲。

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
  --rto 0.2
```

## 虚拟瓶颈链路

发送端内置了一个虚拟漏斗模块，用来模拟“固定带宽 + 有限队列”的瓶颈链路：

- 默认带宽约为 100 包/秒，即每 10ms 漏出一个包。
- 默认队列容量为 20 个包，超过容量的分组会直接丢弃。
- 发送端的 `sendto()` 会先进入虚拟队列，再由后台线程按固定速率真正发送到网卡。

运行示例：

```bash
python3 sender.py \
  --target-host 127.0.0.1 \
  --target-port 9001 \
  --window-size 30 \
  --link-queue-capacity 20 \
  --link-service-delay-ms 10
```

如果想临时关闭虚拟链路，可加上：

```bash
python3 sender.py --disable-virtual-link
```

启用 Q-Learning 智能拥塞控制：

```bash
python3 sender.py \
  --target-host 127.0.0.1 \
  --target-port 9001 \
  --packets 120 \
  --window-size 8 \
  --link-queue-capacity 20 \
  --link-service-delay-ms 10 \
  --cc q-learning \
  --min-window 1 \
  --max-window 32 \
  --q-alpha 0.3 \
  --q-gamma 0.85 \
  --q-epsilon 0.1 \
  --q-table q_table.json
```

为了观察拥塞控制效果，可以让接收端模拟丢包和排队延迟：

```bash
python3 receiver.py \
  --host 127.0.0.1 \
  --port 9001 \
  --loss-rate 0.08 \
  --delay-ms 20 \
  --jitter-ms 10 \
  --seed 1
```

发送端日志中的 `[SENDER][CC]` 行会输出当前状态、选择动作、窗口变化、奖励值和 Q 值。

多轮循环训练：

```bash
python3 train_q_learning.py \
  --rounds 5 \
  --packets 120 \
  --q-table q_table.json \
  --q-epsilon 0.3 \
  --epsilon-decay 0.85 \
  --min-epsilon 0.05
```

## 实现要点

1. `protocol.py` 使用 `struct.pack` / `struct.unpack` 完成 Header 与 ACK 的二进制封装。
2. `sender.py` 使用 `unacked` 字典保存所有未确认分组的 Payload、最近发送时间和发送次数。
3. 主线程按照当前 `window_size` 发送新分组，固定窗口模式下不变，Q-Learning 模式下由控制器每个 RTT 周期更新。
4. ACK 线程持续 `recvfrom()`，收到累计 ACK 后删除所有 `seq <= ack_number` 的未确认分组。
5. ACK 线程根据被确认分组保存的 `wire_timestamp` 计算 RTT，并使用 `SRTT = 0.875 * SRTT + 0.125 * RTT` 平滑。
6. Timer 线程周期扫描 `unacked`，若 `now - last_send >= rto`，则重新封包并发送。
7. ACK 线程统计重复累计 ACK，连续 3 次相同 ACK 时立即快速重传 `ack_number + 1`。
8. `receiver.py` 维护 `expected_seq`，按序推进累计确认号，乱序分组暂存在缓存集合中。
9. Q-Learning 状态严格为 6 个：RTT 趋势 `rtt_up/rtt_down/rtt_stable` × 丢包事件 `loss/no_loss`。
10. Q-Learning 动作为 `0(hold)`、`1(cwnd+1)`、`2(cwnd/2)`，每个 RTT 周期结束时决策下一周期窗口。
11. 奖励函数为 `R = reward_alpha * 本轮成功吞吐量 - reward_beta * 平均RTT(ms) - reward_gamma * 丢包数量`。
12. 使用 epsilon-greedy 探索，并用 Bellman 公式实时更新 Q-Table；`--q-table` 可跨多轮保存和加载学习结果。

## 环境要求

只依赖 Python 标准库，无需安装第三方包。
