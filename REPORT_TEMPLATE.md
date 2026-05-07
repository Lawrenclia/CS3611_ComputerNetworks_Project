# 题目五报告模板：可靠传输与 Q-Learning 拥塞控制版

## 1. 题目概述

本项目基于 UDP Socket 实现应用层可靠传输底层架构，并加入 Q-Learning 智能拥塞控制，重点展示 Header 封装、ACK 确认、RTO 定时重传、RTT/SRTT 采样、多线程控制、快速重传、乱序处理和动态窗口调整。

## 2. 系统架构

系统由两个进程组成：

- `sender.py`：发送端，负责构造数据包、维护未确认队列、接收 ACK 和触发超时重传。
- `receiver.py`：接收端，负责解析数据包、维护接收序号并返回累计 ACK。
- `protocol.py`：协议层封装模块，负责数据包和 ACK 的二进制格式转换。
- `congestion_control.py`：拥塞控制模块，负责固定窗口控制和 Q-Learning 动态窗口控制。

## 3. 协议设计

### 3.1 数据包 Header

```text
Data Packet = Sequence Number (4B) + Timestamp (8B) + Payload (1024B)
ACK Packet  = ACK Number (4B, signed)
```

### 3.2 Header 封装

`protocol.py` 使用网络字节序封装 Header：

- `pack_data_packet()`：将序号、时间戳和负载打包为 UDP Payload。
- `unpack_data_packet()`：从 UDP Payload 中解析 Header 与数据。
- `pack_ack()` / `unpack_ack()`：封装和解析累计 ACK。

## 4. 可靠传输机制

发送端维护 `unacked` 字典，记录尚未被 ACK 确认的分组：

- Payload
- 最近一次发送的单调时钟时间
- 写入 Header 的发送时间戳
- 当前分组发送次数

接收端维护 `expected_seq`。当收到按序分组时推进 `expected_seq`；当收到乱序分组时暂存；每次收到数据后返回 `expected_seq - 1` 作为累计 ACK。发送端连续收到 3 个重复 ACK 时，立即快速重传 `ack_number + 1` 对应分组。

## 5. 定时重传

发送端启动独立 Timer 线程周期扫描 `unacked`：

```text
if now - last_send_monotonic >= RTO:
    retransmit(packet)
```

重传时重新写入 Timestamp，更新最近发送时间，并增加发送次数。这样即使 UDP 丢包，发送端也能在 RTO 到期后恢复传输。

## 6. 多线程控制

发送端使用三个执行流协同：

1. 主线程：按照当前拥塞窗口发送新分组。
2. ACK 线程：持续接收 ACK，更新未确认队列。
3. Timer 线程：定时检查超时分组并重传。

多个线程共享 `unacked`、`acked_packets` 等状态，因此使用 `threading.Lock` 保证互斥访问。

## 7. Q-Learning 拥塞控制

Q-Learning 控制器按 1 个 RTT 为周期统计网络特征。状态空间为 RTT 趋势和丢包事件的组合：RTT 趋势包括 `rtt_up`、`rtt_down`、`rtt_stable`，丢包事件包括 `loss`、`no_loss`，共 6 个状态。动作空间为 `0(CWND保持)`、`1(CWND+1)`、`2(CWND/2)`。

奖励函数：

```text
R = reward_alpha * throughput - reward_beta * avg_rtt_ms - reward_gamma * loss_count
```

控制器使用 epsilon-greedy 探索，并在每个 RTT 周期结束时使用 Bellman 公式更新 Q-Table：

核心更新公式：

```text
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max Q(s', a') - Q(s, a))
```

## 8. 运行方式

启动接收端：

```bash
python3 receiver.py --host 127.0.0.1 --port 9001 --initial-seq 0
```

启动发送端：

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001 --packets 40 --window-size 8 --rto 0.2
```

启用 Q-Learning：

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001 --packets 120 --window-size 8 --cc q-learning --min-window 1 --max-window 32 --q-table q_table.json
```

多轮训练：

```bash
python3 train_q_learning.py --rounds 5 --packets 120 --q-table q_table.json --q-epsilon 0.3 --epsilon-decay 0.85
```

## 9. 总结

本项目实现了题目五中可靠传输协议骨架：自定义 Header、累计 ACK、未确认队列、RTO 定时重传、多线程控制和快速重传。同时，Q-Learning 控制器严格按 6 状态、3 动作、复合奖励、epsilon-greedy 和 Bellman 更新实现，可作为智能拥塞控制设计的核心展示内容。
