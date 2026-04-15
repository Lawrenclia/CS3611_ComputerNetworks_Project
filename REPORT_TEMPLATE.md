# 题目五报告模板

## 1. 题目概述

本项目实现了一个基于 UDP 的应用层可靠传输协议。系统在发送端实现了数据编号、RTT 采样、超时重传、拥塞窗口控制与虚拟瓶颈链路模拟，并对比了传统 AIMD 与 Q-Learning 两种拥塞控制策略。

## 2. 系统架构

### 2.1 进程划分

- `receiver.py`：负责接收 UDP 数据报、缓存乱序分组，并返回累计 ACK。
- `sender.py`：负责可靠发送、拥塞控制、数据统计与结果导出。
- `virtual_link.py`：模拟固定带宽和有限队列的瓶颈链路。

### 2.2 通信流程

1. Sender 生成带序号的数据包并交给 `VirtualLink`。
2. `VirtualLink` 按固定速率漏出分组，队列满时直接丢包。
3. Receiver 收到数据包后更新累计确认号并发送 ACK。
4. Sender 收到 ACK 后更新 RTT/SRTT、滑动窗口和拥塞控制状态。
5. 若超时或出现 3 个重复 ACK，则触发重传。

## 3. 协议设计

### 3.1 数据包格式

- Data Packet: `Sequence Number (4B) + Timestamp (8B) + Payload (1024B)`
- ACK Packet: `ACK Number (4B)`

### 3.2 可靠传输机制

- 发送端维护未确认队列。
- Receiver 使用累计 ACK。
- Sender 支持两种重传机制：
  - RTO 超时重传
  - 3 个重复 ACK 触发的快速重传

## 4. 拥塞控制设计

### 4.1 AIMD

- 初始 `CWND = 1`
- 每收到一个新 ACK：`CWND += 1 / CWND`
- 出现丢包：`CWND = max(1, CWND / 2)`

### 4.2 Q-Learning

状态空间：

- RTT 趋势：减小 / 平稳 / 增大
- 丢包事件：未发生 / 发生
- 共 6 个离散状态

动作空间：

- `0`：保持 CWND
- `1`：`CWND + 1`
- `2`：`CWND / 2`

奖励函数：

```text
Reward = 8 * throughput_mbps - 0.015 * avg_rtt_ms - 1.4 * loss_count
```

## 5. 虚拟瓶颈链路设计

- 固定发送速率：默认 `100 packets/s`
- 队列大小：默认 `20`
- 当发送端瞬时注入数据超过队列容量时：
  - 已入队数据产生排队时延
  - 超过队列容量的分组被直接丢弃

## 6. 实验结果

建议展示以下图表与指标：

- CWND 随时间变化曲线
- AIMD 与 Q-Learning 的平均吞吐量对比
- AIMD 与 Q-Learning 的平均 RTT 对比
- 重传次数、快速重传次数、链路丢包次数

可直接引用 `results/` 或 `smoke_results/` 中的：

- `comparison.png` / `comparison.svg`
- `metrics.csv`
- `summary.json`

## 7. 结果分析

可从以下角度分析：

- AIMD 是否呈现典型锯齿形窗口变化
- Q-Learning 是否能够在低 RTT 和较高吞吐之间取得平衡
- 虚拟瓶颈链路对 RTT 和丢包的影响
- 快速重传相对于纯超时重传的优势

## 8. 不足与改进

- 当前 Q-Table 状态空间仍较粗糙
- 未引入更精细的乱序恢复策略
- 可进一步扩展为 DQN/PPO
- 可加入动态带宽突变场景测试
