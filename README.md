# 基于 UDP 的应用层可靠传输底层架构

本项目保留题目五中的可靠传输底层实现部分，只关注三个核心机制：

- Header 封装：在 UDP Payload 中自定义 `Sequence Number + Timestamp + Payload`。
- 定时重传：发送端维护未确认队列，超过 RTO 后自动重传。
- RTT/SRTT 采样：Sender 收到 ACK 后用当前时间减去包头 Timestamp 得到 RTT，并平滑计算 SRTT。
- 多线程控制：发送线程、ACK 接收线程、定时器线程协同工作。
- 快速重传与乱序处理：连续 3 个重复 ACK 立即重传缺失分组，接收端缓存乱序到达的数据包。

## 目录说明

- `protocol.py`：定义数据包和 ACK 的二进制封装/解析。
- `virtual_link.py`：模拟固定带宽和有限缓存的虚拟瓶颈链路。
- `sender.py`：可靠发送端，负责滑动窗口发送、ACK 处理和 RTO 重传。
- `receiver.py`：接收端，负责解析数据包、维护累计 ACK 并返回确认。
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

## 实现要点

1. `protocol.py` 使用 `struct.pack` / `struct.unpack` 完成 Header 与 ACK 的二进制封装。
2. `sender.py` 使用 `unacked` 字典保存所有未确认分组的 Payload、最近发送时间和发送次数。
3. 主线程按照固定 `window-size` 发送新分组，避免无限制注入 UDP 数据。
4. ACK 线程持续 `recvfrom()`，收到累计 ACK 后删除所有 `seq <= ack_number` 的未确认分组。
5. ACK 线程根据被确认分组保存的 `wire_timestamp` 计算 RTT，并使用 `SRTT = 0.875 * SRTT + 0.125 * RTT` 平滑。
6. Timer 线程周期扫描 `unacked`，若 `now - last_send >= rto`，则重新封包并发送。
7. ACK 线程统计重复累计 ACK，连续 3 次相同 ACK 时立即快速重传 `ack_number + 1`。
8. `receiver.py` 维护 `expected_seq`，按序推进累计确认号，乱序分组暂存在缓存集合中。

## 环境要求

只依赖 Python 标准库，无需安装第三方包。
