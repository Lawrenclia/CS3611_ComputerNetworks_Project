# 基于 UDP 的应用层可靠传输底层架构

本项目保留题目五中的可靠传输底层实现部分，只关注三个核心机制：

- Header 封装：在 UDP Payload 中自定义 `Sequence Number + Timestamp + Payload`。
- 定时重传：发送端维护未确认队列，超过 RTO 后自动重传。
- 多线程控制：发送线程、ACK 接收线程、定时器线程协同工作。

## 目录说明

- `protocol.py`：定义数据包和 ACK 的二进制封装/解析。
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

## 实现要点

1. `protocol.py` 使用 `struct.pack` / `struct.unpack` 完成 Header 与 ACK 的二进制封装。
2. `sender.py` 使用 `unacked` 字典保存所有未确认分组的 Payload、最近发送时间和发送次数。
3. 主线程按照固定 `window-size` 发送新分组，避免无限制注入 UDP 数据。
4. ACK 线程持续 `recvfrom()`，收到累计 ACK 后删除所有 `seq <= ack_number` 的未确认分组。
5. Timer 线程周期扫描 `unacked`，若 `now - last_send >= rto`，则重新封包并发送。
6. `receiver.py` 维护 `expected_seq`，按序推进累计确认号，乱序分组暂存在缓存集合中。

## 环境要求

只依赖 Python 标准库，无需安装第三方包。
