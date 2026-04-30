# 题目五报告模板：可靠传输底层架构版

## 1. 题目概述

本项目基于 UDP Socket 实现应用层可靠传输底层架构，重点展示 Header 封装、ACK 确认、RTO 定时重传和多线程控制。

## 2. 系统架构

系统由两个进程组成：

- `sender.py`：发送端，负责构造数据包、维护未确认队列、接收 ACK 和触发超时重传。
- `receiver.py`：接收端，负责解析数据包、维护接收序号并返回累计 ACK。
- `protocol.py`：协议层封装模块，负责数据包和 ACK 的二进制格式转换。

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

接收端维护 `expected_seq`。当收到按序分组时推进 `expected_seq`；当收到乱序分组时暂存；每次收到数据后返回 `expected_seq - 1` 作为累计 ACK。

## 5. 定时重传

发送端启动独立 Timer 线程周期扫描 `unacked`：

```text
if now - last_send_monotonic >= RTO:
    retransmit(packet)
```

重传时重新写入 Timestamp，更新最近发送时间，并增加发送次数。这样即使 UDP 丢包，发送端也能在 RTO 到期后恢复传输。

## 6. 多线程控制

发送端使用三个执行流协同：

1. 主线程：按照固定滑动窗口发送新分组。
2. ACK 线程：持续接收 ACK，更新未确认队列。
3. Timer 线程：定时检查超时分组并重传。

多个线程共享 `unacked`、`acked_packets` 等状态，因此使用 `threading.Lock` 保证互斥访问。

## 7. 运行方式

启动接收端：

```bash
python3 receiver.py --host 127.0.0.1 --port 9001 --initial-seq 0
```

启动发送端：

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001 --packets 40 --window-size 8 --rto 0.2
```

## 8. 总结

本项目保留并实现了题目五中可靠传输最底层的协议骨架：自定义 Header、累计 ACK、未确认队列、RTO 定时重传和多线程控制。该结构可以作为后续扩展文件传输功能的基础。
