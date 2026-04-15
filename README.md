# 基于 UDP 的应用层可靠传输与 AI 拥塞控制协议实现

本项目对应计算机网络课程大作业题目五，目标是在 Python 应用层基于原生 UDP Socket，从零实现一个带有可靠传输和拥塞控制能力的简化协议系统，并对比传统 AIMD 与 Q-Learning 两种控制策略。

## 项目内容

- 应用层可靠传输：自定义数据包格式、ACK 确认、RTO 超时重传。
- RTT 与 SRTT 采样：基于包头时间戳统计往返时延。
- 虚拟瓶颈链路：固定带宽、有限队列、排队时延、缓存溢出丢包。
- AIMD 基线算法：模拟类似 TCP Reno 的拥塞避免逻辑。
- Q-Learning 智能控制器：基于 RTT 趋势和丢包事件动态调整 `CWND`。
- 扩展功能：3 个重复 ACK 触发快速重传，接收端支持乱序缓存与累计 ACK。
- 结果导出：生成 `CSV`、`SVG`，安装 `matplotlib` 后还可生成 `PNG` 对比图。

## 目录说明

- `sender.py`：发送端主程序，负责发送、收 ACK、重传、拥塞控制和实验统计。
- `receiver.py`：接收端主程序，负责收包、累计 ACK 和乱序缓存。
- `virtual_link.py`：虚拟瓶颈链路，模拟排队和丢包。
- `congestion.py`：AIMD 与 Q-Learning 控制器实现。
- `protocol.py`：数据包与 ACK 格式定义。
- `visualize.py`：导出对比图、指标表和 `Q-Table` 结果。
- `题目五实验报告.md`：实验报告成稿。
- `答辩讲稿提纲.md`：答辩讲稿提纲。

## 环境要求

- Python 3
- 建议安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 运行方式

先在一个终端启动接收端：

```bash
python3 receiver.py --host 127.0.0.1 --port 9001 --initial-seq 0
```

再在另一个终端启动发送端：

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001
```

## 常用测试命令

基础功能验证：

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001 \
  --packets 20 --train-episodes 1 --results-dir test_basic_20260415 --quiet
```

轻量训练测试：

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001 \
  --packets 40 --train-episodes 2 --results-dir test_train_20260415
```

拥塞压力与快速重传测试：

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001 \
  --packets 60 --train-episodes 0 --bandwidth-pps 20 --queue-size 4 \
  --rto 0.15 --results-dir test_fr_20260415
```

## 输出结果

每次实验都会在对应结果目录下生成：

- `summary.json`：实验汇总结果
- `metrics.csv`：吞吐量、RTT、重传等指标
- `aimd_cwnd.csv`：AIMD 的拥塞窗口变化
- `q_learning_cwnd.csv`：Q-Learning 的拥塞窗口变化
- `q_table.json`：Q-Learning 学习后的 Q 表
- `comparison.svg`：对比图
- `comparison.png`：安装 `matplotlib` 后额外生成

当前仓库中已经保留了以下测试结果目录：

- `test_basic_20260415/`
- `test_train_20260415/`
- `test_fr_20260415/`
- `verify_results/`
- `smoke_results/`
- `fr_results/`

## 说明

- 发送端运行期间，接收端需要保持开启。
- 接收端使用累计 ACK，因此单次实验中序号应连续。
- 题面中写作 `Sender.py`、`Receiver.py`；在当前 macOS 环境中，实际使用 `sender.py`、`receiver.py` 即可。
