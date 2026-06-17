import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_log(filepath):
    cwnds = []
    throughput = 0.0
    srtt = 0.0
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "[SENDER][CC]" in line:
                    match = re.search(r"window=\d+->(\d+)", line)
                    if match:
                        cwnds.append(int(match.group(1)))
                elif "[SENDER][DONE]" in line:
                    t_match = re.search(r"throughput=([\d\.]+)Mbps", line)
                    rtt_match = re.search(r"srtt_ms=([\d\.]+)", line)
                    if t_match:
                        throughput = float(t_match.group(1))
                    if rtt_match:
                        srtt = float(rtt_match.group(1))
    except FileNotFoundError:
        print(f"Warning: {filepath} not found.")
        
    return cwnds, throughput, srtt

def main():
    parser = argparse.ArgumentParser(description="Visualize Performance")
    parser.add_argument("--log1", default="tx1.log", help="AIMD/Reno baseline log file")
    parser.add_argument("--log2", default="tx2.log", help="Q-Learning log file")
    parser.add_argument("--output-dir", default="artifacts/plots", help="Directory for generated figures")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cwnds1, thr1, rtt1 = parse_log(args.log1)
    cwnds2, thr2, rtt2 = parse_log(args.log2)

    labels = ['Baseline', 'Q-Learning']

    # 1. 绘制 CWND 曲线对比图
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(cwnds1, label='AIMD/Reno Baseline', linestyle='--', color='blue', alpha=0.8)
    axis.plot(cwnds2, label='Q-Learning', linewidth=2, color='green')
    axis.set_title('CWND vs Time (Control Cycles)')
    axis.set_xlabel('RTT Cycles / Decisions')
    axis.set_ylabel('Congestion Window (CWND)')
    axis.legend()
    axis.grid(True, linestyle=':', alpha=0.6)
    cwnd_output = output_dir / "cwnd_comparison.png"
    fig.savefig(cwnd_output, dpi=300)
    plt.close(fig)
    print(f"Saved CWND plot to {cwnd_output}")

    # 2. 绘制吞吐量柱状图
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    x = np.arange(len(labels))
    rects = axis.bar(x, [thr1, thr2], width=0.45, label='Throughput (Mbps)', color='steelblue')
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_title('Throughput Comparison')
    axis.set_ylabel('Throughput (Mbps)')
    axis.grid(True, axis='y', linestyle=':', alpha=0.6)
    for rect in rects:
        height = rect.get_height()
        axis.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    throughput_output = output_dir / "throughput_comparison.png"
    fig.savefig(throughput_output, dpi=300)
    plt.close(fig)
    print(f"Saved throughput plot to {throughput_output}")

    # 3. 绘制平均 RTT 柱状图
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    rects = axis.bar(x, [rtt1, rtt2], width=0.45, label='Average RTT (ms)', color='darkorange')
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_title('Average RTT Comparison')
    axis.set_ylabel('Average RTT (ms)')
    axis.grid(True, axis='y', linestyle=':', alpha=0.6)
    for rect in rects:
        height = rect.get_height()
        axis.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    rtt_output = output_dir / "rtt_comparison.png"
    fig.savefig(rtt_output, dpi=300)
    plt.close(fig)
    print(f"Saved RTT plot to {rtt_output}")

if __name__ == '__main__':
    main()
