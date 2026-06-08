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

    # 1. 绘制 CWND 曲线对比图
    plt.figure(figsize=(10, 5))
    plt.plot(cwnds1, label='AIMD/Reno Baseline', linestyle='--', color='blue', alpha=0.8)
    plt.plot(cwnds2, label='Q-Learning', linewidth=2, color='green')
    plt.title('CWND vs Time (Control Cycles)')
    plt.xlabel('RTT Cycles / Decisions')
    plt.ylabel('Congestion Window (CWND)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    cwnd_output = output_dir / "cwnd_comparison.png"
    plt.savefig(cwnd_output, dpi=300)
    print(f"Saved CWND plot to {cwnd_output}")

    # 2. 绘制吞吐量与平均延迟柱状图
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    labels = ['Baseline', 'Q-Learning']
    x = np.arange(len(labels))
    width = 0.35
    
    # 吞吐量柱子（左轴）
    rects1 = ax1.bar(x - width/2, [thr1, thr2], width, label='Throughput (Mbps)', color='steelblue')
    ax1.set_ylabel('Throughput (Mbps)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # RTT 柱子（右轴）
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, [rtt1, rtt2], width, label='Average RTT (ms)', color='darkorange')
    ax2.set_ylabel('Average RTT (ms)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Performance Comparison: Throughput and Average RTT')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 在柱子上标注数值
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    bars_output = output_dir / "throughput_rtt_comparison.png"
    plt.savefig(bars_output, dpi=300)
    print(f"Saved bar chart plot to {bars_output}")

if __name__ == '__main__':
    main()
