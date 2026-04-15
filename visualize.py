import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def save_cwnd_csv(path: Path, label: str, samples: list[tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "time_s", "cwnd"])
        for time_s, cwnd in samples:
            writer.writerow([label, f"{time_s:.6f}", f"{cwnd:.6f}"])


def save_metrics_csv(path: Path, metrics: list[dict]) -> None:
    fieldnames = [
        "algorithm",
        "duration_s",
        "throughput_mbps",
        "avg_rtt_ms",
        "retransmissions",
        "fast_retransmissions",
        "timeouts",
        "link_drops",
        "srtt_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def save_comparison_svg(
    path: Path,
    aimd_samples: list[tuple[float, float]],
    q_samples: list[tuple[float, float]],
    aimd_metrics: dict,
    q_metrics: dict,
) -> None:
    width = 1100
    height = 720
    margin_left = 80
    margin_right = 40
    margin_top = 50
    margin_bottom = 80
    chart_width = width - margin_left - margin_right
    chart_height = 340
    bar_top = 470
    bar_height = 150

    max_time = max(
        aimd_samples[-1][0] if aimd_samples else 1.0,
        q_samples[-1][0] if q_samples else 1.0,
        1.0,
    )
    max_cwnd = max(
        max((sample[1] for sample in aimd_samples), default=1.0),
        max((sample[1] for sample in q_samples), default=1.0),
        1.0,
    )
    max_metric = max(
        aimd_metrics["throughput_mbps"],
        q_metrics["throughput_mbps"],
        aimd_metrics["avg_rtt_ms"],
        q_metrics["avg_rtt_ms"],
        1.0,
    )

    def line_points(samples: list[tuple[float, float]]) -> str:
        if not samples:
            return ""
        points = []
        for time_s, cwnd in samples:
            x = margin_left + (time_s / max_time) * chart_width
            y = margin_top + chart_height - (cwnd / max_cwnd) * chart_height
            points.append(f"{x:.2f},{y:.2f}")
        return " ".join(points)

    def bar_rect(x: float, value: float, color: str) -> str:
        h = (value / max_metric) * bar_height
        y = bar_top + bar_height - h
        return (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="80" height="{h:.1f}" '
            f'fill="{color}" rx="6" />'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px Arial, sans-serif; fill: #1f2937; }}
    .subtitle {{ font: 600 16px Arial, sans-serif; fill: #374151; }}
    .axis {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
    .legend {{ font: 13px Arial, sans-serif; fill: #111827; }}
    .metric {{ font: 12px Arial, sans-serif; fill: #111827; }}
    .bg {{ fill: #f8fafc; }}
  </style>
  <rect class="bg" width="{width}" height="{height}" rx="18" />
  <text class="title" x="40" y="36">UDP reliable transport: AIMD vs Q-Learning</text>
  <text class="subtitle" x="40" y="74">CWND over time</text>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#94a3b8" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#94a3b8" stroke-width="1.5" />
  <polyline fill="none" stroke="#2563eb" stroke-width="3" points="{line_points(aimd_samples)}" />
  <polyline fill="none" stroke="#dc2626" stroke-width="3" points="{line_points(q_samples)}" />
  <rect x="{margin_left + 10}" y="{margin_top + 10}" width="16" height="4" fill="#2563eb" />
  <text class="legend" x="{margin_left + 34}" y="{margin_top + 16}">AIMD</text>
  <rect x="{margin_left + 100}" y="{margin_top + 10}" width="16" height="4" fill="#dc2626" />
  <text class="legend" x="{margin_left + 124}" y="{margin_top + 16}">Q-Learning</text>
  <text class="axis" x="{margin_left - 18}" y="{margin_top + chart_height + 20}">0</text>
  <text class="axis" x="{margin_left + chart_width - 18}" y="{margin_top + chart_height + 20}">{max_time:.2f}s</text>
  <text class="axis" x="{margin_left - 32}" y="{margin_top + 6}">{max_cwnd:.1f}</text>
  <text class="axis" x="{margin_left - 24}" y="{margin_top + chart_height}">0</text>

  <text class="subtitle" x="40" y="448">Average throughput and RTT</text>
  <line x1="{margin_left}" y1="{bar_top}" x2="{margin_left}" y2="{bar_top + bar_height}" stroke="#94a3b8" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{bar_top + bar_height}" x2="{margin_left + chart_width}" y2="{bar_top + bar_height}" stroke="#94a3b8" stroke-width="1.5" />
  {bar_rect(170, aimd_metrics["throughput_mbps"], "#2563eb")}
  {bar_rect(270, q_metrics["throughput_mbps"], "#dc2626")}
  {bar_rect(520, aimd_metrics["avg_rtt_ms"], "#2563eb")}
  {bar_rect(620, q_metrics["avg_rtt_ms"], "#dc2626")}
  <text class="axis" x="170" y="{bar_top + bar_height + 20}">AIMD Mbps</text>
  <text class="axis" x="270" y="{bar_top + bar_height + 20}">Q Mbps</text>
  <text class="axis" x="520" y="{bar_top + bar_height + 20}">AIMD RTT</text>
  <text class="axis" x="620" y="{bar_top + bar_height + 20}">Q RTT</text>
  <text class="metric" x="170" y="{bar_top - 10}">{aimd_metrics["throughput_mbps"]:.3f} Mbps</text>
  <text class="metric" x="270" y="{bar_top - 10}">{q_metrics["throughput_mbps"]:.3f} Mbps</text>
  <text class="metric" x="520" y="{bar_top - 10}">{aimd_metrics["avg_rtt_ms"]:.2f} ms</text>
  <text class="metric" x="620" y="{bar_top - 10}">{q_metrics["avg_rtt_ms"]:.2f} ms</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_comparison_plot(
    path: Path,
    aimd_samples: list[tuple[float, float]],
    q_samples: list[tuple[float, float]],
    aimd_metrics: dict,
    q_metrics: dict,
) -> bool:
    if plt is None:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].plot(
        [time_s for time_s, _ in aimd_samples],
        [cwnd for _, cwnd in aimd_samples],
        label="AIMD",
        linewidth=2.2,
    )
    axes[0].plot(
        [time_s for time_s, _ in q_samples],
        [cwnd for _, cwnd in q_samples],
        label="Q-Learning",
        linewidth=2.2,
    )
    axes[0].set_title("CWND over time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("CWND")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    categories = ["Throughput (Mbps)", "Average RTT (ms)"]
    x = [0, 1]
    width = 0.35
    axes[1].bar(
        [value - width / 2 for value in x],
        [aimd_metrics["throughput_mbps"], aimd_metrics["avg_rtt_ms"]],
        width=width,
        label="AIMD",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        [q_metrics["throughput_mbps"], q_metrics["avg_rtt_ms"]],
        width=width,
        label="Q-Learning",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_title("Throughput and RTT comparison")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True
