from __future__ import annotations

import argparse
import csv
import site
import sys
from collections import OrderedDict
from pathlib import Path


def read_latest_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"metrics file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    latest: OrderedDict[str, dict[str, str]] = OrderedDict()
    for row in rows:
        latest[row["mode"]] = row
    return list(latest.values())


def read_latest_history(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    latest_run_by_mode: OrderedDict[str, str] = OrderedDict()
    for row in rows:
        latest_run_by_mode[row["mode"]] = row["run_id"]

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if latest_run_by_mode.get(row["mode"]) == row["run_id"]:
            grouped.setdefault(row["mode"], []).append(row)
    return grouped


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]
    half_window = window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def should_smooth_cwnd(mode: str, smooth_window: int, points: int) -> bool:
    normalized = mode.lower().replace("-", "").replace("_", "").replace(" ", "")
    return smooth_window > 1 and points >= 3 and "qlearning" in normalized


def plot_cwnd_histories(axis, histories: dict[str, list[dict[str, str]]], smooth_window: int) -> None:
    for mode, rows in histories.items():
        times = [float(row["time_s"]) for row in rows]
        cwnds = [float(row["cwnd"]) for row in rows]
        if should_smooth_cwnd(mode, smooth_window, len(cwnds)):
            line = axis.plot(times, cwnds, label=f"{mode} raw", linewidth=1.1, alpha=0.35)[0]
            axis.plot(
                times,
                moving_average(cwnds, smooth_window),
                label=f"{mode} moving avg",
                linewidth=2.2,
                color=line.get_color(),
            )
        else:
            axis.plot(times, cwnds, label=mode, linewidth=1.8)


def plot(
    metrics: list[dict[str, str]],
    histories: dict[str, list[dict[str, str]]],
    output: Path,
    smooth_window: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.append(user_site)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plot_with_pillow(metrics, histories, output, smooth_window)
            return

    modes = [row["mode"] for row in metrics]
    throughput = [float(row["throughput_mbps"]) for row in metrics]
    avg_rtt = [float(row["avg_rtt_ms"]) for row in metrics]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    plot_cwnd_histories(axes[0], histories, smooth_window)
    axes[0].set_title("CWND over time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("CWND (packets)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    x = range(len(modes))
    throughput_axis = axes[1]
    rtt_axis = throughput_axis.twinx()
    throughput_axis.bar(
        [i - 0.18 for i in x],
        throughput,
        width=0.36,
        label="Throughput (Mbps)",
        color="tab:blue",
    )
    rtt_axis.bar(
        [i + 0.18 for i in x],
        avg_rtt,
        width=0.36,
        label="Average RTT (ms)",
        color="tab:orange",
        alpha=0.72,
    )
    throughput_axis.set_title("Throughput and average RTT")
    throughput_axis.set_xticks(list(x), modes)
    throughput_axis.set_ylabel("Throughput (Mbps)")
    rtt_axis.set_ylabel("Average RTT (ms)")
    throughput_axis.grid(True, axis="y", alpha=0.3)

    throughput_handles, throughput_labels = throughput_axis.get_legend_handles_labels()
    rtt_handles, rtt_labels = rtt_axis.get_legend_handles_labels()
    throughput_axis.legend(
        throughput_handles + rtt_handles,
        throughput_labels + rtt_labels,
        loc="upper right",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"saved {output}")

    cwnd_output = output.with_name(f"{output.stem}_cwnd{output.suffix}")
    fig, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    plot_cwnd_histories(axis, histories, smooth_window)
    axis.set_title("CWND over time")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("CWND (packets)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(cwnd_output, dpi=140)
    plt.close(fig)
    print(f"saved {cwnd_output}")

    bars_output = output.with_name(f"{output.stem}_throughput_rtt{output.suffix}")
    fig, throughput_axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    rtt_axis = throughput_axis.twinx()
    throughput_axis.bar(
        [i - 0.18 for i in x],
        throughput,
        width=0.36,
        label="Throughput (Mbps)",
        color="tab:blue",
    )
    rtt_axis.bar(
        [i + 0.18 for i in x],
        avg_rtt,
        width=0.36,
        label="Average RTT (ms)",
        color="tab:orange",
        alpha=0.72,
    )
    throughput_axis.set_title("Throughput and average RTT")
    throughput_axis.set_xticks(list(x), modes)
    throughput_axis.set_ylabel("Throughput (Mbps)")
    rtt_axis.set_ylabel("Average RTT (ms)")
    throughput_axis.grid(True, axis="y", alpha=0.3)
    throughput_handles, throughput_labels = throughput_axis.get_legend_handles_labels()
    rtt_handles, rtt_labels = rtt_axis.get_legend_handles_labels()
    throughput_axis.legend(
        throughput_handles + rtt_handles,
        throughput_labels + rtt_labels,
        loc="upper right",
    )
    fig.savefig(bars_output, dpi=140)
    plt.close(fig)
    print(f"saved {bars_output}")


def plot_with_pillow(
    metrics: list[dict[str, str]],
    histories: dict[str, list[dict[str, str]]],
    output: Path,
    smooth_window: int,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("matplotlib or Pillow is required for comparison plotting") from exc

    colors = {
        "aimd": "#2563eb",
        "qlearning": "#f59e0b",
        "dqn": "#16a34a",
        "fixed": "#7c3aed",
    }
    font = ImageFont.load_default()

    def draw_axes(draw, box, title, x_label, y_label):
        left, top, right, bottom = box
        draw.rectangle(box, outline="#94a3b8", width=1)
        draw.text((left, top - 22), title, fill="#0f172a", font=font)
        draw.text(((left + right) // 2 - 25, bottom + 10), x_label, fill="#334155", font=font)
        draw.text((8, (top + bottom) // 2), y_label, fill="#334155", font=font)

    def draw_cwnd(image, box):
        draw = ImageDraw.Draw(image)
        draw_axes(draw, box, "CWND over time", "Time (s)", "CWND")
        left, top, right, bottom = box
        all_times = [float(row["time_s"]) for rows in histories.values() for row in rows]
        all_cwnds = [float(row["cwnd"]) for rows in histories.values() for row in rows]
        max_time = max(all_times, default=1.0)
        min_cwnd = min(all_cwnds, default=0.0)
        max_cwnd = max(all_cwnds, default=1.0)
        span = max(max_cwnd - min_cwnd, 1.0)
        legend_x = right - 110
        legend_y = top + 10
        for index, (mode, rows) in enumerate(histories.items()):
            times = [float(row["time_s"]) for row in rows]
            cwnds = [float(row["cwnd"]) for row in rows]
            if should_smooth_cwnd(mode, smooth_window, len(cwnds)):
                cwnds = moving_average(cwnds, smooth_window)
            points = [
                (
                    left + int((value / max_time) * (right - left)),
                    bottom - int(((cwnd - min_cwnd) / span) * (bottom - top)),
                )
                for value, cwnd in zip(times, cwnds)
            ]
            color = colors.get(mode.lower().replace("-", ""), "#64748b")
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            draw.line((legend_x, legend_y + index * 18 + 6, legend_x + 22, legend_y + index * 18 + 6), fill=color, width=3)
            draw.text((legend_x + 28, legend_y + index * 18), mode, fill="#0f172a", font=font)

    def draw_bars(image, box):
        draw = ImageDraw.Draw(image)
        draw_axes(draw, box, "Throughput and average RTT", "Controller", "Value")
        left, top, right, bottom = box
        count = max(len(metrics), 1)
        group_width = (right - left) / count
        max_throughput = max((float(row["throughput_mbps"]) for row in metrics), default=1.0)
        max_rtt = max((float(row["avg_rtt_ms"]) for row in metrics), default=1.0)
        for index, row in enumerate(metrics):
            center = left + group_width * (index + 0.5)
            throughput = float(row["throughput_mbps"])
            rtt = float(row["avg_rtt_ms"])
            bar_width = max(12, int(group_width * 0.20))
            throughput_height = int((throughput / max_throughput) * (bottom - top - 25))
            rtt_height = int((rtt / max_rtt) * (bottom - top - 25))
            draw.rectangle(
                (int(center - bar_width - 2), bottom - throughput_height, int(center - 2), bottom),
                fill="#2563eb",
            )
            draw.rectangle(
                (int(center + 2), bottom - rtt_height, int(center + bar_width + 2), bottom),
                fill="#f59e0b",
            )
            draw.text((int(center - 25), bottom + 8), row["mode"], fill="#0f172a", font=font)
            draw.text((int(center - bar_width - 4), bottom - throughput_height - 14), f"{throughput:.3f}", fill="#1d4ed8", font=font)
            draw.text((int(center + 2), bottom - rtt_height - 14), f"{rtt:.1f}", fill="#b45309", font=font)
        draw.rectangle((right - 180, top + 8, right - 168, top + 20), fill="#2563eb")
        draw.text((right - 162, top + 7), "Throughput Mbps", fill="#0f172a", font=font)
        draw.rectangle((right - 180, top + 26, right - 168, top + 38), fill="#f59e0b")
        draw.text((right - 162, top + 25), "Average RTT ms", fill="#0f172a", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1540, 1120), "white")
    draw_cwnd(image, (90, 70, 1480, 520))
    draw_bars(image, (90, 630, 1480, 1030))
    image.save(output)
    print(f"saved {output} (Pillow fallback)")

    cwnd_output = output.with_name(f"{output.stem}_cwnd{output.suffix}")
    cwnd_image = Image.new("RGB", (1540, 650), "white")
    draw_cwnd(cwnd_image, (90, 70, 1480, 560))
    cwnd_image.save(cwnd_output)
    print(f"saved {cwnd_output} (Pillow fallback)")

    bars_output = output.with_name(f"{output.stem}_throughput_rtt{output.suffix}")
    bars_image = Image.new("RGB", (1260, 650), "white")
    draw_bars(bars_image, (90, 70, 1200, 560))
    bars_image.save(bars_output)
    print(f"saved {bars_output} (Pillow fallback)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AIMD/Q-Learning comparison from sender CSV logs")
    parser.add_argument("--metrics-file", default="artifacts/metrics/metrics.csv")
    parser.add_argument("--history-file", default="artifacts/metrics/history.csv")
    parser.add_argument("--output", default="artifacts/plots/comparison.png")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="moving-average window for Q-Learning CWND curves; use 1 to disable",
    )
    args = parser.parse_args()

    metrics = read_latest_metrics(Path(args.metrics_file))
    histories = read_latest_history(Path(args.history_file))
    plot(metrics, histories, Path(args.output), max(1, args.smooth_window))


if __name__ == "__main__":
    main()
