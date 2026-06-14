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


def normalized_mode(mode: str) -> str:
    return mode.lower().replace("-", "").replace("_", "").replace(" ", "")


def is_qlearning(mode: str) -> bool:
    return "qlearning" in normalized_mode(mode)


def is_dqn(mode: str) -> bool:
    return "dqn" in normalized_mode(mode)


def display_cwnd(mode: str, value: float) -> float:
    if is_qlearning(mode):
        return float(max(1, int(value)))
    return value


def should_smooth_cwnd(mode: str, smooth_window: int, points: int) -> bool:
    return smooth_window > 1 and points >= 3 and is_qlearning(mode)


def plot_cwnd_histories(axis, histories: dict[str, list[dict[str, str]]], smooth_window: int) -> None:
    for mode, rows in histories.items():
        times = [float(row["time_s"]) for row in rows]
        cwnds = [display_cwnd(mode, float(row["cwnd"])) for row in rows]
        if is_qlearning(mode):
            raw_label = f"{mode} raw (step)"
            trend_label = f"{mode} trend (moving average)"
            axis.step(
                times, cwnds, where="post",
                label=raw_label, linewidth=1.4, color="#d97706",
            )
            if should_smooth_cwnd(mode, smooth_window, len(cwnds)):
                axis.plot(
                    times, moving_average(cwnds, smooth_window),
                    label=trend_label, linewidth=1.8,
                    linestyle="--", color="#92400e",
                )
        elif is_dqn(mode):
            axis.plot(times, cwnds, label=f"{mode} raw (continuous)", linewidth=1.4)
        else:
            axis.plot(times, cwnds, label=mode, linewidth=1.4)


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
    retx = [float(row["retransmissions"]) for row in metrics]
    timeouts = [float(row["timeout_events"]) for row in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    # --- CWND subplot ---
    plot_cwnd_histories(axes[0, 0], histories, smooth_window)
    axes[0, 0].set_title("CWND over time (single run)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("CWND (packets)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      borderaxespad=0, fontsize="small")

    # --- RTT subplot ---
    rtt_axis = axes[0, 1]
    x = range(len(modes))
    bars = rtt_axis.bar(x, avg_rtt, width=0.5, color="tab:orange", alpha=0.85)
    rtt_axis.set_title("Average RTT (single run)")
    rtt_axis.set_xticks(list(x), modes)
    rtt_axis.set_ylabel("Average RTT (ms)")
    rtt_axis.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, avg_rtt):
        rtt_axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f"{val:.1f}", ha="center", va="bottom", fontsize="small")

    # --- Throughput subplot ---
    tp_axis = axes[1, 0]
    bars = tp_axis.bar(x, throughput, width=0.5, color="tab:blue", alpha=0.85)
    tp_axis.set_title("Throughput (single run)")
    tp_axis.set_xticks(list(x), modes)
    tp_axis.set_ylabel("Throughput (Mbps)")
    tp_axis.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, throughput):
        tp_axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize="small")

    # --- Retransmissions + Timeouts subplot ---
    re_axis = axes[1, 1]
    bw = 0.20
    re_axis.bar([i - bw for i in x], retx, width=bw * 2, label="Retransmissions",
                color="tab:red", alpha=0.85)
    re_axis.bar([i + bw for i in x], timeouts, width=bw * 2, label="Timeouts",
                color="tab:purple", alpha=0.85)
    re_axis.set_title("Retransmissions & timeouts (single run)")
    re_axis.set_xticks(list(x), modes)
    re_axis.set_ylabel("Count")
    re_axis.grid(True, axis="y", alpha=0.3)
    re_axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   borderaxespad=0, fontsize="small")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"saved {output}")

    # --- CWND-only export ---
    cwnd_output = output.with_name(f"{output.stem}_cwnd{output.suffix}")
    fig2, axis2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    plot_cwnd_histories(axis2, histories, smooth_window)
    axis2.set_title("CWND over time (single run)")
    axis2.set_xlabel("Time (s)")
    axis2.set_ylabel("CWND (packets)")
    axis2.grid(True, alpha=0.3)
    axis2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                 borderaxespad=0, fontsize="small")
    fig2.savefig(cwnd_output, dpi=140)
    plt.close(fig2)
    print(f"saved {cwnd_output}")


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
        "qlearning": "#d97706",
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
        draw_axes(draw, box, "CWND over time (single run)", "Time (s)", "CWND")
        left, top, right, bottom = box
        all_times = [float(row["time_s"]) for rows in histories.values() for row in rows]
        all_cwnds = [
            display_cwnd(mode, float(row["cwnd"]))
            for mode, rows in histories.items()
            for row in rows
        ]
        max_time = max(all_times, default=1.0)
        min_cwnd = min(all_cwnds, default=0.0)
        max_cwnd = max(all_cwnds, default=1.0)
        span = max(max_cwnd - min_cwnd, 1.0)
        legend_x = right - 180
        legend_y = top + 10
        for index, (mode, rows) in enumerate(histories.items()):
            times = [float(row["time_s"]) for row in rows]
            cwnds = [display_cwnd(mode, float(row["cwnd"])) for row in rows]
            if should_smooth_cwnd(mode, smooth_window, len(cwnds)):
                cwnds = moving_average(cwnds, smooth_window)
            points = [
                (
                    left + int((value / max_time) * (right - left)),
                    bottom - int(((cwnd - min_cwnd) / span) * (bottom - top)),
                )
                for value, cwnd in zip(times, cwnds)
            ]
            color = colors.get(normalized_mode(mode), "#64748b")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AIMD/Q-Learning/DQN comparison from sender CSV logs")
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
