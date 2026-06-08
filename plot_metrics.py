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
            raise SystemExit("matplotlib is required for comparison plotting") from exc

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
