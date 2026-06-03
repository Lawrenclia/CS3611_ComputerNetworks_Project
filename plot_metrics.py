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


def plot(metrics: list[dict[str, str]], histories: dict[str, list[dict[str, str]]], output: Path) -> None:
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
    for mode, rows in histories.items():
        times = [float(row["time_s"]) for row in rows]
        cwnds = [float(row["cwnd"]) for row in rows]
        axes[0].plot(times, cwnds, label=mode, linewidth=1.8)
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
    for mode, rows in histories.items():
        times = [float(row["time_s"]) for row in rows]
        cwnds = [float(row["cwnd"]) for row in rows]
        axis.plot(times, cwnds, label=mode, linewidth=1.8)
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
    parser.add_argument("--metrics-file", default="metrics.csv")
    parser.add_argument("--history-file", default="history.csv")
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()

    metrics = read_latest_metrics(Path(args.metrics_file))
    histories = read_latest_history(Path(args.history_file))
    plot(metrics, histories, Path(args.output))


if __name__ == "__main__":
    main()
