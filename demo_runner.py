from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Scenario:
    name: str
    mode: str
    packets: int
    window_size: int
    extra_args: list[str]
    metrics_file: Path
    history_file: Path
    sender_log: Path
    receiver_log: Path
    receiver_port: int
    sender_port: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-click verifier for the UDP congestion-control project")
    parser.add_argument("--packets", type=int, default=300)
    parser.add_argument("--base-port", type=int, default=19000)
    parser.add_argument("--loss-rate", type=float, default=0.02)
    parser.add_argument("--delay-ms", type=float, default=10.0)
    parser.add_argument("--jitter-ms", type=float, default=3.0)
    parser.add_argument("--rto", type=float, default=0.20)
    parser.add_argument("--max-cwnd", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--q-table", default="artifacts/models/active/q_table.json")
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--include-dqn", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def run_command(command: list[str], cwd: Path, stdout_path: Path | None = None) -> subprocess.CompletedProcess:
    if stdout_path is None:
        return subprocess.run(command, cwd=cwd)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout:
        return subprocess.run(command, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT)


def start_receiver(root: Path, scenario: Scenario, args: argparse.Namespace) -> subprocess.Popen:
    command = [
        sys.executable,
        str(root / "receiver.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(scenario.receiver_port),
        "--initial-seq",
        "0",
        "--loss-rate",
        str(args.loss_rate),
        "--delay-ms",
        str(args.delay_ms),
        "--jitter-ms",
        str(args.jitter_ms),
        "--seed",
        str(args.seed),
    ]
    scenario.receiver_log.parent.mkdir(parents=True, exist_ok=True)
    receiver_log = scenario.receiver_log.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=root, stdout=receiver_log, stderr=subprocess.STDOUT)
    process._demo_log_handle = receiver_log  # type: ignore[attr-defined]
    time.sleep(0.4)
    return process


def stop_receiver(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)
    log_handle = getattr(process, "_demo_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def sender_command(root: Path, scenario: Scenario, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(root / "sender.py"),
        "--target-host",
        "127.0.0.1",
        "--target-port",
        str(scenario.receiver_port),
        "--local-host",
        "127.0.0.1",
        "--local-port",
        str(scenario.sender_port),
        "--packets",
        str(scenario.packets),
        "--window-size",
        str(scenario.window_size),
        "--rto",
        str(args.rto),
        "--cc-mode",
        scenario.mode,
        "--max-cwnd",
        str(args.max_cwnd),
        "--metrics-file",
        str(scenario.metrics_file),
        "--history-file",
        str(scenario.history_file),
        *scenario.extra_args,
    ]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def latest_row_for_run(path: Path, mode: str) -> dict[str, str] | None:
    rows = read_metrics(path)
    for row in reversed(rows):
        if row.get("mode") == mode:
            return row
    return rows[-1] if rows else None


def make_scenarios(root: Path, result_dir: Path, args: argparse.Namespace) -> tuple[list[Scenario], list[str]]:
    main_metrics = result_dir / "metrics_main.csv"
    main_history = result_dir / "history_main.csv"
    drop_metrics = result_dir / "metrics_drop.csv"
    logs = result_dir / "logs"
    notes: list[str] = []
    q_table = Path(args.q_table)
    if not q_table.is_absolute():
        q_table = root / q_table

    scenarios = [
        Scenario(
            name="AIMD baseline",
            mode="aimd",
            packets=args.packets,
            window_size=1,
            extra_args=[],
            metrics_file=main_metrics,
            history_file=main_history,
            sender_log=logs / "aimd_sender.log",
            receiver_log=logs / "aimd_receiver.log",
            receiver_port=args.base_port + 1,
            sender_port=args.base_port,
        ),
        Scenario(
            name="Q-Learning",
            mode="qlearning",
            packets=args.packets,
            window_size=1,
            extra_args=[
                "--epsilon",
                "0.0",
                "--q-eval",
                "--qtable-file",
                str(q_table),
            ],
            metrics_file=main_metrics,
            history_file=main_history,
            sender_log=logs / "qlearning_sender.log",
            receiver_log=logs / "qlearning_receiver.log",
            receiver_port=args.base_port + 11,
            sender_port=args.base_port + 10,
        ),
    ]

    dqn_requested = args.include_dqn == "yes" or (args.include_dqn == "auto" and has_torch())
    if dqn_requested:
        scenarios.append(
            Scenario(
                name="DQN",
                mode="dqn",
                packets=args.packets,
                window_size=8,
                extra_args=[
                    "--epsilon",
                    "0.0",
                    "--dqn-eval",
                    "--dqn-model-file",
                    str(root / "artifacts" / "models" / "active" / "dqn_model.pt"),
                    "--dqn-batch-size",
                    "32",
                ],
                metrics_file=main_metrics,
                history_file=main_history,
                sender_log=logs / "dqn_sender.log",
                receiver_log=logs / "dqn_receiver.log",
                receiver_port=args.base_port + 21,
                sender_port=args.base_port + 20,
            )
        )
    else:
        notes.append("DQN skipped: PyTorch is not installed or --include-dqn=no was selected.")

    # --- Bandwidth drop scenarios: compare AIMD vs Q-Learning recovery ---
    drop_at = str(max(10, args.packets // 2))
    drop_factor = "0.5"

    scenarios.append(
        Scenario(
            name="AIMD bandwidth drop",
            mode="aimd",
            packets=args.packets,
            window_size=1,
            extra_args=[
                "--link-bandwidth-drop-after-packets", drop_at,
                "--link-bandwidth-drop-factor", drop_factor,
            ],
            metrics_file=drop_metrics,
            history_file=result_dir / "history_drop_aimd.csv",
            sender_log=logs / "drop_aimd_sender.log",
            receiver_log=logs / "drop_aimd_receiver.log",
            receiver_port=args.base_port + 41,
            sender_port=args.base_port + 40,
        )
    )
    scenarios.append(
        Scenario(
            name="Q-Learning bandwidth drop",
            mode="qlearning",
            packets=args.packets,
            window_size=1,
            extra_args=[
                "--epsilon", "0.0",
                "--q-eval",
                "--qtable-file", str(q_table),
                "--link-bandwidth-drop-after-packets", drop_at,
                "--link-bandwidth-drop-factor", drop_factor,
            ],
            metrics_file=drop_metrics,
            history_file=result_dir / "history_drop_qlearn.csv",
            sender_log=logs / "drop_qlearn_sender.log",
            receiver_log=logs / "drop_qlearn_receiver.log",
            receiver_port=args.base_port + 51,
            sender_port=args.base_port + 50,
        )
    )

    return scenarios, notes


def plot_if_possible(root: Path, result_dir: Path, metrics: Path, history: Path, output_name: str) -> str:
    output = result_dir / output_name
    command = [
        sys.executable,
        str(root / "plot_metrics.py"),
        "--metrics-file",
        str(metrics),
        "--history-file",
        str(history),
        "--output",
        str(output),
    ]
    completed = run_command(command, root, result_dir / f"{output.stem}_plot.log")
    return "ok" if completed.returncode == 0 else f"failed, see {output.stem}_plot.log"


def plot_drop_comparison(root: Path, result_dir: Path) -> str:
    """Generate CWND recovery + metrics bar chart for bandwidth drop comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "matplotlib not available"

    aimd_hist = result_dir / "history_drop_aimd.csv"
    ql_hist = result_dir / "history_drop_qlearn.csv"
    drop_metrics = result_dir / "metrics_drop.csv"

    if not aimd_hist.exists() or not ql_hist.exists():
        return "drop history files missing"

    def load_history(path):
        times, cwnds = [], []
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                times.append(float(r["time_s"]))
                cwnds.append(float(r["cwnd"]))
        return times, cwnds

    def load_metrics(path, mode):
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("mode") == mode:
                    return r
        return None

    aimd_t, aimd_c = load_history(aimd_hist)
    ql_t, ql_c = load_history(ql_hist)
    aimd_m = load_metrics(drop_metrics, "aimd")
    ql_m = load_metrics(drop_metrics, "qlearning")

    if not aimd_m or not ql_m:
        return "drop metrics rows missing"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    # --- CWND recovery traces ---
    ax1.step(aimd_t, aimd_c, where="post", label="AIMD", linewidth=1.6, color="#2563eb")
    ax1.step(ql_t, ql_c, where="post", label="Q-Learning", linewidth=1.6, color="#d97706")
    # Mark approximate drop point at half of max time
    drop_t = max(aimd_t) * 0.45 if aimd_t else 0
    ax1.axvline(x=drop_t, color="red", linestyle="--", alpha=0.5, linewidth=1.5,
                label="Bandwidth halving")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("CWND (packets)")
    ax1.set_title("CWND Recovery: AIMD vs Q-Learning under Bandwidth Halving")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Annotate peaks
    if aimd_c:
        aimd_peak = max(aimd_c)
        aimd_pi = aimd_c.index(aimd_peak)
        ax1.annotate(f"AIMD peak={aimd_peak:.0f}", xy=(aimd_t[aimd_pi], aimd_peak),
                     xytext=(0, 20), textcoords="offset points", fontsize=9, color="#2563eb",
                     arrowprops=dict(arrowstyle="->", color="#2563eb", alpha=0.5))
    if ql_c:
        ql_peak = max(ql_c)
        ql_pi = ql_c.index(ql_peak)
        ax1.annotate(f"Q-L peak={ql_peak:.0f}", xy=(ql_t[ql_pi], ql_peak),
                     xytext=(0, -25), textcoords="offset points", fontsize=9, color="#d97706",
                     arrowprops=dict(arrowstyle="->", color="#d97706", alpha=0.5))

    # --- Metrics bar chart ---
    labels = ["Throughput\n(Mbps)", "Avg RTT\n(ms)", "Retrans-\nmissions", "Timeouts"]
    aimd_vals = [
        float(aimd_m.get("throughput_mbps", 0)),
        float(aimd_m.get("avg_rtt_ms", 0)),
        float(aimd_m.get("retransmissions", 0)),
        float(aimd_m.get("timeout_events", 0)),
    ]
    ql_vals = [
        float(ql_m.get("throughput_mbps", 0)),
        float(ql_m.get("avg_rtt_ms", 0)),
        float(ql_m.get("retransmissions", 0)),
        float(ql_m.get("timeout_events", 0)),
    ]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, aimd_vals, width, label="AIMD", color="#2563eb", alpha=0.85)
    bars2 = ax2.bar(x + width / 2, ql_vals, width, label="Q-Learning", color="#d97706", alpha=0.85)

    for bar, val in zip(bars1, aimd_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(aimd_vals) * 0.02,
                 f"{val:.1f}" if val < 100 else str(int(val)),
                 ha="center", va="bottom", fontsize=8, color="#2563eb")
    for bar, val in zip(bars2, ql_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ql_vals) * 0.02,
                 f"{val:.1f}" if val < 100 else str(int(val)),
                 ha="center", va="bottom", fontsize=8, color="#d97706")

    # Improvement percentages
    improvements = []
    for a, q in zip(aimd_vals, ql_vals):
        if a != 0:
            pct = (q - a) / abs(a) * 100
            improvements.append(f"{pct:+.1f}%")
        else:
            improvements.append("-")
    for i, imp in enumerate(improvements):
        y_pos = max(aimd_vals[i], ql_vals[i]) * 0.55
        ax2.text(x[i], y_pos, imp, ha="center", fontsize=9, fontweight="bold",
                 color="#16a34a", bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", alpha=0.8))

    ax2.set_xticks(x, labels)
    ax2.set_title("Performance Metrics under Bandwidth Halving")
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", alpha=0.3)

    output = result_dir / "comparison_drop.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return "ok"


def html_link(path: Path, base: Path, label: str | None = None) -> str:
    if not path.exists():
        return "not generated"
    rel = path.relative_to(base)
    text = label or rel.name
    return f'<a href="{html.escape(str(rel))}">{html.escape(text)}</a>'


def write_dashboard(
    root: Path,
    result_dir: Path,
    scenarios: list[Scenario],
    notes: list[str],
    plot_status: list[str],
) -> Path:
    cards = []
    for scenario in scenarios:
        row = latest_row_for_run(scenario.metrics_file, scenario.mode)
        sender_log = read_text(scenario.sender_log)
        receiver_log = read_text(scenario.receiver_log)
        acked = row.get("acked", "0") if row else "0"
        packets = row.get("packets", str(scenario.packets)) if row else str(scenario.packets)
        reliable = acked == packets
        markers = {
            "DONE": "[SENDER][DONE]" in sender_log,
            "QLEARN": "[SENDER][QLEARN]" in sender_log,
            "DQN": "[SENDER][DQN]" in sender_log,
            "FAST": "[SENDER][FAST]" in sender_log,
            "RTO": "[SENDER][RTO]" in sender_log,
            "BANDWIDTH": "[VLINK][BANDWIDTH]" in sender_log,
            "OUT_OF_ORDER": "out_of_order" in receiver_log,
        }
        cards.append((scenario, row, reliable, markers))

    rows_html = []
    for scenario, row, reliable, markers in cards:
        row = row or {}
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(scenario.name)}</td>"
            f"<td>{html.escape(row.get('mode', scenario.mode))}</td>"
            f"<td>{html.escape(row.get('acked', '0'))}/{html.escape(row.get('packets', str(scenario.packets)))}</td>"
            f"<td>{'PASS' if reliable else 'CHECK'}</td>"
            f"<td>{html.escape(row.get('throughput_mbps', '-'))}</td>"
            f"<td>{html.escape(row.get('avg_rtt_ms', '-'))}</td>"
            f"<td>{html.escape(row.get('retransmissions', '-'))}</td>"
            f"<td>{html.escape(row.get('fast_retransmissions', '-'))}</td>"
            f"<td>{html_link(scenario.sender_log, result_dir, 'sender log')} / {html_link(scenario.receiver_log, result_dir, 'receiver log')}</td>"
            "</tr>"
        )

    checks_html = []
    for scenario, _, reliable, markers in cards:
        active_markers = ", ".join(name for name, present in markers.items() if present) or "none"
        checks_html.append(
            f"<li><strong>{html.escape(scenario.name)}</strong>: "
            f"{'acked all packets' if reliable else 'packet count needs attention'}; "
            f"log markers: {html.escape(active_markers)}</li>"
        )

    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in notes) or "<li>No skipped optional step.</li>"
    plots_html = "".join(f"<li>{html.escape(status)}</li>" for status in plot_status)
    main_plot = result_dir / "comparison_main.png"
    drop_plot = result_dir / "comparison_drop.png"
    main_plot_image = (
        f'<img src="{html.escape(str(main_plot.relative_to(result_dir)))}" alt="main comparison">'
        if main_plot.exists()
        else "<p>main comparison plot was not generated.</p>"
    )
    drop_plot_image = (
        f'<img src="{html.escape(str(drop_plot.relative_to(result_dir)))}" alt="bandwidth drop comparison">'
        if drop_plot.exists()
        else "<p>bandwidth drop plot was not generated.</p>"
    )

    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>UDP Reliable Transport Demo Results</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.5; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>UDP 可靠传输与拥塞控制一键验证结果</h1>
  <p>生成时间：{html.escape(time.strftime('%Y-%m-%d %H:%M:%S'))}</p>

  <h2>结果汇总</h2>
  <table>
    <thead>
      <tr><th>场景</th><th>模式</th><th>ACK</th><th>可靠性</th><th>吞吐量 Mbps</th><th>平均 RTT ms</th><th>重传</th><th>快速重传</th><th>日志</th></tr>
    </thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>

  <h2>功能观察点</h2>
  <ul>{''.join(checks_html)}</ul>

  <h2>图表</h2>
  <ul>{plots_html}</ul>
  <div class="grid">
    <div><h3>AIMD / Q-Learning / DQN 对比</h3>{html_link(main_plot, result_dir)}<br>{main_plot_image}</div>
    <div><h3>带宽突变场景</h3>{html_link(drop_plot, result_dir)}<br>{drop_plot_image}</div>
  </div>

  <h2>备注</h2>
  <ul>{notes_html}</ul>

  <h2>如何复现</h2>
  <p>双击项目根目录的 <code>Run_Demo.command</code>，或在终端执行：</p>
  <pre><code>python3 demo_runner.py</code></pre>
</body>
</html>
"""
    index = result_dir / "index.html"
    index.write_text(body, encoding="utf-8")
    return index


def main() -> None:
    args = build_parser().parse_args()
    if args.packets <= 0:
        raise SystemExit("--packets must be positive")
    if not 0.0 <= args.loss_rate <= 1.0:
        raise SystemExit("--loss-rate must be in [0, 1]")

    root = Path(__file__).resolve().parent
    run_name = time.strftime("%Y%m%d-%H%M%S")
    if args.result_tag:
        run_name = f"{run_name}-{args.result_tag}"
    result_dir = root / "artifacts" / "demo_results" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    scenarios, notes = make_scenarios(root, result_dir, args)

    print(f"[DEMO] results: {result_dir}")
    if args.dry_run:
        for scenario in scenarios:
            print("[DRY-RUN]", " ".join(sender_command(root, scenario, args)))
        return

    for scenario in scenarios:
        print(f"[DEMO] running {scenario.name} ({scenario.mode}) ...")
        receiver = start_receiver(root, scenario, args)
        try:
            completed = run_command(sender_command(root, scenario, args), root, scenario.sender_log)
            if completed.returncode != 0:
                notes.append(f"{scenario.name} sender exited with code {completed.returncode}; inspect logs.")
        finally:
            stop_receiver(receiver)

    plot_status = []
    main_metrics = result_dir / "metrics_main.csv"
    main_history = result_dir / "history_main.csv"
    plot_status.append("main comparison: " + plot_if_possible(root, result_dir, main_metrics, main_history, "comparison_main.png"))
    drop_status = plot_drop_comparison(root, result_dir)
    plot_status.append("bandwidth drop comparison (CWND + metrics): " + drop_status)

    index = write_dashboard(root, result_dir, scenarios, notes, plot_status)
    print(f"[DEMO] dashboard: {index}")

    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(index)], check=False)


if __name__ == "__main__":
    main()
