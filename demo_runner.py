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

    # Bandwidth drop is handled separately by bandwidth_drop_experiment.py
    # which runs both AIMD and Q-Learning with proper acked-based halving,
    # adaptation mechanism, and recovery-time calculation.

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
    """Run bandwidth_drop_experiment.py (with timeout) and copy its plots into result_dir.
    Falls back to cached results if available."""
    import shutil
    exp_script = root / "bandwidth_drop_experiment.py"
    if not exp_script.exists():
        return "bandwidth_drop_experiment.py not found"

    exp_out = root / "artifacts" / "bandwidth_drop_experiment"

    # Try running the experiment with timeout
    try:
        print("[DEMO] running bandwidth drop experiment (AIMD/Q-Learning, packets=300) ...", flush=True)
        completed = subprocess.run(
            [sys.executable, str(exp_script), "--packets", "300", "--no-plot"],
            cwd=root, timeout=90,
        )
        ok = completed.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False

    if not ok:
        # Fall back to cached results
        if (exp_out / "summary.json").exists():
            print("[DEMO] bandwidth drop experiment timed out; using cached results", flush=True)
        else:
            return "experiment timed out and no cached results available"

    # Generate plots from the experiment's telemetry
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import json
    except ImportError:
        return "matplotlib not available"

    summary_path = exp_out / "summary.json"
    if not summary_path.exists():
        return "experiment summary.json missing"

    with summary_path.open() as f:
        summary = json.load(f)

    results = {r["mode"]: r for r in summary["results"]}
    if "aimd" not in results or "qlearning" not in results:
        return "experiment results incomplete"

    aimd = results["aimd"]
    ql = results["qlearning"]
    ht = aimd["bandwidth_halving_time_s"]

    # Load CWND history from telemetry CSVs
    def load_cwnd(path):
        ts, cs = [], []
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                ts.append(float(r["time_s"]))
                cs.append(float(r["cwnd"]))
        return ts, cs

    aimd_t, aimd_c = load_cwnd(exp_out / "telemetry_aimd.csv")
    ql_t, ql_c = load_cwnd(exp_out / "telemetry_qlearn.csv")

    # ── Figure 1: CWND Recovery + RTT ──
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

    ax1.step(aimd_t, aimd_c, where="post", label="AIMD", lw=1.6, color="#2563eb")
    ax1.step(ql_t, ql_c, where="post", label="Q-Learning", lw=1.6, color="#d97706")
    ax1.axvline(x=ht, color="red", ls="--", lw=1.8, alpha=0.7,
                label=f"Bandwidth halving (t={ht:.2f}s)")

    # Peaks
    for data, color, label, yoff in [(aimd_c, "#2563eb", "AIMD", 3), (ql_c, "#d97706", "Q-L", -5)]:
        peak_v = max(data)
        peak_i = data.index(peak_v)
        ax1.annotate(f"{label} peak={peak_v:.0f}", xy=(aimd_t[peak_i] if label == "AIMD" else ql_t[peak_i], peak_v),
                     xytext=(0, yoff), textcoords="offset points", fontsize=10, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, alpha=0.5), fontweight="bold")

    # Recovery spans
    for res, color, label, y in [(aimd, "#2563eb", "AIMD", 1.5), (ql, "#d97706", "Q-L", 1.0)]:
        rt = res["recovery_time_s"]
        if rt != float("inf") and rt > 0:
            ax1.axvspan(ht, ht + rt, alpha=0.08, color=color)
            ax1.text(ht + rt / 2, y, f"{label} recovery\n{rt:.2f}s",
                     ha="center", fontsize=8, color=color, fontweight="bold")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("CWND (packets)")
    ax1.set_title("CWND Recovery under Bandwidth Halving")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # RTT subplot
    def load_rtt(path):
        ts, rs = [], []
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                ts.append(float(r["time_s"]))
                rs.append(float(r.get("rtt_ms", 0)))
        return ts, rs

    for label, color in [("AIMD", "#2563eb"), ("Q-Learning", "#d97706")]:
        fname = f"telemetry_{'aimd' if label == 'AIMD' else 'qlearn'}.csv"
        rt_ts, rt_rs = load_rtt(exp_out / fname)
        ax2.plot(rt_ts, rt_rs, label=label, lw=1.2, alpha=0.8, color=color)
    ax2.axvline(x=ht, color="red", ls="--", lw=1.8, alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RTT (ms)")
    ax2.set_title("RTT Evolution under Bandwidth Halving")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    cwnd_path = result_dir / "comparison_drop.png"
    fig1.savefig(cwnd_path, dpi=150)
    plt.close(fig1)

    # ── Figure 2: Post-halving metrics ──
    fig2, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    metric_specs = [
        ("Recovery\nTime (s)", "recovery_time_s"),
        ("Throughput\nafter halving\n(Mbps)", "throughput_after_halving_mbps"),
        ("Avg RTT\nafter halving\n(ms)", "avg_rtt_after_halving_ms"),
        ("Retransmissions\nafter halving", "retransmissions_after_halving"),
    ]
    for ax, (title, key) in zip(axes, metric_specs):
        vals = [aimd[key], ql[key]]
        bars = ax.bar(["AIMD", "Q-Learning"], vals, color=["#2563eb", "#d97706"], alpha=0.85, width=0.5)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            txt = f"{val:.2f}" if isinstance(val, float) and val < 100 else str(int(val)) if isinstance(val, (int, float)) and val not in (float("inf"),) else "inf"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.03,
                    txt, ha="center", fontsize=10, fontweight="bold")
        if vals[0] not in (0, float("inf")) and vals[1] not in (float("inf"),):
            pct = (vals[1] - vals[0]) / abs(vals[0]) * 100
            clr = "#16a34a" if (("Throughput" in title and pct > 0) or ("Throughput" not in title and pct < 0)) else "#dc2626"
            ax.text(1, max(vals) * 0.5, f"{pct:+.1f}%", ha="center", fontsize=9, fontweight="bold",
                    color=clr, bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0fdf4", alpha=0.7))
    fig2.suptitle("Performance Metrics after Bandwidth Halving", fontsize=13, fontweight="bold")
    metrics_path = result_dir / "comparison_drop_metrics.png"
    fig2.savefig(metrics_path, dpi=150)
    plt.close(fig2)

    # Copy experiment outputs
    for name in ["summary.json", "conclusion.md", "telemetry_aimd.csv", "telemetry_qlearn.csv"]:
        src = exp_out / name
        if src.exists():
            shutil.copy2(src, result_dir / f"drop_{name}")

    return f"ok (cwnd_recovery + post_halving_metrics + {summary['experiment']['total_packets']} pkts)"


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
    main_plots = [
        (result_dir / "comparison_main.png", "CWND 对比", "main comparison cwnd"),
        (result_dir / "comparison_main_rtt.png", "平均 RTT 对比", "main comparison rtt"),
        (result_dir / "comparison_main_throughput.png", "吞吐量对比", "main comparison throughput"),
        (result_dir / "comparison_main_retransmissions.png", "重传次数对比", "main comparison retransmissions"),
        (result_dir / "comparison_main_timeouts.png", "Timeout 次数对比", "main comparison timeouts"),
    ]
    drop_cwnd_plot = result_dir / "comparison_drop.png"
    drop_metrics_plot = result_dir / "comparison_drop_metrics.png"

    def image_panel(path: Path, title: str, alt: str) -> str:
        if not path.exists():
            return f"<div><h3>{html.escape(title)}</h3><p>plot was not generated.</p></div>"
        return (
            f"<div><h3>{html.escape(title)}</h3>"
            f"{html_link(path, result_dir)}<br>"
            f'<img src="{html.escape(str(path.relative_to(result_dir)))}" alt="{html.escape(alt)}">'
            "</div>"
        )

    main_plot_images = "".join(image_panel(path, title, alt) for path, title, alt in main_plots)
    drop_cwnd_image = (
        f'<img src="{html.escape(str(drop_cwnd_plot.relative_to(result_dir)))}" alt="CWND recovery">'
        if drop_cwnd_plot.exists()
        else "<p>CWND recovery plot was not generated.</p>"
    )
    drop_metrics_image = (
        f'<img src="{html.escape(str(drop_metrics_plot.relative_to(result_dir)))}" alt="post-halving metrics">'
        if drop_metrics_plot.exists()
        else "<p>post-halving metrics plot was not generated.</p>"
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
    {main_plot_images}
    <div><h3>带宽突变 — CWND 恢复</h3>{html_link(drop_cwnd_plot, result_dir)}<br>{drop_cwnd_image}</div>
    <div><h3>带宽突变 — 减半后指标</h3>{html_link(drop_metrics_plot, result_dir)}<br>{drop_metrics_image}</div>
  </div>

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
