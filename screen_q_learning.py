from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must use NAME=PATH")
    return name, Path(raw_path)


def latest_metric(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"no metrics written to {path}")
    return rows[-1]


def metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def score(row: dict[str, str]) -> float:
    return (
        metric(row, "throughput_mbps")
        - 0.003 * max(0.0, metric(row, "avg_rtt_ms") - 40.0)
        - 0.004 * metric(row, "retransmissions")
        - 0.030 * metric(row, "timeout_events")
    )


def run_case(
    root: Path,
    candidate_name: str,
    candidate: Path | None,
    mode: str,
    seed: int,
    scenario: str,
    case_index: int,
    packets: int,
    receiver_port: int,
    sender_port: int,
    output_dir: Path,
    online_update: bool,
    case_timeout_s: float,
    rto: float,
    q_control_interval_ms: float,
    q_low_window_control_interval_ms: float | None,
    q_low_window_threshold: int,
    q_additive_step: int,
    fast_retransmit_threshold: int,
    initial_cwnd: int,
) -> dict[str, str]:
    run_name = f"{case_index:03d}_{candidate_name}_s{seed}_{scenario}"
    metrics_file = output_dir / f"{run_name}_metrics.csv"
    history_file = output_dir / f"{run_name}_history.csv"
    receiver_cmd = [
        sys.executable,
        str(root / "receiver.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(receiver_port),
        "--initial-seq",
        "0",
        "--loss-rate",
        "0.02",
        "--delay-ms",
        "10",
        "--jitter-ms",
        "3",
        "--seed",
        str(seed),
        "--quiet",
    ]
    run_q_table = candidate
    if mode == "qlearning" and online_update:
        assert candidate is not None
        run_q_table = output_dir / f"{run_name}_qtable.json"
        shutil.copy2(candidate, run_q_table)
    sender_cmd = [
        sys.executable,
        str(root / "sender.py"),
        "--target-host",
        "127.0.0.1",
        "--target-port",
        str(receiver_port),
        "--local-host",
        "127.0.0.1",
        "--local-port",
        str(sender_port),
        "--packets",
        str(packets),
        "--window-size",
        str(initial_cwnd),
        "--rto",
        str(rto),
        "--q-control-interval-ms",
        str(q_control_interval_ms),
        "--q-low-window-threshold",
        str(q_low_window_threshold),
        "--q-additive-step",
        str(q_additive_step),
        "--fast-retransmit-threshold",
        str(fast_retransmit_threshold),
        "--cc-mode",
        mode,
        "--max-cwnd",
        "64",
        "--metrics-file",
        str(metrics_file),
        "--history-file",
        str(history_file),
        "--quiet",
    ]
    if q_low_window_control_interval_ms is not None:
        sender_cmd.extend(
            [
                "--q-low-window-control-interval-ms",
                str(q_low_window_control_interval_ms),
            ]
        )
    if mode == "qlearning":
        assert run_q_table is not None
        sender_cmd.extend(["--epsilon", "0", "--qtable-file", str(run_q_table)])
        if not online_update:
            sender_cmd.append("--q-eval")
    if scenario == "drop":
        sender_cmd.extend(
            [
                "--link-bandwidth-drop-after-packets",
                str(packets // 2),
                "--link-bandwidth-drop-factor",
                "0.5",
            ]
        )

    receiver = subprocess.Popen(
        receiver_cmd,
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    status = "ok"
    try:
        time.sleep(0.15)
        subprocess.run(sender_cmd, cwd=root, check=True, timeout=case_timeout_s)
    except subprocess.TimeoutExpired:
        status = "timeout"
    except subprocess.CalledProcessError:
        status = "failed"
    finally:
        receiver.terminate()
        try:
            receiver.wait(timeout=2)
        except subprocess.TimeoutExpired:
            receiver.kill()
            receiver.wait()
    if status != "ok":
        # Keep the batch running and make a hung/broken policy unambiguously
        # worse than every completed candidate.
        return {
            "throughput_mbps": "0",
            "avg_rtt_ms": "10000",
            "retransmissions": str(packets),
            "timeout_events": str(packets),
            "_status": status,
        }
    row = latest_metric(metrics_file)
    row["_status"] = status
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed Q-table screening")
    parser.add_argument("--candidate", action="append", type=parse_candidate, default=[])
    parser.add_argument("--include-aimd", action="store_true")
    parser.add_argument("--seeds", default="3,7,11")
    parser.add_argument("--packets", type=int, default=300)
    parser.add_argument("--base-port", type=int, default=28000)
    parser.add_argument("--output-dir", default="artifacts/training/q_screen")
    parser.add_argument("--rto", type=float, default=0.20)
    parser.add_argument("--q-control-interval-ms", type=float, default=100.0)
    parser.add_argument("--q-low-window-control-interval-ms", type=float, default=None)
    parser.add_argument("--q-low-window-threshold", type=int, default=3)
    parser.add_argument("--q-additive-step", type=int, default=1)
    parser.add_argument("--fast-retransmit-threshold", type=int, default=3)
    parser.add_argument("--initial-cwnd", type=int, default=1)
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=45.0,
        help="maximum wall-clock seconds for one sender case before marking it failed",
    )
    parser.add_argument(
        "--online-update",
        action="store_true",
        help="allow Bellman updates from a fresh copy of each candidate for every run",
    )
    args = parser.parse_args()
    if not args.include_aimd and not args.candidate:
        raise SystemExit("provide at least one --candidate or use --include-aimd")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    if args.packets <= 0:
        raise SystemExit("--packets must be positive")
    if args.case_timeout_s <= 0:
        raise SystemExit("--case-timeout-s must be positive")
    if args.rto <= 0:
        raise SystemExit("--rto must be positive")
    if args.q_control_interval_ms <= 0:
        raise SystemExit("--q-control-interval-ms must be positive")
    if (
        args.q_low_window_control_interval_ms is not None
        and args.q_low_window_control_interval_ms <= 0
    ):
        raise SystemExit("--q-low-window-control-interval-ms must be positive")
    if args.q_low_window_threshold < 1:
        raise SystemExit("--q-low-window-threshold must be at least 1")
    if args.q_additive_step < 1:
        raise SystemExit("--q-additive-step must be at least 1")
    if args.fast_retransmit_threshold <= 0:
        raise SystemExit("--fast-retransmit-threshold must be positive")
    if args.initial_cwnd < 1:
        raise SystemExit("--initial-cwnd must be at least 1")

    root = Path(__file__).resolve().parent
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    case_index = 0
    candidates: list[tuple[str, Path | None, str]] = []
    if args.include_aimd:
        candidates.append(("aimd", None, "aimd"))
    for name, raw_path in args.candidate:
        candidate = (root / raw_path).resolve()
        if not candidate.exists():
            raise SystemExit(f"candidate does not exist: {candidate}")
        candidates.append((name, candidate, "qlearning"))

    for name, candidate, mode in candidates:
        for seed in seeds:
            for scenario in ("normal", "drop"):
                receiver_port = args.base_port + case_index * 2
                sender_port = receiver_port + 1
                row = run_case(
                    root,
                    name,
                    candidate,
                    mode,
                    seed,
                    scenario,
                    case_index,
                    args.packets,
                    receiver_port,
                    sender_port,
                    output_dir,
                    args.online_update,
                    args.case_timeout_s,
                    args.rto,
                    args.q_control_interval_ms,
                    args.q_low_window_control_interval_ms,
                    args.q_low_window_threshold,
                    args.q_additive_step,
                    args.fast_retransmit_threshold,
                    args.initial_cwnd,
                )
                result = {
                    "candidate": name,
                    "path": "" if candidate is None else str(candidate.relative_to(root)),
                    "seed": seed,
                    "scenario": scenario,
                    "status": row.get("_status", "ok"),
                    "throughput_mbps": metric(row, "throughput_mbps"),
                    "avg_rtt_ms": metric(row, "avg_rtt_ms"),
                    "retransmissions": metric(row, "retransmissions"),
                    "timeout_events": metric(row, "timeout_events"),
                    "score": score(row),
                }
                run_rows.append(result)
                case_index += 1
                print(
                    "[SCREEN] {candidate} seed={seed} {scenario} "
                    "tp={throughput_mbps:.6f} rtt={avg_rtt_ms:.2f} "
                    "retx={retransmissions:.0f} timeout={timeout_events:.0f} "
                    "score={score:.4f} status={status}".format(**result),
                    flush=True,
                )

    fieldnames = list(run_rows[0])
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        groups[(str(row["candidate"]), str(row["scenario"]))].append(row)
    aggregate_rows = []
    for (name, scenario), rows in groups.items():
        aggregate = {"candidate": name, "scenario": scenario, "runs": len(rows)}
        for field in (
            "throughput_mbps",
            "avg_rtt_ms",
            "retransmissions",
            "timeout_events",
            "score",
        ):
            aggregate[field] = sum(float(row[field]) for row in rows) / len(rows)
        aggregate_rows.append(aggregate)
        print(
            "[AVERAGE] {candidate} {scenario} n={runs} tp={throughput_mbps:.6f} "
            "rtt={avg_rtt_ms:.2f} retx={retransmissions:.2f} "
            "timeout={timeout_events:.2f} score={score:.4f}".format(**aggregate),
            flush=True,
        )

    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)


if __name__ == "__main__":
    main()
