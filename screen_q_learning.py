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
    candidate: Path,
    seed: int,
    scenario: str,
    case_index: int,
    packets: int,
    receiver_port: int,
    sender_port: int,
    output_dir: Path,
    online_update: bool,
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
    if online_update:
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
        "1",
        "--rto",
        "0.20",
        "--cc-mode",
        "qlearning",
        "--max-cwnd",
        "64",
        "--epsilon",
        "0",
        "--qtable-file",
        str(run_q_table),
        "--metrics-file",
        str(metrics_file),
        "--history-file",
        str(history_file),
        "--quiet",
    ]
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
    try:
        time.sleep(0.15)
        subprocess.run(sender_cmd, cwd=root, check=True)
    finally:
        receiver.terminate()
        try:
            receiver.wait(timeout=2)
        except subprocess.TimeoutExpired:
            receiver.kill()
            receiver.wait()
    return latest_metric(metrics_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed Q-table screening")
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--seeds", default="3,7,11")
    parser.add_argument("--packets", type=int, default=300)
    parser.add_argument("--base-port", type=int, default=28000)
    parser.add_argument("--output-dir", default="artifacts/training/q_screen")
    parser.add_argument(
        "--online-update",
        action="store_true",
        help="allow Bellman updates from a fresh copy of each candidate for every run",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    if args.packets <= 0:
        raise SystemExit("--packets must be positive")

    root = Path(__file__).resolve().parent
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    case_index = 0
    for name, raw_path in args.candidate:
        candidate = (root / raw_path).resolve()
        if not candidate.exists():
            raise SystemExit(f"candidate does not exist: {candidate}")
        for seed in seeds:
            for scenario in ("normal", "drop"):
                receiver_port = args.base_port + case_index * 2
                sender_port = receiver_port + 1
                row = run_case(
                    root,
                    name,
                    candidate,
                    seed,
                    scenario,
                    case_index,
                    args.packets,
                    receiver_port,
                    sender_port,
                    output_dir,
                    args.online_update,
                )
                result = {
                    "candidate": name,
                    "path": str(candidate.relative_to(root)),
                    "seed": seed,
                    "scenario": scenario,
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
                    "score={score:.4f}".format(**result),
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
