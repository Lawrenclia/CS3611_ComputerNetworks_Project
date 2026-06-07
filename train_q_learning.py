from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


def load_tqdm() -> Callable[..., object] | None:
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm


def read_latest_metrics(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in reversed(rows):
        if row.get("mode") == "qlearning":
            return row
    return None


def save_checkpoint(source: Path, checkpoint_dir: Path, round_index: int, row: dict[str, str] | None) -> Path | None:
    if not source.exists():
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id = (row or {}).get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    target = checkpoint_dir / f"qlearning_round_{round_index:03d}_{run_id}.json"
    shutil.copy2(source, target)
    return target


def append_summary(
    path: Path,
    round_index: int,
    epsilon: float,
    row: dict[str, str] | None,
    checkpoint: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "round",
        "epsilon",
        "checkpoint",
        "timestamp",
        "run_id",
        "mode",
        "packets",
        "acked",
        "duration_s",
        "throughput_mbps",
        "avg_rtt_ms",
        "srtt_ms",
        "retransmissions",
        "fast_retransmissions",
        "timeout_events",
    ]
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        source = row or {}
        writer.writerow(
            {
                "round": round_index,
                "epsilon": f"{epsilon:.6f}",
                "checkpoint": str(checkpoint) if checkpoint else "",
                **{name: source.get(name, "") for name in fieldnames[3:]},
            }
        )


def format_round_metrics(
    round_index: int,
    rounds: int,
    epsilon: float,
    row: dict[str, str] | None,
    checkpoint: Path | None,
) -> str:
    if row is None:
        return f"[TRAIN] round={round_index}/{rounds} epsilon={epsilon:.3f} metrics=missing ckpt={checkpoint or '-'}"
    return (
        "[TRAIN] round={round_no}/{rounds} epsilon={epsilon:.3f} "
        "acked={acked}/{packets} duration={duration}s throughput={throughput}Mbps "
        "avg_rtt={avg_rtt}ms srtt={srtt}ms retx={retx} fast={fast} timeout={timeout} "
        "ckpt={checkpoint}".format(
            round_no=round_index,
            rounds=rounds,
            epsilon=epsilon,
            acked=row.get("acked", "?"),
            packets=row.get("packets", "?"),
            duration=row.get("duration_s", "?"),
            throughput=row.get("throughput_mbps", "?"),
            avg_rtt=row.get("avg_rtt_ms", "?"),
            srtt=row.get("srtt_ms", "?"),
            retx=row.get("retransmissions", "?"),
            fast=row.get("fast_retransmissions", "?"),
            timeout=row.get("timeout_events", "?"),
            checkpoint=checkpoint or "-",
        )
    )


def tqdm_postfix(row: dict[str, str] | None, checkpoint: Path | None) -> dict[str, str]:
    if row is None:
        return {"metrics": "missing", "ckpt": checkpoint.name if checkpoint else "-"}
    return {
        "acked": f"{row.get('acked', '?')}/{row.get('packets', '?')}",
        "mbps": row.get("throughput_mbps", "?"),
        "rtt_ms": row.get("avg_rtt_ms", "?"),
        "retx": row.get("retransmissions", "?"),
        "ckpt": checkpoint.name if checkpoint else "-",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-round Q-Learning training runner")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--packets", type=int, default=120)
    parser.add_argument("--receiver-port", type=int, default=9201)
    parser.add_argument("--sender-port", type=int, default=9200)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--max-window", type=int, default=32)
    parser.add_argument("--rto", type=float, default=0.20)
    parser.add_argument("--q-table", default="q_table.json")
    parser.add_argument("--q-alpha", type=float, default=0.30)
    parser.add_argument("--q-gamma", type=float, default=0.85)
    parser.add_argument("--q-epsilon", type=float, default=0.30)
    parser.add_argument("--reward-throughput-weight", type=float, default=1.0)
    parser.add_argument("--reward-timeout-weight", type=float, default=10.0)
    parser.add_argument("--reward-retx-weight", type=float, default=2.0)
    parser.add_argument("--reward-rtt-weight", type=float, default=0.015)
    parser.add_argument("--epsilon-decay", type=float, default=0.85)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--loss-rate", type=float, default=0.08)
    parser.add_argument("--delay-ms", type=float, default=20.0)
    parser.add_argument("--jitter-ms", type=float, default=10.0)
    parser.add_argument("--metrics-file", default="artifacts/training/qlearning_metrics.csv")
    parser.add_argument("--history-file", default="artifacts/training/qlearning_history.csv")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/qlearning")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--summary-file", default="artifacts/training/qlearning_summary.csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet-sender", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verbose-sender", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.packets <= 0:
        raise SystemExit("--packets must be positive")
    if not 0.0 <= args.loss_rate <= 1.0:
        raise SystemExit("--loss-rate must be in [0, 1]")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")
    if args.reward_throughput_weight < 0:
        raise SystemExit("--reward-throughput-weight must be non-negative")
    if args.reward_timeout_weight < 0:
        raise SystemExit("--reward-timeout-weight must be non-negative")
    if args.reward_retx_weight < 0:
        raise SystemExit("--reward-retx-weight must be non-negative")
    if args.reward_rtt_weight < 0:
        raise SystemExit("--reward-rtt-weight must be non-negative")

    root = Path(__file__).resolve().parent
    q_table = str((root / args.q_table).resolve())
    metrics_file = str((root / args.metrics_file).resolve())
    history_file = str((root / args.history_file).resolve())
    checkpoint_dir = (root / args.checkpoint_dir).resolve()
    summary_file = (root / args.summary_file).resolve()
    receiver_cmd = [
        sys.executable,
        str(root / "receiver.py"),
        "--port",
        str(args.receiver_port),
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

    receiver = subprocess.Popen(
        receiver_cmd,
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    time.sleep(0.5)

    tqdm = load_tqdm()
    progress = None
    if tqdm is None:
        print("[TRAIN] tqdm is not installed; install it with: python3 -m pip install tqdm", flush=True)
    else:
        progress = tqdm(
            total=args.rounds,
            desc="Q-Learning training",
            unit="round",
            dynamic_ncols=True,
        )

    try:
        for round_index in range(args.rounds):
            epsilon = max(
                args.min_epsilon,
                args.q_epsilon * (args.epsilon_decay ** round_index),
            )
            local_port = args.sender_port + round_index
            if local_port == args.receiver_port:
                local_port += args.rounds + 1
            sender_cmd = [
                sys.executable,
                str(root / "sender.py"),
                "--target-port",
                str(args.receiver_port),
                "--local-port",
                str(local_port),
                "--packets",
                str(args.packets),
                "--start-seq",
                str(round_index * args.packets),
                "--window-size",
                str(args.window_size),
                "--rto",
                str(args.rto),
                "--cc-mode",
                "qlearning",
                "--max-cwnd",
                str(args.max_window),
                "--q-alpha",
                str(args.q_alpha),
                "--q-gamma",
                str(args.q_gamma),
                "--reward-throughput-weight",
                str(args.reward_throughput_weight),
                "--reward-timeout-weight",
                str(args.reward_timeout_weight),
                "--reward-retx-weight",
                str(args.reward_retx_weight),
                "--reward-rtt-weight",
                str(args.reward_rtt_weight),
                "--epsilon",
                str(epsilon),
                "--qtable-file",
                q_table,
                "--metrics-file",
                metrics_file,
                "--history-file",
                history_file,
            ]
            if args.quiet_sender or not args.verbose_sender:
                sender_cmd.append("--quiet")
            completed = subprocess.run(sender_cmd, cwd=root)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
            latest_metrics = read_latest_metrics(Path(metrics_file))
            checkpoint = None
            if (round_index + 1) % args.checkpoint_every == 0:
                checkpoint = save_checkpoint(
                    source=Path(q_table),
                    checkpoint_dir=checkpoint_dir,
                    round_index=round_index + 1,
                    row=latest_metrics,
                )
            append_summary(
                path=summary_file,
                round_index=round_index + 1,
                epsilon=epsilon,
                row=latest_metrics,
                checkpoint=checkpoint,
            )
            message = format_round_metrics(
                round_index=round_index + 1,
                rounds=args.rounds,
                epsilon=epsilon,
                row=latest_metrics,
                checkpoint=checkpoint,
            )
            if progress is not None:
                progress.set_postfix(tqdm_postfix(latest_metrics, checkpoint), refresh=False)
                progress.update(1)
                progress.write(message)
            else:
                print(message, flush=True)
    finally:
        if progress is not None:
            progress.close()
        receiver.terminate()
        try:
            receiver.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            receiver.kill()


if __name__ == "__main__":
    main()
