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
        if row.get("mode") == "dqn":
            return row
    return None


def save_checkpoint(source: Path, checkpoint_dir: Path, round_index: int, row: dict[str, str] | None) -> Path | None:
    if not source.exists():
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id = (row or {}).get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    target = checkpoint_dir / f"dqn_round_{round_index:03d}_{run_id}.pt"
    shutil.copy2(source, target)
    return target


def reset_model(model_path: Path, backup_dir: Path) -> Path | None:
    if not model_path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{model_path.stem}_backup_{time.strftime('%Y%m%d-%H%M%S')}{model_path.suffix}"
    shutil.copy2(model_path, backup)
    model_path.unlink()
    return backup


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
        return f"[DQN-TRAIN] round={round_index}/{rounds} epsilon={epsilon:.3f} metrics=missing ckpt={checkpoint or '-'}"
    return (
        "[DQN-TRAIN] round={round_no}/{rounds} epsilon={epsilon:.3f} "
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


def is_storm_round(
    row: dict[str, str] | None,
    retx_threshold: int,
    duration_threshold: float,
) -> bool:
    if row is None:
        return False
    try:
        retransmissions = int(row.get("retransmissions", "0") or 0)
        duration = float(row.get("duration_s", "0") or 0.0)
    except ValueError:
        return False
    return retransmissions > retx_threshold or duration > duration_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-round DQN congestion-control trainer")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use a faster training preset for quick iteration",
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--packets", type=int, default=160)
    parser.add_argument("--receiver-port", type=int, default=9301)
    parser.add_argument("--sender-port", type=int, default=9300)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--max-window", type=int, default=40)
    parser.add_argument("--rto", type=float, default=0.30)
    parser.add_argument("--dqn-model", default="artifacts/models/active/dqn_model.pt")
    parser.add_argument("--reset-dqn-model", action="store_true", help="backup and remove the active model before training")
    parser.add_argument("--model-backup-dir", default="artifacts/models/backups")
    parser.add_argument("--dqn-lr", type=float, default=0.0007)
    parser.add_argument("--dqn-batch-size", type=int, default=32)
    parser.add_argument("--dqn-replay-capacity", type=int, default=4096)
    parser.add_argument("--dqn-target-update", type=int, default=20)
    parser.add_argument("--dqn-gamma", type=float, default=0.90)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--reward-throughput-weight", type=float, default=1.8)
    parser.add_argument("--reward-timeout-weight", type=float, default=18.0)
    parser.add_argument("--reward-retx-weight", type=float, default=3.0)
    parser.add_argument("--reward-rtt-weight", type=float, default=0.004)
    parser.add_argument("--reward-cwnd-weight", type=float, default=1.0,
                        help="CWND efficiency bonus: encourages DQN to maintain higher CWND")
    parser.add_argument("--epsilon-decay", type=float, default=0.95)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--loss-rate", type=float, default=0.02)
    parser.add_argument("--delay-ms", type=float, default=20.0)
    parser.add_argument("--jitter-ms", type=float, default=10.0)
    parser.add_argument("--link-service-delay-ms", type=float, default=10.0)
    parser.add_argument("--link-queue-capacity", type=int, default=20)
    parser.add_argument("--link-bandwidth-drop-after-packets", type=int, default=0)
    parser.add_argument("--link-bandwidth-drop-factor", type=float, default=0.5)
    parser.add_argument("--metrics-file", default="artifacts/training/dqn_metrics.csv")
    parser.add_argument("--history-file", default="artifacts/training/dqn_history.csv")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/dqn")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--summary-file", default="artifacts/training/dqn_summary.csv")
    parser.add_argument("--storm-retx-threshold", type=int, default=1000)
    parser.add_argument("--storm-duration-threshold", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet-sender", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verbose-sender", action="store_true")
    return parser


def collect_cli_options(argv: list[str]) -> set[str]:
    return {item.split("=", 1)[0] for item in argv if item.startswith("--")}


def apply_fast_preset(args: argparse.Namespace, provided_options: set[str]) -> None:
    if not args.fast:
        return
    replacements = {
        "packets": ("--packets", 80),
        "loss_rate": ("--loss-rate", 0.03),
        "delay_ms": ("--delay-ms", 5.0),
        "jitter_ms": ("--jitter-ms", 2.0),
        "rto": ("--rto", 0.10),
        "dqn_batch_size": ("--dqn-batch-size", 8),
        "dqn_replay_capacity": ("--dqn-replay-capacity", 512),
        "link_service_delay_ms": ("--link-service-delay-ms", 2.0),
        "link_bandwidth_drop_after_packets": ("--link-bandwidth-drop-after-packets", 0),
        "checkpoint_every": ("--checkpoint-every", 5),
    }
    for name, (option, value) in replacements.items():
        if option not in provided_options:
            setattr(args, name, value)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_fast_preset(args, collect_cli_options(sys.argv[1:]))
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.packets <= 0:
        raise SystemExit("--packets must be positive")
    if not 0.0 <= args.loss_rate <= 1.0:
        raise SystemExit("--loss-rate must be in [0, 1]")
    if args.delay_ms < 0:
        raise SystemExit("--delay-ms must be non-negative")
    if args.jitter_ms < 0:
        raise SystemExit("--jitter-ms must be non-negative")
    if args.link_service_delay_ms < 0:
        raise SystemExit("--link-service-delay-ms must be non-negative")
    if args.link_queue_capacity <= 0:
        raise SystemExit("--link-queue-capacity must be positive")
    if args.link_bandwidth_drop_after_packets < 0:
        raise SystemExit("--link-bandwidth-drop-after-packets must be non-negative")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")
    if args.dqn_lr <= 0:
        raise SystemExit("--dqn-lr must be positive")
    if args.dqn_batch_size <= 0:
        raise SystemExit("--dqn-batch-size must be positive")
    if args.dqn_replay_capacity <= 0:
        raise SystemExit("--dqn-replay-capacity must be positive")
    if args.dqn_target_update <= 0:
        raise SystemExit("--dqn-target-update must be positive")
    if not 0 <= args.dqn_gamma <= 1:
        raise SystemExit("--dqn-gamma must be between 0 and 1")
    if args.reward_throughput_weight < 0:
        raise SystemExit("--reward-throughput-weight must be non-negative")
    if args.reward_timeout_weight < 0:
        raise SystemExit("--reward-timeout-weight must be non-negative")
    if args.reward_retx_weight < 0:
        raise SystemExit("--reward-retx-weight must be non-negative")
    if args.reward_rtt_weight < 0:
        raise SystemExit("--reward-rtt-weight must be non-negative")
    if args.reward_cwnd_weight < 0:
        raise SystemExit("--reward-cwnd-weight must be non-negative")
    if args.storm_retx_threshold < 0:
        raise SystemExit("--storm-retx-threshold must be non-negative")
    if args.storm_duration_threshold < 0:
        raise SystemExit("--storm-duration-threshold must be non-negative")

    root = Path(__file__).resolve().parent
    dqn_model = str((root / args.dqn_model).resolve())
    dqn_model_path = Path(dqn_model)
    metrics_file = str((root / args.metrics_file).resolve())
    history_file = str((root / args.history_file).resolve())
    checkpoint_dir = (root / args.checkpoint_dir).resolve()
    summary_file = (root / args.summary_file).resolve()
    model_backup_dir = (root / args.model_backup_dir).resolve()
    if args.reset_dqn_model:
        backup = reset_model(dqn_model_path, model_backup_dir)
        if backup is None:
            print(f"[DQN-TRAIN] no existing model to reset at {dqn_model_path}", flush=True)
        else:
            print(f"[DQN-TRAIN] backed up old model to {backup}", flush=True)
            print(f"[DQN-TRAIN] starting from a fresh DQN model at {dqn_model_path}", flush=True)
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
        print("[DQN-TRAIN] tqdm is not installed; install it with: python3 -m pip install tqdm", flush=True)
    else:
        progress = tqdm(
            total=args.rounds,
            desc="DQN training",
            unit="round",
            dynamic_ncols=True,
        )

    try:
        epsilon_scale = 1.0
        for round_index in range(args.rounds):
            epsilon = max(
                args.min_epsilon,
                args.epsilon * (args.epsilon_decay ** round_index) * epsilon_scale,
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
                "--link-service-delay-ms",
                str(args.link_service_delay_ms),
                "--link-queue-capacity",
                str(args.link_queue_capacity),
                "--cc-mode",
                "dqn",
                "--max-cwnd",
                str(args.max_window),
                "--epsilon",
                str(epsilon),
                "--q-gamma",
                str(args.dqn_gamma),
                "--reward-throughput-weight",
                str(args.reward_throughput_weight),
                "--reward-timeout-weight",
                str(args.reward_timeout_weight),
                "--reward-retx-weight",
                str(args.reward_retx_weight),
                "--reward-rtt-weight",
                str(args.reward_rtt_weight),
                "--reward-cwnd-weight",
                str(args.reward_cwnd_weight),
                "--dqn-model-file",
                dqn_model,
                "--dqn-lr",
                str(args.dqn_lr),
                "--dqn-batch-size",
                str(args.dqn_batch_size),
                "--dqn-replay-capacity",
                str(args.dqn_replay_capacity),
                "--dqn-target-update",
                str(args.dqn_target_update),
                "--link-bandwidth-drop-after-packets",
                str(args.link_bandwidth_drop_after_packets),
                "--link-bandwidth-drop-factor",
                str(args.link_bandwidth_drop_factor),
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
                    source=Path(dqn_model),
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
            if is_storm_round(
                latest_metrics,
                retx_threshold=args.storm_retx_threshold,
                duration_threshold=args.storm_duration_threshold,
            ):
                epsilon_scale = max(0.25, epsilon_scale * 0.5)
                print(
                    "[DQN-TRAIN] storm detected; reducing future exploration "
                    f"epsilon_scale={epsilon_scale:.3f}",
                    flush=True,
                )
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
