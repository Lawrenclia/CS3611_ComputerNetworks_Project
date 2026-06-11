from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

Q_STATE_NAMES = (
    "rtt_up|no_loss",
    "rtt_up|loss",
    "rtt_down|no_loss",
    "rtt_down|loss",
    "rtt_stable|no_loss",
    "rtt_stable|loss",
)
Q_ACTION_KEYS = ("0", "1", "2")


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


def copy_q_table(source: Path, target: Path) -> Path | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def seed_q_table(path: Path, backup_dir: Path) -> Path | None:
    backup = None
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{path.stem}_backup_{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}"
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": {
            "state_features": ["rtt_trend", "loss_flag"],
            "state_count": len(Q_STATE_NAMES),
            "rtt_trend_threshold_ratio": 0.05,
            "actions": {"0": "hold", "1": "cwnd+1", "2": "cwnd/2"},
        }
    }
    for state in Q_STATE_NAMES:
        trend, loss_flag = state.split("|")
        data[state] = seed_values_for_state(trend, loss_flag)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return backup


def metric_float(row: dict[str, str] | None, name: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        return float(row.get(name, "") or default)
    except ValueError:
        return default


def score_metrics(
    row: dict[str, str] | None,
    target_rtt_ms: float,
    rtt_weight: float,
    retx_weight: float,
    timeout_weight: float,
) -> float | None:
    if row is None:
        return None
    throughput = metric_float(row, "throughput_mbps")
    avg_rtt_ms = metric_float(row, "avg_rtt_ms")
    retx = metric_float(row, "retransmissions")
    timeouts = metric_float(row, "timeout_events")
    packets = metric_float(row, "packets")
    acked = metric_float(row, "acked")
    ack_ratio = acked / packets if packets > 0 else 0.0
    excess_rtt = max(0.0, avg_rtt_ms - target_rtt_ms)
    return (
        throughput
        + 0.02 * ack_ratio
        - rtt_weight * excess_rtt
        - retx_weight * retx
        - timeout_weight * timeouts
    )


def seed_values_for_state(trend: str, loss_flag: str) -> dict[str, float]:
    if loss_flag == "loss":
        return {"0": 0.0, "1": -1.0, "2": 1.5}
    if trend == "rtt_up":
        return {"0": 0.8, "1": 0.2, "2": 0.0}
    if trend == "rtt_down":
        return {"0": 0.2, "1": 1.6, "2": -0.8}
    return {"0": 0.4, "1": 1.0, "2": -0.5}


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
    score: float | None = None,
) -> str:
    score_part = f" score={score:.4f}" if score is not None else ""
    if row is None:
        return f"[TRAIN] round={round_index}/{rounds} epsilon={epsilon:.3f} metrics=missing{score_part} ckpt={checkpoint or '-'}"
    return (
        "[TRAIN] round={round_no}/{rounds} epsilon={epsilon:.3f} "
        "acked={acked}/{packets} duration={duration}s throughput={throughput}Mbps "
        "avg_rtt={avg_rtt}ms srtt={srtt}ms retx={retx} fast={fast} timeout={timeout} "
        "score={score_part} ckpt={checkpoint}".format(
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
            score_part=f"{score:.4f}" if score is not None else "?",
            checkpoint=checkpoint or "-",
        )
    )


def format_eval_metrics(
    round_index: int,
    rounds: int,
    row: dict[str, str] | None,
    score: float | None = None,
) -> str:
    if row is None:
        return f"[EVAL] round={round_index}/{rounds} metrics=missing"
    return (
        "[EVAL] round={round_no}/{rounds} acked={acked}/{packets} "
        "duration={duration}s throughput={throughput}Mbps avg_rtt={avg_rtt}ms "
        "retx={retx} timeout={timeout} score={score_part}".format(
            round_no=round_index,
            rounds=rounds,
            acked=row.get("acked", "?"),
            packets=row.get("packets", "?"),
            duration=row.get("duration_s", "?"),
            throughput=row.get("throughput_mbps", "?"),
            avg_rtt=row.get("avg_rtt_ms", "?"),
            retx=row.get("retransmissions", "?"),
            timeout=row.get("timeout_events", "?"),
            score_part=f"{score:.4f}" if score is not None else "?",
        )
    )


def tqdm_postfix(row: dict[str, str] | None, checkpoint: Path | None, score: float | None) -> dict[str, str]:
    if row is None:
        return {"metrics": "missing", "ckpt": checkpoint.name if checkpoint else "-"}
    return {
        "acked": f"{row.get('acked', '?')}/{row.get('packets', '?')}",
        "mbps": row.get("throughput_mbps", "?"),
        "rtt_ms": row.get("avg_rtt_ms", "?"),
        "retx": row.get("retransmissions", "?"),
        "score": f"{score:.4f}" if score is not None else "?",
        "ckpt": checkpoint.name if checkpoint else "-",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-round Q-Learning training runner")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use a faster training preset for quick iteration",
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--packets", type=int, default=300)
    parser.add_argument("--receiver-port", type=int, default=9201)
    parser.add_argument("--sender-port", type=int, default=9200)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--max-window", type=int, default=64)
    parser.add_argument("--rto", type=float, default=0.22)
    parser.add_argument("--q-table", default="artifacts/models/active/q_table.json")
    parser.add_argument(
        "--reset-q-table",
        dest="reset_q_table",
        action="store_true",
        default=True,
        help="backup and re-seed the Q-table before training (default)",
    )
    parser.add_argument(
        "--continue-q-table",
        dest="reset_q_table",
        action="store_false",
        help="continue training from the existing Q-table instead of re-seeding it",
    )
    parser.add_argument("--qtable-backup-dir", default="artifacts/models/backups")
    parser.add_argument("--q-alpha", type=float, default=0.003)
    parser.add_argument("--q-gamma", type=float, default=0.90)
    parser.add_argument("--q-epsilon", type=float, default=0.12)
    parser.add_argument("--reward-throughput-weight", type=float, default=1.2)
    parser.add_argument("--reward-timeout-weight", type=float, default=16.0)
    parser.add_argument("--reward-retx-weight", type=float, default=3.0)
    parser.add_argument("--reward-rtt-weight", type=float, default=0.010)
    parser.add_argument(
        "--reward-target-rtt-ms",
        type=float,
        default=30.0,
        help="only penalize RTT above this target in the Q-Learning reward",
    )
    parser.add_argument("--epsilon-decay", type=float, default=0.975)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--loss-rate", type=float, default=0.02)
    parser.add_argument("--delay-ms", type=float, default=10.0)
    parser.add_argument("--jitter-ms", type=float, default=3.0)
    parser.add_argument("--link-service-delay-ms", type=float, default=10.0)
    parser.add_argument("--link-queue-capacity", type=int, default=20)
    parser.add_argument("--link-bandwidth-drop-after-packets", type=int, default=0)
    parser.add_argument("--link-bandwidth-drop-factor", type=float, default=0.5)
    parser.add_argument("--metrics-file", default="artifacts/training/qlearning_metrics.csv")
    parser.add_argument("--history-file", default="artifacts/training/qlearning_history.csv")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/qlearning")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--summary-file", default="artifacts/training/qlearning_summary.csv")
    parser.add_argument("--best-table", default="artifacts/models/candidates/q_table_best.json")
    parser.add_argument("--score-target-rtt-ms", type=float, default=30.0)
    parser.add_argument("--score-rtt-weight", type=float, default=0.003)
    parser.add_argument("--score-retx-weight", type=float, default=0.004)
    parser.add_argument("--score-timeout-weight", type=float, default=0.030)
    parser.add_argument(
        "--selection-window",
        type=int,
        default=10,
        help="select the best Q-table by the rolling mean score over this many rounds",
    )
    parser.add_argument("--eval-rounds", type=int, default=5)
    parser.add_argument("--eval-packets", type=int, default=0, help="defaults to --packets when unset or 0")
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--install-best",
        dest="install_best",
        action="store_true",
        default=True,
        help="copy the best training Q-table back to --q-table before evaluation (default)",
    )
    parser.add_argument(
        "--no-install-best",
        dest="install_best",
        action="store_false",
        help="keep the active Q-table at the final training state",
    )
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
        "rounds": ("--rounds", 20),
        "packets": ("--packets", 60),
        "loss_rate": ("--loss-rate", 0.04),
        "delay_ms": ("--delay-ms", 5.0),
        "jitter_ms": ("--jitter-ms", 2.0),
        "rto": ("--rto", 0.10),
        "link_service_delay_ms": ("--link-service-delay-ms", 2.0),
        "checkpoint_every": ("--checkpoint-every", 5),
        "eval_rounds": ("--eval-rounds", 2),
    }
    for name, (option, value) in replacements.items():
        if option not in provided_options:
            setattr(args, name, value)


def build_sender_command(
    root: Path,
    args: argparse.Namespace,
    q_table: Path,
    metrics_file: Path,
    history_file: Path,
    local_port: int,
    start_seq: int,
    packets: int,
    epsilon: float,
    q_eval: bool,
) -> list[str]:
    sender_cmd = [
        sys.executable,
        str(root / "sender.py"),
        "--target-port",
        str(args.receiver_port),
        "--local-port",
        str(local_port),
        "--packets",
        str(packets),
        "--start-seq",
        str(start_seq),
        "--window-size",
        str(args.window_size),
        "--rto",
        str(args.rto),
        "--link-service-delay-ms",
        str(args.link_service_delay_ms),
        "--link-queue-capacity",
        str(args.link_queue_capacity),
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
        "--reward-target-rtt-ms",
        str(args.reward_target_rtt_ms),
        "--epsilon",
        str(epsilon),
        "--qtable-file",
        str(q_table),
        "--q-seed",
        str(args.seed + local_port + start_seq),
        "--metrics-file",
        str(metrics_file),
        "--history-file",
        str(history_file),
    ]
    if args.link_bandwidth_drop_after_packets > 0:
        sender_cmd.extend(
            [
                "--link-bandwidth-drop-after-packets",
                str(args.link_bandwidth_drop_after_packets),
                "--link-bandwidth-drop-factor",
                str(args.link_bandwidth_drop_factor),
            ]
        )
    if q_eval:
        sender_cmd.append("--q-eval")
    if args.quiet_sender or not args.verbose_sender:
        sender_cmd.append("--quiet")
    return sender_cmd


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_fast_preset(args, collect_cli_options(sys.argv[1:]))
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.packets <= 0:
        raise SystemExit("--packets must be positive")
    if args.eval_packets <= 0:
        args.eval_packets = args.packets
    if args.eval_rounds < 0:
        raise SystemExit("--eval-rounds must be non-negative")
    if args.eval_packets <= 0:
        raise SystemExit("--eval-packets must be positive")
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
    if not 0.0 < args.link_bandwidth_drop_factor <= 1.0:
        raise SystemExit("--link-bandwidth-drop-factor must be in (0, 1]")
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
    if args.reward_target_rtt_ms < 0:
        raise SystemExit("--reward-target-rtt-ms must be non-negative")
    if args.score_target_rtt_ms < 0:
        raise SystemExit("--score-target-rtt-ms must be non-negative")
    if args.score_rtt_weight < 0:
        raise SystemExit("--score-rtt-weight must be non-negative")
    if args.score_retx_weight < 0:
        raise SystemExit("--score-retx-weight must be non-negative")
    if args.score_timeout_weight < 0:
        raise SystemExit("--score-timeout-weight must be non-negative")
    if args.selection_window <= 0:
        raise SystemExit("--selection-window must be positive")
    if not 0.0 <= args.q_epsilon <= 1.0:
        raise SystemExit("--q-epsilon must be in [0, 1]")
    if not 0.0 <= args.min_epsilon <= 1.0:
        raise SystemExit("--min-epsilon must be in [0, 1]")
    if not 0.0 <= args.eval_epsilon <= 1.0:
        raise SystemExit("--eval-epsilon must be in [0, 1]")

    root = Path(__file__).resolve().parent
    q_table = (root / args.q_table).resolve()
    metrics_file = (root / args.metrics_file).resolve()
    history_file = (root / args.history_file).resolve()
    checkpoint_dir = (root / args.checkpoint_dir).resolve()
    summary_file = (root / args.summary_file).resolve()
    qtable_backup_dir = (root / args.qtable_backup_dir).resolve()
    best_table = (root / args.best_table).resolve()
    if args.reset_q_table or not q_table.exists():
        backup = seed_q_table(q_table, qtable_backup_dir)
        if backup is None:
            print(f"[TRAIN] seeded new Q-table at {q_table}", flush=True)
        else:
            print(f"[TRAIN] backed up old Q-table to {backup}", flush=True)
            print(f"[TRAIN] seeded fresh Q-table at {q_table}", flush=True)
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

    best_score = None
    best_round = None
    best_row = None
    recent_scores: list[float] = []

    try:
        for round_index in range(args.rounds):
            epsilon = max(
                args.min_epsilon,
                args.q_epsilon * (args.epsilon_decay ** round_index),
            )
            local_port = args.sender_port + round_index
            if local_port == args.receiver_port:
                local_port += args.rounds + 1
            sender_cmd = build_sender_command(
                root=root,
                args=args,
                q_table=q_table,
                metrics_file=metrics_file,
                history_file=history_file,
                local_port=local_port,
                start_seq=round_index * args.packets,
                packets=args.packets,
                epsilon=epsilon,
                q_eval=False,
            )
            completed = subprocess.run(sender_cmd, cwd=root)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
            latest_metrics = read_latest_metrics(metrics_file)
            current_score = score_metrics(
                latest_metrics,
                target_rtt_ms=args.score_target_rtt_ms,
                rtt_weight=args.score_rtt_weight,
                retx_weight=args.score_retx_weight,
                timeout_weight=args.score_timeout_weight,
            )
            if current_score is not None:
                recent_scores.append(current_score)
                if len(recent_scores) > args.selection_window:
                    recent_scores.pop(0)
            selection_score = (
                sum(recent_scores) / len(recent_scores)
                if len(recent_scores) == args.selection_window
                else None
            )
            is_best = False
            if selection_score is not None and (best_score is None or selection_score > best_score):
                copied = copy_q_table(q_table, best_table)
                if copied is not None:
                    best_score = selection_score
                    best_round = round_index + 1
                    best_row = latest_metrics
                    is_best = True
            checkpoint = None
            if (round_index + 1) % args.checkpoint_every == 0:
                checkpoint = save_checkpoint(
                    source=q_table,
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
                score=current_score,
            )
            if is_best:
                message = (
                    f"{message} rolling_score={selection_score:.4f} "
                    f"best={best_table}"
                )
            if progress is not None:
                progress.set_postfix(tqdm_postfix(latest_metrics, checkpoint, current_score), refresh=False)
                progress.update(1)
                progress.write(message)
            else:
                print(message, flush=True)

        if best_score is not None and best_round is not None:
            if args.install_best:
                installed = copy_q_table(best_table, q_table)
                if installed is not None:
                    print(
                        f"[TRAIN] installed best Q-table from round={best_round} score={best_score:.4f} to {q_table}",
                        flush=True,
                    )
            else:
                print(
                    f"[TRAIN] best Q-table kept at {best_table} from round={best_round} score={best_score:.4f}",
                    flush=True,
                )
            if best_row is not None:
                print(
                    "[TRAIN] best metrics throughput={throughput}Mbps avg_rtt={rtt}ms retx={retx} timeout={timeout}".format(
                        throughput=best_row.get("throughput_mbps", "?"),
                        rtt=best_row.get("avg_rtt_ms", "?"),
                        retx=best_row.get("retransmissions", "?"),
                        timeout=best_row.get("timeout_events", "?"),
                    ),
                    flush=True,
                )
        else:
            print("[TRAIN] no best Q-table was selected; keeping the final training table", flush=True)

        if args.eval_rounds > 0:
            eval_table = q_table if args.install_best or best_score is None else best_table
            print(
                f"[EVAL] running {args.eval_rounds} greedy evaluation round(s) with {eval_table}",
                flush=True,
            )
            for eval_index in range(args.eval_rounds):
                local_port = args.sender_port + args.rounds + eval_index + 1000
                if local_port == args.receiver_port:
                    local_port += args.eval_rounds + 1
                start_seq = args.rounds * args.packets + eval_index * args.eval_packets
                sender_cmd = build_sender_command(
                    root=root,
                    args=args,
                    q_table=eval_table,
                    metrics_file=metrics_file,
                    history_file=history_file,
                    local_port=local_port,
                    start_seq=start_seq,
                    packets=args.eval_packets,
                    epsilon=args.eval_epsilon,
                    q_eval=True,
                )
                completed = subprocess.run(sender_cmd, cwd=root)
                if completed.returncode != 0:
                    raise SystemExit(completed.returncode)
                latest_metrics = read_latest_metrics(metrics_file)
                eval_score = score_metrics(
                    latest_metrics,
                    target_rtt_ms=args.score_target_rtt_ms,
                    rtt_weight=args.score_rtt_weight,
                    retx_weight=args.score_retx_weight,
                    timeout_weight=args.score_timeout_weight,
                )
                append_summary(
                    path=summary_file,
                    round_index=args.rounds + eval_index + 1,
                    epsilon=args.eval_epsilon,
                    row=latest_metrics,
                    checkpoint=None,
                )
                print(
                    format_eval_metrics(
                        round_index=eval_index + 1,
                        rounds=args.eval_rounds,
                        row=latest_metrics,
                        score=eval_score,
                    ),
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
