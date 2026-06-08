from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-stage Q-Learning curriculum trainer")
    parser.add_argument("--stage1-rounds", type=int, default=80)
    parser.add_argument("--stage2-rounds", type=int, default=100)
    parser.add_argument("--q-table", default="artifacts/models/candidates/q_table_curriculum.json")
    parser.add_argument("--output-table", default="artifacts/models/candidates/q_table_good.json")
    parser.add_argument("--install", action="store_true", help="copy --output-table to artifacts/models/active/q_table.json after training")
    parser.add_argument("--continue-existing", action="store_true", help="continue from the existing --q-table instead of resetting it")
    parser.add_argument("--packets-stage1", type=int, default=120)
    parser.add_argument("--packets-stage2", type=int, default=160)
    parser.add_argument("--receiver-port", type=int, default=9201)
    parser.add_argument("--sender-port", type=int, default=9200)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser


def run_stage(
    root: Path,
    run_dir: Path,
    q_table: str,
    stage: str,
    rounds: int,
    packets: int,
    receiver_port: int,
    sender_port: int,
    checkpoint_every: int,
    reset: bool,
) -> None:
    if rounds <= 0:
        raise SystemExit(f"--{stage}-rounds must be positive")
    command = [
        sys.executable,
        str(root / "train_q_learning.py"),
        "--fast",
        "--rounds",
        str(rounds),
        "--packets",
        str(packets),
        "--window-size",
        "8",
        "--max-window",
        "64",
        "--receiver-port",
        str(receiver_port),
        "--sender-port",
        str(sender_port),
        "--q-table",
        q_table,
        "--checkpoint-every",
        str(checkpoint_every),
        "--metrics-file",
        str(run_dir / f"{stage}_metrics.csv"),
        "--history-file",
        str(run_dir / f"{stage}_history.csv"),
        "--summary-file",
        str(run_dir / f"{stage}_summary.csv"),
        "--checkpoint-dir",
        str(root / "artifacts" / "checkpoints" / run_dir.name / stage),
    ]
    if stage == "stage1":
        command.extend(
            [
                "--q-epsilon",
                "0.45",
                "--epsilon-decay",
                "0.97",
                "--min-epsilon",
                "0.08",
                "--q-alpha",
                "0.15",
                "--loss-rate",
                "0.03",
                "--reward-timeout-weight",
                "6",
                "--reward-retx-weight",
                "1.5",
                "--reward-rtt-weight",
                "0.010",
            ]
        )
    else:
        command.extend(
            [
                "--q-epsilon",
                "0.20",
                "--epsilon-decay",
                "0.98",
                "--min-epsilon",
                "0.05",
                "--q-alpha",
                "0.10",
                "--loss-rate",
                "0.05",
                "--reward-timeout-weight",
                "7",
                "--reward-retx-weight",
                "1.6",
                "--reward-rtt-weight",
                "0.010",
            ]
        )
    if reset:
        command.append("--reset-q-table")

    print(f"[CURRICULUM] running {stage}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=root)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def backup_and_copy(source: Path, target: Path, backup_dir: Path) -> Path | None:
    backup = None
    if target.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{target.stem}_backup_{time.strftime('%Y%m%d-%H%M%S')}{target.suffix}"
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    return backup


def main() -> None:
    args = build_parser().parse_args()
    if args.stage1_rounds <= 0:
        raise SystemExit("--stage1-rounds must be positive")
    if args.stage2_rounds <= 0:
        raise SystemExit("--stage2-rounds must be positive")
    if args.packets_stage1 <= 0 or args.packets_stage2 <= 0:
        raise SystemExit("--packets-stage1 and --packets-stage2 must be positive")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")

    root = Path(__file__).resolve().parent
    run_dir = root / "artifacts" / "training" / f"q_curriculum_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_stage(
        root=root,
        run_dir=run_dir,
        q_table=args.q_table,
        stage="stage1",
        rounds=args.stage1_rounds,
        packets=args.packets_stage1,
        receiver_port=args.receiver_port,
        sender_port=args.sender_port,
        checkpoint_every=args.checkpoint_every,
        reset=not args.continue_existing,
    )
    run_stage(
        root=root,
        run_dir=run_dir,
        q_table=args.q_table,
        stage="stage2",
        rounds=args.stage2_rounds,
        packets=args.packets_stage2,
        receiver_port=args.receiver_port,
        sender_port=args.sender_port,
        checkpoint_every=args.checkpoint_every,
        reset=False,
    )

    trained_table = root / args.q_table
    output_table = root / args.output_table
    if not trained_table.exists():
        raise SystemExit(f"expected trained Q-table not found: {trained_table}")
    shutil.copy2(trained_table, output_table)
    print(f"[CURRICULUM] wrote {output_table}", flush=True)

    if args.install:
        backup = backup_and_copy(
            source=output_table,
            target=root / "artifacts" / "models" / "active" / "q_table.json",
            backup_dir=root / "artifacts" / "models" / "backups",
        )
        if backup is not None:
            print(f"[CURRICULUM] backed up previous active Q-table to {backup}", flush=True)
        print("[CURRICULUM] installed trained table to artifacts/models/active/q_table.json", flush=True)

    print(f"[CURRICULUM] artifacts: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
