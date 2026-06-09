"""Run Q-Learning 100 rounds + DQN 100 rounds sequentially."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"[TRAIN-ALL] {desc}")
    print(f"[TRAIN-ALL] CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[TRAIN-ALL] {desc} FAILED with code {result.returncode}", flush=True)
        raise SystemExit(result.returncode)
    print(f"[TRAIN-ALL] {desc} DONE\n", flush=True)


def main() -> None:
    print("[TRAIN-ALL] Starting Q-Learning 100 rounds + DQN 100 rounds", flush=True)

    # Step 1: Q-Learning 100 rounds
    run(
        [
            sys.executable, str(ROOT / "train_q_learning.py"),
            "--rounds", "100",
            "--packets", "120",
            "--q-epsilon", "0.3",
            "--epsilon-decay", "0.95",
            "--min-epsilon", "0.05",
            "--q-table", "artifacts/models/active/q_table.json",
            "--metrics-file", "artifacts/training/qlearning_metrics.csv",
            "--history-file", "artifacts/training/qlearning_history.csv",
            "--quiet-sender",
        ],
        "Q-Learning 100 rounds",
    )

    # Step 2: DQN 100 rounds
    run(
        [
            sys.executable, str(ROOT / "train_dqn.py"),
            "--rounds", "100",
            "--packets", "240",
            "--reset-dqn-model",
            "--epsilon", "0.45",
            "--epsilon-decay", "0.97",
            "--min-epsilon", "0.03",
            "--dqn-model", "artifacts/models/active/dqn_model.pt",
            "--dqn-lr", "0.0007",
            "--dqn-batch-size", "32",
            "--dqn-replay-capacity", "4096",
            "--reward-throughput-weight", "2.4",
            "--reward-timeout-weight", "7.0",
            "--reward-retx-weight", "1.2",
            "--reward-rtt-weight", "0.006",
            "--metrics-file", "artifacts/training/dqn_metrics.csv",
            "--history-file", "artifacts/training/dqn_history.csv",
            "--quiet-sender",
        ],
        "DQN 100 rounds",
    )

    print("\n[TRAIN-ALL] All done!", flush=True)


if __name__ == "__main__":
    main()
