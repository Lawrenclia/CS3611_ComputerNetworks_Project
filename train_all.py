"""Run Q-Learning 100 rounds + DQN 100 rounds sequentially."""
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
            "--q-table", "q_table.json",
            "--metrics-file", "metrics.csv",
            "--history-file", "history.csv",
            "--quiet-sender",
        ],
        "Q-Learning 100 rounds",
    )

    # Step 2: DQN 100 rounds
    run(
        [
            sys.executable, str(ROOT / "train_dqn.py"),
            "--rounds", "100",
            "--packets", "120",
            "--epsilon", "0.5",
            "--epsilon-decay", "0.96",
            "--min-epsilon", "0.05",
            "--dqn-model", "dqn_model.pt",
            "--dqn-lr", "0.0005",
            "--dqn-batch-size", "32",
            "--dqn-replay-capacity", "2048",
            "--link-bandwidth-drop-after-packets", "60",
            "--link-bandwidth-drop-factor", "0.5",
            "--metrics-file", "metrics.csv",
            "--history-file", "history.csv",
            "--quiet-sender",
        ],
        "DQN 100 rounds",
    )

    print("\n[TRAIN-ALL] All done!", flush=True)


if __name__ == "__main__":
    main()
