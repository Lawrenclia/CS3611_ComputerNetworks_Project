import argparse
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-round DQN congestion-control trainer")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--packets", type=int, default=160)
    parser.add_argument("--receiver-port", type=int, default=9301)
    parser.add_argument("--sender-port", type=int, default=9300)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--max-window", type=int, default=80)
    parser.add_argument("--rto", type=float, default=0.20)
    parser.add_argument("--dqn-model", default="dqn_model.pt")
    parser.add_argument("--dqn-lr", type=float, default=0.001)
    parser.add_argument("--dqn-batch-size", type=int, default=16)
    parser.add_argument("--dqn-replay-capacity", type=int, default=1024)
    parser.add_argument("--dqn-target-update", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--epsilon-decay", type=float, default=0.85)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--loss-rate", type=float, default=0.06)
    parser.add_argument("--delay-ms", type=float, default=20.0)
    parser.add_argument("--jitter-ms", type=float, default=10.0)
    parser.add_argument("--link-bandwidth-drop-after-packets", type=int, default=80)
    parser.add_argument("--link-bandwidth-drop-factor", type=float, default=0.5)
    parser.add_argument("--metrics-file", default="metrics.csv")
    parser.add_argument("--history-file", default="history.csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet-sender", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.packets <= 0:
        raise SystemExit("--packets must be positive")
    if not 0.0 <= args.loss_rate <= 1.0:
        raise SystemExit("--loss-rate must be in [0, 1]")

    root = Path(__file__).resolve().parent
    dqn_model = str((root / args.dqn_model).resolve())
    metrics_file = str((root / args.metrics_file).resolve())
    history_file = str((root / args.history_file).resolve())
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

    try:
        for round_index in range(args.rounds):
            epsilon = max(args.min_epsilon, args.epsilon * (args.epsilon_decay ** round_index))
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
                "dqn",
                "--max-cwnd",
                str(args.max_window),
                "--epsilon",
                str(epsilon),
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
            if args.quiet_sender:
                sender_cmd.append("--quiet")

            print(
                "[DQN-TRAIN] round={round_no}/{rounds} start_seq={start_seq} "
                "epsilon={epsilon:.3f} model={model}".format(
                    round_no=round_index + 1,
                    rounds=args.rounds,
                    start_seq=round_index * args.packets,
                    epsilon=epsilon,
                    model=dqn_model,
                ),
                flush=True,
            )
            completed = subprocess.run(sender_cmd, cwd=root)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
    finally:
        receiver.terminate()
        try:
            receiver.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            receiver.kill()


if __name__ == "__main__":
    main()
