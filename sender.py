import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from congestion import AIMDController, QLearningController
from protocol import PAYLOAD_SIZE, build_payload, pack_data_packet, unpack_ack
from virtual_link import VirtualLink
from visualize import (
    save_comparison_plot,
    save_comparison_svg,
    save_cwnd_csv,
    save_metrics_csv,
)


@dataclass
class PacketState:
    payload: bytes
    last_send_monotonic: float
    wire_timestamp: float
    transmissions: int = 1
    acked: bool = False


class ReliableSender:
    def __init__(
        self,
        target_host: str,
        target_port: int,
        local_host: str,
        local_port: int,
        bandwidth_pps: float,
        queue_size: int,
        rto: float,
        verbose: bool = True,
    ) -> None:
        self.target = (target_host, target_port)
        self.local = (local_host, local_port)
        self.bandwidth_pps = bandwidth_pps
        self.queue_size = queue_size
        self.base_rto = rto
        self.verbose = verbose

    def run_episode(
        self,
        controller,
        total_packets: int,
        start_seq: int,
        training: bool,
    ) -> dict:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self.local)
        sock.settimeout(0.2)
        lock = threading.Lock()
        stop_event = threading.Event()
        link = VirtualLink(sock, self.bandwidth_pps, self.queue_size, self._log)

        controller.reset(training=training)
        unacked: dict[int, PacketState] = {}
        acked_packets = 0
        retransmissions = 0
        fast_retransmissions = 0
        timeout_events = 0
        srtt = None
        cwnd_history = [(0.0, controller.cwnd)]
        interval_history = []
        start_time = time.monotonic()
        next_seq = start_seq
        finished = False
        last_ack_number = None
        duplicate_ack_count = 0

        def record_cwnd() -> None:
            elapsed = time.monotonic() - start_time
            if not cwnd_history or cwnd_history[-1][1] != controller.cwnd:
                cwnd_history.append((elapsed, controller.cwnd))

        def retransmit_packet(seq: int, now: float, reason: str) -> bool:
            nonlocal retransmissions, timeout_events, fast_retransmissions
            state = unacked.get(seq)
            if state is None or state.acked:
                return False

            wire_timestamp = time.time()
            packet = pack_data_packet(seq, wire_timestamp, state.payload)
            link.enqueue(packet, self.target, seq)
            state.last_send_monotonic = now
            state.wire_timestamp = wire_timestamp
            state.transmissions += 1
            retransmissions += 1
            if reason == "fast":
                fast_retransmissions += 1
                self._log(
                    "FAST",
                    f"seq={seq} fast_retransmissions={fast_retransmissions} cwnd={controller.cwnd:.2f}",
                )
            else:
                timeout_events += 1
                self._log(
                    "RTO",
                    f"seq={seq} timeout retransmissions={retransmissions} cwnd={controller.cwnd:.2f}",
                )
            return True

        def ack_worker() -> None:
            nonlocal acked_packets, srtt, finished, last_ack_number, duplicate_ack_count
            while not stop_event.is_set():
                try:
                    packet, address = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError:
                    break

                try:
                    ack_number = unpack_ack(packet)
                except ValueError:
                    self._log("ACK", f"ignore non-ack packet from {address}")
                    continue

                now = time.monotonic()
                wall_now = time.time()
                with lock:
                    newly_acked = sorted(
                        seq
                        for seq, state in unacked.items()
                        if not state.acked and seq <= ack_number
                    )

                    if not newly_acked:
                        if last_ack_number == ack_number:
                            duplicate_ack_count += 1
                        else:
                            last_ack_number = ack_number
                            duplicate_ack_count = 1
                        self._log(
                            "ACK",
                            f"duplicate cumulative_ack={ack_number} dup_count={duplicate_ack_count} from={address}",
                        )
                        if duplicate_ack_count >= 3:
                            missing_seq = ack_number + 1
                            if missing_seq in unacked and not unacked[missing_seq].acked:
                                if hasattr(controller, "record_loss"):
                                    controller.record_loss()
                                controller.on_loss()
                                record_cwnd()
                                retransmit_packet(missing_seq, now, reason="fast")
                            duplicate_ack_count = 0
                        continue

                    last_ack_number = ack_number
                    duplicate_ack_count = 0
                    latest_rtt = None
                    for seq in newly_acked:
                        state = unacked[seq]
                        state.acked = True
                        latest_rtt = wall_now - state.wire_timestamp
                        if hasattr(controller, "record_ack"):
                            controller.record_ack(PAYLOAD_SIZE, latest_rtt)
                        controller.on_ack()
                        del unacked[seq]

                    acked_packets += len(newly_acked)
                    if latest_rtt is not None:
                        srtt = latest_rtt if srtt is None else (0.875 * srtt + 0.125 * latest_rtt)
                    record_cwnd()
                    self._log(
                        "ACK",
                        "cumulative_ack={ack} newly_acked={count} range={start}-{end} "
                        "rtt_ms={rtt:.2f} inflight={inflight} cwnd={cwnd:.2f}".format(
                            ack=ack_number,
                            count=len(newly_acked),
                            start=newly_acked[0],
                            end=newly_acked[-1],
                            rtt=(latest_rtt or 0.0) * 1000,
                            inflight=len(unacked),
                            cwnd=controller.cwnd,
                        ),
                    )
                    if acked_packets >= total_packets:
                        finished = True

        thread = threading.Thread(target=ack_worker, daemon=True)
        thread.start()

        while not finished:
            with lock:
                window_limit = controller.window_limit()
                while next_seq < start_seq + total_packets and len(unacked) < window_limit:
                    now = time.monotonic()
                    seq = next_seq
                    payload = build_payload(seq)
                    wire_timestamp = time.time()
                    packet = pack_data_packet(seq, wire_timestamp, payload)
                    link.enqueue(packet, self.target, seq)
                    unacked[seq] = PacketState(
                        payload=payload,
                        last_send_monotonic=now,
                        wire_timestamp=wire_timestamp,
                    )
                    self._log(
                        "SEND",
                        f"seq={seq} inflight={len(unacked)} cwnd={controller.cwnd:.2f} window={window_limit}",
                    )
                    next_seq += 1

                now = time.monotonic()
                dynamic_rto = max(self.base_rto, (srtt or self.base_rto) * 2.0)
                timed_out = [
                    seq
                    for seq, state in unacked.items()
                    if not state.acked and now - state.last_send_monotonic >= dynamic_rto
                ]
                if timed_out:
                    controller.on_loss()
                    record_cwnd()
                for seq in timed_out:
                    if hasattr(controller, "record_loss"):
                        controller.record_loss()
                    retransmit_packet(seq, now, reason="timeout")

                snapshot = controller.maybe_step(now, srtt)
                if snapshot is not None:
                    interval_history.append(snapshot)
                    record_cwnd()
                    self._log(
                        "QL",
                        "state={state} action={action} reward={reward:.3f} "
                        "avg_rtt_ms={rtt:.2f} throughput_mbps={throughput:.3f}".format(
                            state=snapshot.state,
                            action=snapshot.action,
                            reward=snapshot.reward,
                            rtt=snapshot.avg_rtt_ms,
                            throughput=snapshot.throughput_mbps,
                        ),
                    )

                if acked_packets >= total_packets and not unacked:
                    finished = True

            time.sleep(0.005)

        stop_event.set()
        link.close()
        sock.close()
        thread.join(timeout=1.0)

        duration = time.monotonic() - start_time
        cwnd_history.append((duration, controller.cwnd))
        avg_rtt_ms = (
            sum(item.avg_rtt_ms for item in interval_history) / len(interval_history)
            if interval_history
            else (srtt or 0.0) * 1000.0
        )
        metrics = {
            "algorithm": controller.name,
            "duration_s": duration,
            "throughput_mbps": (acked_packets * PAYLOAD_SIZE * 8.0) / duration / 1_000_000.0,
            "avg_rtt_ms": avg_rtt_ms,
            "retransmissions": retransmissions,
            "fast_retransmissions": fast_retransmissions,
            "timeouts": timeout_events,
            "link_drops": link.stats.dropped_packets,
            "srtt_ms": (srtt or 0.0) * 1000.0,
            "acked_packets": acked_packets,
            "cwnd_history": cwnd_history,
        }
        self._log("DONE", json.dumps(metrics, ensure_ascii=True))
        return metrics

    def _log(self, category: str, message: str) -> None:
        if not self.verbose:
            return
        now = time.strftime("%H:%M:%S")
        print(f"[{now}][SENDER][{category}] {message}", flush=True)


def compare_algorithms(args) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sender = ReliableSender(
        target_host=args.target_host,
        target_port=args.target_port,
        local_host=args.local_host,
        local_port=args.local_port,
        bandwidth_pps=args.bandwidth_pps,
        queue_size=args.queue_size,
        rto=args.rto,
        verbose=not args.quiet,
    )

    aimd = AIMDController()
    aimd_metrics = sender.run_episode(
        controller=aimd,
        total_packets=args.packets,
        start_seq=0,
        training=False,
    )

    learner = QLearningController(
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
    )
    for episode in range(args.train_episodes):
        if not args.quiet:
            print(
                f"[TRAIN] q-learning episode {episode + 1}/{args.train_episodes} epsilon={learner.epsilon:.3f}",
                flush=True,
            )
        sender.run_episode(
            controller=learner,
            total_packets=args.packets,
            start_seq=(episode + 1) * args.packets,
            training=True,
        )
        learner.finish_episode()

    eval_start_seq = (args.train_episodes + 1) * args.packets
    q_metrics = sender.run_episode(
        controller=learner,
        total_packets=args.packets,
        start_seq=eval_start_seq,
        training=False,
    )

    learner.save(str(results_dir / "q_table.json"))
    save_cwnd_csv(results_dir / "aimd_cwnd.csv", "aimd", aimd_metrics["cwnd_history"])
    save_cwnd_csv(results_dir / "q_learning_cwnd.csv", "q_learning", q_metrics["cwnd_history"])
    save_metrics_csv(results_dir / "metrics.csv", [aimd_metrics, q_metrics])
    save_comparison_svg(
        results_dir / "comparison.svg",
        aimd_metrics["cwnd_history"],
        q_metrics["cwnd_history"],
        aimd_metrics,
        q_metrics,
    )
    save_comparison_plot(
        results_dir / "comparison.png",
        aimd_metrics["cwnd_history"],
        q_metrics["cwnd_history"],
        aimd_metrics,
        q_metrics,
    )
    summary = {
        "aimd": {key: value for key, value in aimd_metrics.items() if key != "cwnd_history"},
        "q_learning": {key: value for key, value in q_metrics.items() if key != "cwnd_history"},
        "results_dir": str(results_dir.resolve()),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UDP reliable sender with AIMD and Q-learning")
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=9001)
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--local-port", type=int, default=9000)
    parser.add_argument("--packets", type=int, default=120)
    parser.add_argument("--bandwidth-pps", type=float, default=100.0)
    parser.add_argument("--queue-size", type=int, default=20)
    parser.add_argument("--rto", type=float, default=0.20)
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--epsilon-decay", type=float, default=0.92)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    compare_algorithms(args)


if __name__ == "__main__":
    main()
