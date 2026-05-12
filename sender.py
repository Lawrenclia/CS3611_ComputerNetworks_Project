import argparse
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

from congestion_control import (
    ACTION_NAMES,
    AIMDCongestionController,
    ControlDecision,
    FixedWindowController,
    QLearningCongestionController,
)
from protocol import PAYLOAD_SIZE, build_payload, pack_data_packet, unpack_ack
from virtual_link import VirtualFunnelLink


@dataclass
class PacketState:
    payload: bytes
    last_send_monotonic: float
    wire_timestamp: float
    transmissions: int = 1


class ReliableSender:
    def __init__(
        self,
        target_host: str,
        target_port: int,
        local_host: str,
        local_port: int,
        total_packets: int,
        window_size: int,
        rto: float,
        verbose: bool = True,
        start_seq: int = 0,
        use_virtual_link: bool = True,
        link_queue_capacity: int = 20,
        link_service_delay_ms: float = 10.0,
        cc_mode: str = "fixed",
        min_window: int = 1,
        max_window: int = 64,
        q_alpha: float = 0.30,
        q_gamma: float = 0.85,
        q_epsilon: float = 0.10,
        q_table: Optional[str] = None,
        q_seed: Optional[int] = None,
        reward_alpha: float = 1.0,
        reward_beta: float = 0.02,
        reward_gamma: float = 3.0,
        rtt_trend_threshold: float = 0.10,
        min_cycle_seconds: float = 0.001,
    ) -> None:
        self.target = (target_host, target_port)
        self.local = (local_host, local_port)
        self.total_packets = total_packets
        self.window_size = max(1, window_size)
        self.rto = rto
        self.verbose = verbose
        self.start_seq = start_seq
        self.end_seq = start_seq + total_packets

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.unacked: dict[int, PacketState] = {}
        self.next_seq = start_seq
        self.acked_packets = 0
        self.retransmissions = 0
        self.fast_retransmissions = 0
        self.last_ack_number = None
        self.duplicate_ack_count = 0
        self.srtt = None
        self.latest_rtt = None
        self.finished = False
        self.virtual_link = None
        self.use_virtual_link = use_virtual_link
        self.link_queue_capacity = link_queue_capacity
        self.link_service_delay_ms = link_service_delay_ms
        self.last_congestion_signal_monotonic = 0.0

        if cc_mode == "fixed":
            self.controller = FixedWindowController(self.window_size)
        elif cc_mode == "aimd":
            self.controller = AIMDCongestionController(
                min_window=min_window,
                max_window=max_window,
            )
            self.window_size = self.controller.current_window
        elif cc_mode == "q-learning":
            self.controller = QLearningCongestionController(
                initial_window=self.window_size,
                min_window=min_window,
                max_window=max(max_window, min_window, self.window_size),
                alpha=q_alpha,
                gamma=q_gamma,
                epsilon=q_epsilon,
                q_table_path=q_table,
                seed=q_seed,
                reward_throughput_weight=reward_alpha,
                reward_rtt_weight=reward_beta,
                reward_loss_weight=reward_gamma,
                rtt_trend_threshold=rtt_trend_threshold,
                min_cycle_seconds=min_cycle_seconds,
            )
            self.window_size = self.controller.current_window
        else:
            raise ValueError(f"unsupported congestion controller: {cc_mode}")

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self.local)
        sock.settimeout(0.2)
        if self.use_virtual_link:
            self.virtual_link = VirtualFunnelLink(
                sock,
                service_delay_ms=self.link_service_delay_ms,
                queue_capacity=self.link_queue_capacity,
                verbose=self.verbose,
            )

        ack_thread = threading.Thread(target=self._ack_worker, args=(sock,), daemon=True)
        timer_thread = threading.Thread(target=self._timer_worker, args=(sock,), daemon=True)
        ack_thread.start()
        timer_thread.start()

        started_at = time.monotonic()
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    if self.acked_packets >= self.total_packets and not self.unacked:
                        self.finished = True
                        break

                    while (
                        self.next_seq < self.end_seq
                        and len(self.unacked) < self.window_size
                    ):
                        self._send_new_packet(sock, self.next_seq)
                        self.next_seq += 1

                time.sleep(0.005)
        finally:
            self.stop_event.set()
            ack_thread.join(timeout=1.0)
            timer_thread.join(timeout=1.0)
            if self.virtual_link is not None:
                self.virtual_link.close()
            sock.close()

        duration = max(time.monotonic() - started_at, 1e-6)
        throughput_mbps = (self.acked_packets * PAYLOAD_SIZE * 8.0) / duration / 1_000_000.0
        self._log(
            "DONE",
            "acked={acked}/{total} retransmissions={retx} fast_retransmissions={fast} "
            "srtt_ms={srtt:.2f} duration={duration:.3f}s throughput={throughput:.3f}Mbps".format(
                acked=self.acked_packets,
                total=self.total_packets,
                retx=self.retransmissions,
                fast=self.fast_retransmissions,
                srtt=(self.srtt or 0.0) * 1000.0,
                duration=duration,
                throughput=throughput_mbps,
            ),
        )
        if self.virtual_link is not None:
            stats = self.virtual_link.snapshot()
            self._log(
                "VLINK",
                "enqueued={enqueued} forwarded={forwarded} dropped={dropped} "
                "max_depth={depth}/{capacity} service_delay_ms={delay:.1f}".format(
                    enqueued=stats.enqueued_packets,
                    forwarded=stats.forwarded_packets,
                    dropped=stats.dropped_packets,
                    depth=stats.max_queue_depth,
                    capacity=self.link_queue_capacity,
                    delay=self.link_service_delay_ms,
                ),
            )
        self._log("CC", self.controller.summary())
        self.controller.close()

    def _send_new_packet(self, sock: socket.socket, seq: int) -> None:
        payload = build_payload(seq)
        now = time.monotonic()
        timestamp = time.time()
        packet = pack_data_packet(seq, timestamp, payload)
        self._send_packet(sock, packet)
        self.unacked[seq] = PacketState(
            payload=payload,
            last_send_monotonic=now,
            wire_timestamp=timestamp,
        )
        self._log(
            "SEND",
            f"seq={seq} inflight={len(self.unacked)} window={self.window_size}",
        )

    def _retransmit_packet(
        self,
        sock: socket.socket,
        seq: int,
        state: PacketState,
        reason: str = "RTO",
    ) -> None:
        now = time.monotonic()
        timestamp = time.time()
        packet = pack_data_packet(seq, timestamp, state.payload)
        self._send_packet(sock, packet)
        state.last_send_monotonic = now
        state.wire_timestamp = timestamp
        state.transmissions += 1
        self.retransmissions += 1
        if reason == "FAST":
            self.fast_retransmissions += 1
        self._log(
            reason,
            "seq={seq} transmissions={tx} retx_total={retx} fast_total={fast}".format(
                seq=seq,
                tx=state.transmissions,
                retx=self.retransmissions,
                fast=self.fast_retransmissions,
            ),
        )
        if reason in {"RTO", "FAST"}:
            self._notify_congestion_locked(reason, len(self.unacked), now)

    def _send_packet(self, sock: socket.socket, packet: bytes) -> None:
        if self.virtual_link is None:
            sock.sendto(packet, self.target)
            return
        self.virtual_link.sendto(packet, self.target)

    def _ack_worker(self, sock: socket.socket) -> None:
        while not self.stop_event.is_set():
            try:
                packet, address = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                ack_number = unpack_ack(packet)
            except ValueError:
                self._log("ACK", f"ignore invalid ack from {address}")
                continue

            with self.lock:
                self._handle_ack_locked(sock, ack_number)

    def _handle_ack_locked(self, sock: socket.socket, ack_number: int) -> None:
        newly_acked = sorted(seq for seq in self.unacked if seq <= ack_number)
        latest_rtt = None
        wall_now = time.time()
        for seq in newly_acked:
            state = self.unacked.pop(seq)
            latest_rtt = wall_now - state.wire_timestamp
            self.latest_rtt = latest_rtt
            self.srtt = latest_rtt if self.srtt is None else (0.875 * self.srtt + 0.125 * latest_rtt)
        self.acked_packets += len(newly_acked)

        if newly_acked:
            self.last_ack_number = ack_number
            self.duplicate_ack_count = 0
            self._log(
                "ACK",
                "cumulative_ack={ack} newly_acked={count} range={start}-{end} "
                "rtt_ms={rtt:.2f} srtt_ms={srtt:.2f} inflight={inflight}".format(
                    ack=ack_number,
                    count=len(newly_acked),
                    start=newly_acked[0],
                    end=newly_acked[-1],
                    rtt=(latest_rtt or 0.0) * 1000.0,
                    srtt=(self.srtt or 0.0) * 1000.0,
                    inflight=len(self.unacked),
                ),
            )
            decision = self.controller.observe_ack(
                newly_acked=len(newly_acked),
                srtt=self.srtt,
                latest_rtt=latest_rtt,
                inflight=len(self.unacked),
            )
            self._apply_control_decision_locked(decision)
            return

        if self.last_ack_number == ack_number:
            self.duplicate_ack_count += 1
        else:
            self.last_ack_number = ack_number
            self.duplicate_ack_count = 1

        self._log(
            "ACK",
            f"duplicate cumulative_ack={ack_number} dup_count={self.duplicate_ack_count}",
        )
        if self.duplicate_ack_count < 3:
            return

        missing_seq = ack_number + 1
        state = self.unacked.get(missing_seq)
        if state is None:
            self._log("FAST", f"skip missing_seq={missing_seq} not_in_unacked")
        else:
            self._retransmit_packet(sock, missing_seq, state, reason="FAST")
        self.duplicate_ack_count = 0

    def _timer_worker(self, sock: socket.socket) -> None:
        while not self.stop_event.is_set():
            now = time.monotonic()
            with self.lock:
                for seq, state in list(self.unacked.items()):
                    if now - state.last_send_monotonic >= self.rto:
                        self._retransmit_packet(sock, seq, state, reason="RTO")
            time.sleep(min(self.rto / 2.0, 0.05))

    def _notify_congestion_locked(
        self,
        reason: str,
        inflight: int,
        now: float,
    ) -> None:
        guard_interval = max(self.srtt or self.rto, 0.05)
        if now - self.last_congestion_signal_monotonic < guard_interval:
            return
        self.last_congestion_signal_monotonic = now

        decision = self.controller.observe_loss(
            reason=reason,
            srtt=self.srtt,
            latest_rtt=self.latest_rtt,
            inflight=inflight,
        )
        self._apply_control_decision_locked(decision)

    def _apply_control_decision_locked(
        self,
        decision: Optional[ControlDecision],
    ) -> None:
        if decision is None:
            return
        self.window_size = decision.new_window
        self._log(
            "CC",
            "event={event} state={state} action={action}({action_name}) "
            "window={old}->{new} reward={reward:.3f} q={q:.3f} "
            "throughput={throughput:.2f}pkt/s avg_rtt_ms={rtt:.2f} "
            "loss_count={loss_count} loss_ewma={loss:.3f}".format(
                event=decision.event,
                state=decision.state,
                action=decision.action,
                action_name=ACTION_NAMES.get(decision.action, "unknown"),
                old=decision.old_window,
                new=decision.new_window,
                reward=decision.reward,
                q=decision.q_value,
                throughput=decision.throughput,
                rtt=decision.avg_rtt * 1000.0,
                loss_count=decision.loss_count,
                loss=decision.loss_ewma,
            ),
        )

    def _log(self, category: str, message: str) -> None:
        if not self.verbose:
            return
        now = time.strftime("%H:%M:%S")
        print(f"[{now}][SENDER][{category}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UDP reliable sender")
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=9001)
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--local-port", type=int, default=9000)
    parser.add_argument("--packets", type=int, default=40)
    parser.add_argument("--start-seq", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--rto", type=float, default=0.20)
    parser.add_argument("--link-queue-capacity", type=int, default=20)
    parser.add_argument("--link-service-delay-ms", type=float, default=10.0)
    parser.add_argument("--disable-virtual-link", action="store_true")
    parser.add_argument(
        "--cc",
        choices=("fixed", "aimd", "q-learning"),
        default="fixed",
        help="congestion controller: fixed window, AIMD/Reno baseline, or Q-Learning adaptive window",
    )
    parser.add_argument("--min-window", type=int, default=1)
    parser.add_argument("--max-window", type=int, default=64)
    parser.add_argument("--q-alpha", type=float, default=0.30)
    parser.add_argument("--q-gamma", type=float, default=0.85)
    parser.add_argument("--q-epsilon", type=float, default=0.10)
    parser.add_argument(
        "--q-table",
        default=None,
        help="optional JSON file used to load and save the learned Q table",
    )
    parser.add_argument("--q-seed", type=int, default=None)
    parser.add_argument("--reward-alpha", type=float, default=1.0)
    parser.add_argument("--reward-beta", type=float, default=0.02)
    parser.add_argument("--reward-gamma", type=float, default=3.0)
    parser.add_argument("--rtt-trend-threshold", type=float, default=0.10)
    parser.add_argument("--min-cycle-seconds", type=float, default=0.001)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.packets < 0:
        raise SystemExit("--packets must be non-negative")
    if args.start_seq < 0:
        raise SystemExit("--start-seq must be non-negative")
    if args.start_seq + max(args.packets - 1, 0) > 2_147_483_647:
        raise SystemExit("--start-seq + --packets exceeds signed ACK range")
    if args.rto <= 0:
        raise SystemExit("--rto must be positive")
    if args.link_queue_capacity <= 0:
        raise SystemExit("--link-queue-capacity must be positive")
    if args.link_service_delay_ms < 0:
        raise SystemExit("--link-service-delay-ms must be non-negative")
    if args.min_window <= 0:
        raise SystemExit("--min-window must be positive")
    if args.max_window < args.min_window:
        raise SystemExit("--max-window must be >= --min-window")
    if not 0.0 <= args.q_alpha <= 1.0:
        raise SystemExit("--q-alpha must be in [0, 1]")
    if not 0.0 <= args.q_gamma <= 1.0:
        raise SystemExit("--q-gamma must be in [0, 1]")
    if not 0.0 <= args.q_epsilon <= 1.0:
        raise SystemExit("--q-epsilon must be in [0, 1]")
    if args.reward_alpha < 0 or args.reward_beta < 0 or args.reward_gamma < 0:
        raise SystemExit("--reward-alpha, --reward-beta and --reward-gamma must be non-negative")
    if args.rtt_trend_threshold < 0:
        raise SystemExit("--rtt-trend-threshold must be non-negative")
    if args.min_cycle_seconds <= 0:
        raise SystemExit("--min-cycle-seconds must be positive")

    sender = ReliableSender(
        target_host=args.target_host,
        target_port=args.target_port,
        local_host=args.local_host,
        local_port=args.local_port,
        total_packets=args.packets,
        window_size=args.window_size,
        rto=args.rto,
        verbose=not args.quiet,
        start_seq=args.start_seq,
        use_virtual_link=not args.disable_virtual_link,
        link_queue_capacity=args.link_queue_capacity,
        link_service_delay_ms=args.link_service_delay_ms,
        cc_mode=args.cc,
        min_window=args.min_window,
        max_window=args.max_window,
        q_alpha=args.q_alpha,
        q_gamma=args.q_gamma,
        q_epsilon=args.q_epsilon,
        q_table=args.q_table,
        q_seed=args.q_seed,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_gamma=args.reward_gamma,
        rtt_trend_threshold=args.rtt_trend_threshold,
        min_cycle_seconds=args.min_cycle_seconds,
    )
    sender.run()


if __name__ == "__main__":
    main()
