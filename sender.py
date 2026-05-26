from __future__ import annotations

import argparse
import csv
import json
import random
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from protocol import PAYLOAD_SIZE, build_payload, pack_data_packet, unpack_ack
from virtual_link import VirtualFunnelLink


@dataclass
class PacketState:
    payload: bytes
    last_send_monotonic: float
    wire_timestamp: float
    transmissions: int = 1


class CongestionController:
    def __init__(
        self,
        mode: str,
        initial_cwnd: float,
        max_cwnd: float,
        epsilon: float,
        alpha: float,
        gamma: float,
        qtable_file: str | None,
        verbose: bool,
    ) -> None:
        self.mode = mode
        self.cwnd = max(1.0, float(initial_cwnd))
        self.max_cwnd = max(1.0, float(max_cwnd))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.qtable_file = Path(qtable_file) if qtable_file else None
        self.verbose = verbose

        self.q_table = [[0.0, 0.0, 0.0] for _ in range(6)]
        self.last_state = 0
        self.last_action = 0
        self.last_srtt = None
        self.interval_acked = 0
        self.interval_losses = 0
        self.interval_rtt_sum = 0.0
        self.interval_rtt_count = 0
        self._load_q_table()

    def window_limit(self) -> int:
        return max(1, int(self.cwnd))

    def on_ack(self, newly_acked: int, latest_rtt: float | None) -> None:
        if newly_acked <= 0:
            return
        self.interval_acked += newly_acked
        if latest_rtt is not None:
            self.interval_rtt_sum += latest_rtt * newly_acked
            self.interval_rtt_count += newly_acked

        if self.mode == "aimd":
            self.cwnd = min(self.max_cwnd, self.cwnd + newly_acked / self.cwnd)

    def on_loss(self) -> None:
        self.interval_losses += 1
        if self.mode == "aimd":
            self.cwnd = max(1.0, self.cwnd / 2.0)

    def maybe_step_qlearning(self, srtt: float | None) -> tuple[int, int, float] | None:
        if self.mode != "qlearning":
            self._reset_interval()
            return None

        state = self._state_from_interval(srtt)
        reward = self._reward()
        old = self.q_table[self.last_state][self.last_action]
        best_next = max(self.q_table[state])
        self.q_table[self.last_state][self.last_action] = old + self.alpha * (
            reward + self.gamma * best_next - old
        )

        action = self._choose_action(state)
        self._apply_action(action)
        self.last_state = state
        self.last_action = action
        self._reset_interval()
        self.last_srtt = srtt
        return state, action, reward

    def save(self) -> None:
        if self.mode != "qlearning" or self.qtable_file is None:
            return
        try:
            self.qtable_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"q_table": self.q_table, "epsilon": self.epsilon}
            self.qtable_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            if self.verbose:
                print(f"[SENDER][QLEARN] failed to save q-table to {self.qtable_file}: {exc}", flush=True)
    def _choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(3)
        row = self.q_table[state]
        return max(range(3), key=lambda i: row[i])

    def _apply_action(self, action: int) -> None:
        if action == 1:
            self.cwnd = min(self.max_cwnd, self.cwnd + 1.0)
        elif action == 2:
            self.cwnd = max(1.0, self.cwnd / 2.0)

    def _state_from_interval(self, srtt: float | None) -> int:
        if srtt is None or self.last_srtt is None:
            trend = 1
        else:
            delta = srtt - self.last_srtt
            threshold = max(self.last_srtt * 0.05, 0.001)
            if delta > threshold:
                trend = 0
            elif delta < -threshold:
                trend = 1
            else:
                trend = 2
        loss_flag = 1 if self.interval_losses > 0 else 0
        return trend * 2 + loss_flag

    def _reward(self) -> float:
        avg_rtt = (
            self.interval_rtt_sum / self.interval_rtt_count
            if self.interval_rtt_count
            else 0.0
        )
        return self.interval_acked - 20.0 * avg_rtt - 2.0 * self.interval_losses

    def _reset_interval(self) -> None:
        self.interval_acked = 0
        self.interval_losses = 0
        self.interval_rtt_sum = 0.0
        self.interval_rtt_count = 0

    def _load_q_table(self) -> None:
        if self.mode != "qlearning" or self.qtable_file is None or not self.qtable_file.exists():
            return
        try:
            data = json.loads(self.qtable_file.read_text(encoding="utf-8"))
            rows = data.get("q_table", [])
            if len(rows) == 6 and all(len(row) == 3 for row in rows):
                self.q_table = [[float(value) for value in row] for row in rows]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            if self.verbose:
                print("[SENDER][QLEARN] ignore invalid q-table file", flush=True)


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
        max_cwnd: float = 100.0,
        epsilon: float = 0.10,
        q_alpha: float = 0.30,
        q_gamma: float = 0.85,
        qtable_file: str | None = "q_table.json",
        metrics_file: str | None = None,
        history_file: str | None = None,
        plot_file: str | None = None,
    ) -> None:
        self.target = (target_host, target_port)
        self.local = (local_host, local_port)
        self.total_packets = total_packets
        self.window_size = max(1, window_size)
        self.cc_mode = cc_mode
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
        self.controller = CongestionController(
            mode=cc_mode,
            initial_cwnd=self.window_size,
            max_cwnd=max_cwnd,
            epsilon=epsilon,
            alpha=q_alpha,
            gamma=q_gamma,
            qtable_file=qtable_file,
            verbose=verbose,
        )
        self.metrics_file = Path(metrics_file) if metrics_file else None
        self.history_file = Path(history_file) if history_file else None
        self.plot_file = Path(plot_file) if plot_file else None
        self.cwnd_history: list[tuple[float, float]] = []
        self.rtt_history: list[tuple[float, float]] = []
        self.started_at = None
        self.timeout_events = 0
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{cc_mode}"

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
        self.started_at = started_at
        next_control_at = started_at + self.rto
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    if self.acked_packets >= self.total_packets and not self.unacked:
                        self.finished = True
                        break

                    now = time.monotonic()
                    if now >= next_control_at:
                        self._control_step_locked(now)
                        interval = max(self.srtt or self.rto, 0.05)
                        next_control_at = now + interval

                    while (
                        self.next_seq < self.end_seq
                        and len(self.unacked) < self.controller.window_limit()
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

        self.controller.save()
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
        self._write_metrics(duration, throughput_mbps)
        self._write_history()
        self._plot_results(duration, throughput_mbps)
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
        self._record_cwnd(now)
        self._log(
            "SEND",
            "seq={seq} inflight={inflight} cwnd={cwnd:.2f} mode={mode}".format(
                seq=seq,
                inflight=len(self.unacked),
                cwnd=self.controller.cwnd,
                mode=self.cc_mode,
            ),
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
        self.controller.on_loss()
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
            if self.started_at is not None:
                self.rtt_history.append((time.monotonic() - self.started_at, latest_rtt))
        self.acked_packets += len(newly_acked)
        self.controller.on_ack(len(newly_acked), latest_rtt)
        self._record_cwnd(time.monotonic())

        if newly_acked:
            self.last_ack_number = ack_number
            self.duplicate_ack_count = 0
            self._log(
                "ACK",
                "cumulative_ack={ack} newly_acked={count} range={start}-{end} "
                "rtt_ms={rtt:.2f} srtt_ms={srtt:.2f} inflight={inflight} cwnd={cwnd:.2f}".format(
                    ack=ack_number,
                    count=len(newly_acked),
                    start=newly_acked[0],
                    end=newly_acked[-1],
                    rtt=(latest_rtt or 0.0) * 1000.0,
                    srtt=(self.srtt or 0.0) * 1000.0,
                    inflight=len(self.unacked),
                    cwnd=self.controller.cwnd,
                ),
            )
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
                        self.timeout_events += 1
                        self._retransmit_packet(sock, seq, state, reason="RTO")
            time.sleep(min(self.rto / 2.0, 0.05))

    def _control_step_locked(self, now: float) -> None:
        result = self.controller.maybe_step_qlearning(self.srtt)
        self._record_cwnd(now)
        if result is None:
            return
        state, action, reward = result
        self._log(
            "QLEARN",
            "state={state} action={action} reward={reward:.3f} cwnd={cwnd:.2f}".format(
                state=state,
                action=action,
                reward=reward,
                cwnd=self.controller.cwnd,
            ),
        )

    def _record_cwnd(self, now: float) -> None:
        if self.started_at is None:
            return
        elapsed = now - self.started_at
        if self.cwnd_history and elapsed - self.cwnd_history[-1][0] < 0.01:
            self.cwnd_history[-1] = (elapsed, self.controller.cwnd)
        else:
            self.cwnd_history.append((elapsed, self.controller.cwnd))

    def _write_metrics(self, duration: float, throughput_mbps: float) -> None:
        if self.metrics_file is None:
            return
        exists = self.metrics_file.exists()
        avg_rtt = (
            sum(rtt for _, rtt in self.rtt_history) / len(self.rtt_history)
            if self.rtt_history
            else 0.0
        )
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
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
                ],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": self.run_id,
                    "mode": self.cc_mode,
                    "packets": self.total_packets,
                    "acked": self.acked_packets,
                    "duration_s": f"{duration:.6f}",
                    "throughput_mbps": f"{throughput_mbps:.6f}",
                    "avg_rtt_ms": f"{avg_rtt * 1000.0:.3f}",
                    "srtt_ms": f"{(self.srtt or 0.0) * 1000.0:.3f}",
                    "retransmissions": self.retransmissions,
                    "fast_retransmissions": self.fast_retransmissions,
                    "timeout_events": self.timeout_events,
                }
            )

    def _write_history(self) -> None:
        if self.history_file is None:
            return
        exists = self.history_file.exists()
        rtt_by_time = {round(elapsed, 2): rtt for elapsed, rtt in self.rtt_history}
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with self.history_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["run_id", "mode", "time_s", "cwnd", "rtt_ms"],
            )
            if not exists:
                writer.writeheader()
            for elapsed, cwnd in self.cwnd_history:
                rtt = rtt_by_time.get(round(elapsed, 2))
                writer.writerow(
                    {
                        "run_id": self.run_id,
                        "mode": self.cc_mode,
                        "time_s": f"{elapsed:.6f}",
                        "cwnd": f"{cwnd:.6f}",
                        "rtt_ms": "" if rtt is None else f"{rtt * 1000.0:.3f}",
                    }
                )

    def _plot_results(self, duration: float, throughput_mbps: float) -> None:
        if self.plot_file is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._log("PLOT", "matplotlib not installed, skip plot")
            return

        times = [item[0] for item in self.cwnd_history] or [0.0, duration]
        cwnds = [item[1] for item in self.cwnd_history] or [self.controller.cwnd, self.controller.cwnd]
        rtt_times = [item[0] for item in self.rtt_history]
        rtts = [item[1] * 1000.0 for item in self.rtt_history]

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
        axes[0].plot(times, cwnds, label=f"{self.cc_mode} cwnd", linewidth=1.8)
        axes[0].set_title("CWND over time")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("CWND (packets)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        if rtts:
            axes[1].plot(rtt_times, rtts, label="RTT", color="tab:orange", linewidth=1.4)
        axes[1].bar(
            [0],
            [throughput_mbps],
            width=0.35,
            label=f"Throughput {throughput_mbps:.3f} Mbps",
            color="tab:green",
            alpha=0.45,
        )
        axes[1].set_title("RTT samples and throughput")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("RTT (ms) / Mbps")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        self.plot_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.plot_file, dpi=140)
        plt.close(fig)
        self._log("PLOT", f"saved {self.plot_file}")

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
    parser.add_argument(
        "--cc-mode",
        choices=("fixed", "aimd", "qlearning"),
        default="fixed",
        help="congestion control mode: fixed window, AIMD baseline, or Q-Learning",
    )
    parser.add_argument("--max-cwnd", type=float, default=100.0)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--q-alpha", type=float, default=0.30)
    parser.add_argument("--q-gamma", type=float, default=0.85)
    parser.add_argument("--qtable-file", default="q_table.json")
    parser.add_argument("--metrics-file", default="metrics.csv")
    parser.add_argument("--history-file", default="history.csv")
    parser.add_argument("--plot-file", default=None)
    parser.add_argument("--rto", type=float, default=0.20)
    parser.add_argument("--link-queue-capacity", type=int, default=20)
    parser.add_argument("--link-service-delay-ms", type=float, default=10.0)
    parser.add_argument("--disable-virtual-link", action="store_true")
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
    if args.max_cwnd < 1:
        raise SystemExit("--max-cwnd must be at least 1")
    if not 0 <= args.epsilon <= 1:
        raise SystemExit("--epsilon must be between 0 and 1")
    if not 0 < args.q_alpha <= 1:
        raise SystemExit("--q-alpha must be in (0, 1]")
    if not 0 <= args.q_gamma <= 1:
        raise SystemExit("--q-gamma must be between 0 and 1")
    if args.link_queue_capacity <= 0:
        raise SystemExit("--link-queue-capacity must be positive")
    if args.link_service_delay_ms < 0:
        raise SystemExit("--link-service-delay-ms must be non-negative")

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
        cc_mode=args.cc_mode,
        max_cwnd=args.max_cwnd,
        epsilon=args.epsilon,
        q_alpha=args.q_alpha,
        q_gamma=args.q_gamma,
        qtable_file=args.qtable_file,
        metrics_file=args.metrics_file,
        history_file=args.history_file,
        plot_file=args.plot_file,
    )
    sender.run()


if __name__ == "__main__":
    main()
