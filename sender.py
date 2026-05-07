import argparse
import socket
import threading
import time
from dataclasses import dataclass

from protocol import PAYLOAD_SIZE, build_payload, pack_data_packet, unpack_ack


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
        self.finished = False

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self.local)
        sock.settimeout(0.2)

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
            sock.close()

        duration = max(time.monotonic() - started_at, 1e-6)
        throughput_mbps = (self.acked_packets * PAYLOAD_SIZE * 8.0) / duration / 1_000_000.0
        self._log(
            "DONE",
            "acked={acked}/{total} retransmissions={retx} duration={duration:.3f}s "
            "throughput={throughput:.3f}Mbps".format(
                acked=self.acked_packets,
                total=self.total_packets,
                retx=self.retransmissions,
                duration=duration,
                throughput=throughput_mbps,
            ),
        )

    def _send_new_packet(self, sock: socket.socket, seq: int) -> None:
        payload = build_payload(seq)
        now = time.monotonic()
        timestamp = time.time()
        packet = pack_data_packet(seq, timestamp, payload)
        sock.sendto(packet, self.target)
        self.unacked[seq] = PacketState(
            payload=payload,
            last_send_monotonic=now,
            wire_timestamp=timestamp,
        )
        self._log("SEND", f"seq={seq} inflight={len(self.unacked)} window={self.window_size}")

    def _retransmit_packet(self, sock: socket.socket, seq: int, state: PacketState) -> None:
        now = time.monotonic()
        timestamp = time.time()
        packet = pack_data_packet(seq, timestamp, state.payload)
        sock.sendto(packet, self.target)
        state.last_send_monotonic = now
        state.wire_timestamp = timestamp
        state.transmissions += 1
        self.retransmissions += 1
        self._log(
            "RTO",
            f"seq={seq} transmissions={state.transmissions} retx_total={self.retransmissions}",
        )

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
                newly_acked = sorted(seq for seq in self.unacked if seq <= ack_number)
                for seq in newly_acked:
                    del self.unacked[seq]
                self.acked_packets += len(newly_acked)

                if newly_acked:
                    self._log(
                        "ACK",
                        "cumulative_ack={ack} newly_acked={count} range={start}-{end} "
                        "inflight={inflight}".format(
                            ack=ack_number,
                            count=len(newly_acked),
                            start=newly_acked[0],
                            end=newly_acked[-1],
                            inflight=len(self.unacked),
                        ),
                    )
                else:
                    self._log("ACK", f"duplicate_or_old cumulative_ack={ack_number}")

    def _timer_worker(self, sock: socket.socket) -> None:
        while not self.stop_event.is_set():
            now = time.monotonic()
            with self.lock:
                for seq, state in list(self.unacked.items()):
                    if now - state.last_send_monotonic >= self.rto:
                        self._retransmit_packet(sock, seq, state)
            time.sleep(min(self.rto / 2.0, 0.05))

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
    )
    sender.run()


if __name__ == "__main__":
    main()
