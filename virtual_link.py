from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class VirtualLinkStats:
    enqueued_packets: int = 0
    forwarded_packets: int = 0
    dropped_packets: int = 0
    max_queue_depth: int = 0


class VirtualFunnelLink:
    def __init__(
        self,
        sock,
        service_delay_ms: float = 10.0,
        queue_capacity: int = 20,
        verbose: bool = True,
        label: str = "VLINK",
        bandwidth_drop_after_packets: int | None = None,
        bandwidth_drop_factor: float = 0.5,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if service_delay_ms < 0:
            raise ValueError("service_delay_ms must be non-negative")
        if bandwidth_drop_after_packets is not None and bandwidth_drop_after_packets <= 0:
            raise ValueError("bandwidth_drop_after_packets must be positive")
        if not 0 < bandwidth_drop_factor <= 1:
            raise ValueError("bandwidth_drop_factor must be in (0, 1]")

        self.sock = sock
        self.service_delay_ms = float(service_delay_ms)
        self.service_delay = self.service_delay_ms / 1000.0
        self.queue_capacity = queue_capacity
        self.verbose = verbose
        self.label = label
        self.bandwidth_drop_after_packets = bandwidth_drop_after_packets
        self.bandwidth_drop_factor = float(bandwidth_drop_factor)
        self._bandwidth_drop_applied = False

        self._queue: queue.Queue[tuple[bytes, tuple[str, int], float]] = queue.Queue(
            maxsize=queue_capacity
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stats = VirtualLinkStats()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def sendto(self, packet: bytes, address: tuple[str, int]) -> bool:
        if self._stop_event.is_set():
            raise RuntimeError("virtual link is closed")

        queued_at = time.monotonic()
        try:
            self._queue.put_nowait((packet, address, queued_at))
        except queue.Full:
            with self._lock:
                self._stats.dropped_packets += 1
                dropped_total = self._stats.dropped_packets
            self._log(
                "DROP",
                "queue_full capacity={capacity} dropped_total={dropped} dest={dest}".format(
                    capacity=self.queue_capacity,
                    dropped=dropped_total,
                    dest=f"{address[0]}:{address[1]}",
                ),
            )
            return False

        with self._lock:
            self._stats.enqueued_packets += 1
            depth = self._queue.qsize()
            if depth > self._stats.max_queue_depth:
                self._stats.max_queue_depth = depth

        return True

    def snapshot(self) -> VirtualLinkStats:
        with self._lock:
            return VirtualLinkStats(
                enqueued_packets=self._stats.enqueued_packets,
                forwarded_packets=self._stats.forwarded_packets,
                dropped_packets=self._stats.dropped_packets,
                max_queue_depth=self._stats.max_queue_depth,
            )

    def set_service_delay(self, delay_ms: float) -> None:
        """Externally set the service delay (e.g. to simulate bandwidth drop)."""
        with self._lock:
            self.service_delay_ms = float(delay_ms)
            self.service_delay = self.service_delay_ms / 1000.0

    def close(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        next_send_at = None
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break

            try:
                packet, address, queued_at = self._queue.get(timeout=0.05)
            except queue.Empty:
                next_send_at = None
                continue

            try:
                if next_send_at is not None:
                    sleep_for = next_send_at - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                send_started = time.monotonic()
                self.sock.sendto(packet, address)
                queue_delay_ms = (send_started - queued_at) * 1000.0

                with self._lock:
                    self._stats.forwarded_packets += 1
                    forwarded_total = self._stats.forwarded_packets

                self._maybe_apply_bandwidth_drop(forwarded_total)
                self._log(
                    "FORWARD",
                    "queued_ms={queued_ms:.2f} forwarded_total={forwarded_total} "
                    "depth={depth} dest={dest}".format(
                        queued_ms=queue_delay_ms,
                        forwarded_total=forwarded_total,
                        depth=self._queue.qsize(),
                        dest=f"{address[0]}:{address[1]}",
                    ),
                )
                next_send_at = send_started + self.service_delay
            except OSError as exc:
                if not self._stop_event.is_set():
                    self._log("ERROR", f"sendto failed: {exc}")
                break
            finally:
                self._queue.task_done()

    def _maybe_apply_bandwidth_drop(self, forwarded_total: int) -> None:
        if self.bandwidth_drop_after_packets is None or self._bandwidth_drop_applied:
            return
        if forwarded_total < self.bandwidth_drop_after_packets:
            return

        with self._lock:
            if self._bandwidth_drop_applied:
                return
            old_delay_ms = self.service_delay_ms
            self.service_delay_ms = self.service_delay_ms / self.bandwidth_drop_factor
            self.service_delay = self.service_delay_ms / 1000.0
            self._bandwidth_drop_applied = True

        self._log(
            "BANDWIDTH",
            "drop_after_forwarded={forwarded} factor={factor:.3f} "
            "service_delay_ms={old:.2f}->{new:.2f}".format(
                forwarded=forwarded_total,
                factor=self.bandwidth_drop_factor,
                old=old_delay_ms,
                new=self.service_delay_ms,
            ),
        )

    def _log(self, category: str, message: str) -> None:
        if not self.verbose:
            return
        now = time.strftime("%H:%M:%S")
        print(f"[{now}][{self.label}][{category}] {message}", flush=True)
