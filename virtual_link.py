import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class LinkStats:
    enqueued_packets: int = 0
    sent_packets: int = 0
    dropped_packets: int = 0


class VirtualLink:
    def __init__(self, sock, bandwidth_pps: float, queue_size: int, logger):
        self.sock = sock
        self.bandwidth_pps = bandwidth_pps
        self.departure_interval = 1.0 / bandwidth_pps
        self.queue = queue.Queue(maxsize=queue_size)
        self.logger = logger
        self.stats = LinkStats()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._next_departure = None
        self._worker.start()

    def enqueue(self, packet: bytes, address: tuple[str, int], seq: int) -> bool:
        if self._stop_event.is_set():
            return False
        try:
            self.queue.put_nowait((packet, address, seq))
            self.stats.enqueued_packets += 1
            self.logger(
                "LINK",
                f"enqueue seq={seq} qsize={self.queue.qsize()}/{self.queue.maxsize}",
            )
            return True
        except queue.Full:
            self.stats.dropped_packets += 1
            self.logger(
                "LINK",
                f"drop seq={seq} qsize={self.queue.qsize()}/{self.queue.maxsize}",
            )
            return False

    def close(self) -> None:
        self._stop_event.set()
        try:
            self.queue.put_nowait((None, None, None))
        except queue.Full:
            pass
        self._worker.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet, address, seq = self.queue.get(timeout=0.1)
            except queue.Empty:
                self._next_departure = None
                continue

            if packet is None:
                break

            now = time.monotonic()
            if self._next_departure is None or self._next_departure < now:
                self._next_departure = now
            sleep_for = self._next_departure - now
            if sleep_for > 0:
                time.sleep(sleep_for)

            self.sock.sendto(packet, address)
            self.stats.sent_packets += 1
            self.logger(
                "LINK",
                f"send seq={seq} departed qsize={self.queue.qsize()}/{self.queue.maxsize}",
            )
            self._next_departure += self.departure_interval
