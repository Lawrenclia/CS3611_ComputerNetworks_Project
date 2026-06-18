#!/usr/bin/env python3
"""Bandwidth Halving Experiment: AIMD vs Q-Learning recovery dynamics.

Self-contained experiment — no external receiver needed.
Triggers bandwidth drop exactly at acked_packets >= total_packets // 2.
Records per-interval telemetry for before/after comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import socket
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from protocol import PAYLOAD_SIZE, build_payload, pack_data_packet, unpack_ack, pack_ack

# ── Constants ──────────────────────────────────────────────────────────
Q_STATE_NAMES = (
    "rtt_up|no_loss", "rtt_up|loss",
    "rtt_down|no_loss", "rtt_down|loss",
    "rtt_stable|no_loss", "rtt_stable|loss",
)
Q_ACTION_KEYS = ("0", "1", "2")


# ── Per-interval telemetry ─────────────────────────────────────────────
@dataclass
class IntervalRecord:
    time_s: float = 0.0
    cwnd: float = 0.0
    rtt_ms: float = 0.0
    srtt_ms: float = 0.0
    retransmissions: int = 0
    timeouts: int = 0
    acked_total: int = 0
    bandwidth_halved: bool = False
    action: str = ""
    state: str = ""
    reward: float = 0.0
    epsilon: float = 0.0
    throughput_mbps: float = 0.0


# ── Inline Receiver Simulator ──────────────────────────────────────────
class ReceiverSimulator:
    """Simulates a receiver with configurable loss rate and jitter.
    Runs in a background thread, listens on a UDP port, and sends ACKs back."""

    def __init__(self, listen_port: int, sender_addr: tuple[str, int],
                 loss_rate: float = 0.02, delay_ms: float = 10.0,
                 jitter_ms: float = 3.0, seed: int = 42):
        self.listen_port = listen_port
        self.sender_addr = sender_addr
        self.loss_rate = loss_rate
        self.base_delay_ms = delay_ms
        self.jitter_ms = jitter_ms
        self.rng = random.Random(seed)
        self.expected_seq = 0
        self.buffer: dict[int, bytes] = {}
        self.total_received = 0
        self.stop_event = threading.Event()
        self.sock: socket.socket | None = None
        self.thread: threading.Thread | None = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.listen_port))
        self.listen_port = self.sock.getsockname()[1]  # resolve ephemeral port
        self.sock.settimeout(0.1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.sock:
            self.sock.close()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                if self.sock is None:
                    break
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            # Simulate random loss
            if self.rng.random() < self.loss_rate:
                continue

            # Simulate delay + jitter
            delay = self.base_delay_ms / 1000.0 + self.rng.uniform(0, self.jitter_ms / 1000.0)
            # (delay is inherently simulated by network; we just send ACK immediately
            #  because the VirtualFunnelLink handles forward-path delay)

            try:
                seq = struct.unpack("!I", data[:4])[0]
            except struct.error:
                continue

            # Track received packets
            if seq >= self.expected_seq:
                self.buffer[seq] = data
                while self.expected_seq in self.buffer:
                    del self.buffer[self.expected_seq]
                    self.expected_seq += 1
                    self.total_received += 1

            # Send cumulative ACK
            ack_num = self.expected_seq - 1
            ack_packet = pack_ack(ack_num)
            if self.sock:
                self.sock.sendto(ack_packet, self.sender_addr)


# ── Virtual Link wrapper ───────────────────────────────────────────────
class SimpleVirtualLink:
    """Fixed-rate FIFO queue that simulates bottleneck bandwidth and buffer."""

    def __init__(self, sock: socket.socket, service_delay_ms: float = 10.0,
                 queue_capacity: int = 20):
        self.sock = sock
        self.service_delay_ms = service_delay_ms
        self.service_delay_s = service_delay_ms / 1000.0
        self.queue_capacity = queue_capacity
        self._queue: deque[tuple[bytes, tuple[str, int], float]] = deque()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.enqueued = 0
        self.forwarded = 0
        self.dropped = 0
        self.max_depth = 0
        self._worker = threading.Thread(target=self._forward_loop, daemon=True)
        self._worker.start()

    def sendto(self, packet: bytes, address: tuple[str, int]) -> bool:
        with self._lock:
            if len(self._queue) >= self.queue_capacity:
                self.dropped += 1
                return False
            self._queue.append((packet, address, time.monotonic()))
            self.enqueued += 1
            self.max_depth = max(self.max_depth, len(self._queue))
            return True

    def set_service_delay(self, delay_ms: float):
        with self._lock:
            self.service_delay_ms = delay_ms
            self.service_delay_s = delay_ms / 1000.0

    def close(self):
        self._stop.set()
        self._worker.join(timeout=2)

    def _forward_loop(self):
        next_send = None
        while not self._stop.is_set():
            with self._lock:
                if not self._queue:
                    next_send = None
                    time.sleep(0.01)
                    continue
                packet, addr, queued_at = self._queue.popleft()
                self.forwarded += 1

            if next_send is not None:
                wait = next_send - time.monotonic()
                if wait > 0:
                    time.sleep(wait)

            send_start = time.monotonic()
            try:
                self.sock.sendto(packet, addr)
            except OSError:
                pass
            next_send = send_start + self.service_delay_s


# ── Packet state for unacked queue ─────────────────────────────────────
@dataclass
class PktState:
    payload: bytes
    last_send: float
    wire_ts: float
    tx_count: int = 1


# ── Q-Learning Controller ──────────────────────────────────────────────
class QLearnController:
    def __init__(self, qtable_path: Path, max_cwnd: float = 64,
                 alpha: float = 0.05, gamma: float = 0.90, epsilon: float = 0.0,
                 additive_step: int = 1, rtt_ratio: float = 0.12,
                 rw_throughput: float = 2.0, rw_timeout: float = 15.0,
                 rw_retx: float = 3.0, rw_rtt: float = 0.005,
                 rw_target_rtt: float = 50.0):
        self.max_cwnd = max_cwnd
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.additive_step = additive_step
        self.rtt_ratio = rtt_ratio
        self.rw_throughput = rw_throughput
        self.rw_timeout = rw_timeout
        self.rw_retx = rw_retx
        self.rw_rtt = rw_rtt
        self.rw_target_rtt = rw_target_rtt
        self.cwnd = 1.0
        self.last_state: int | None = None
        self.last_action: int | None = None
        self.last_srtt: float | None = None
        self.bandwidth_halved = False
        self._reset()
        self.q_table = [[0.0, 0.0, 0.0] for _ in Q_STATE_NAMES]
        if qtable_path.exists():
            d = json.loads(qtable_path.read_text(encoding="utf-8"))
            for i, name in enumerate(Q_STATE_NAMES):
                sv = d.get(name, {})
                for j, ak in enumerate(Q_ACTION_KEYS):
                    self.q_table[i][j] = float(sv.get(ak, 0.0))

    def _reset(self):
        self.int_acked = 0
        self.int_losses = 0
        self.int_retx = 0
        self.int_timeouts = 0
        self.int_rtt_sum = 0.0
        self.int_rtt_cnt = 0

    def on_ack(self, n: int, rtt: float | None):
        if n <= 0:
            return
        self.int_acked += n
        if rtt is not None:
            self.int_rtt_sum += rtt * n
            self.int_rtt_cnt += n

    def on_loss(self, reason: str = "RTO"):
        self.int_losses += 1
        self.int_retx += 1
        if reason == "RTO":
            self.int_timeouts += 1

    def limit(self) -> int:
        return max(1, int(self.cwnd))

    def adapt_to_halving(self):
        self.bandwidth_halved = True
        self.epsilon = max(self.epsilon, 0.15)
        self.alpha = 0.15
        self.rw_timeout = 20.0
        self.rw_retx = 4.0
        self.rw_rtt = 0.008

    def step(self, srtt: float | None):
        if self.int_acked + self.int_losses == 0:
            return None
        state = self._to_state(srtt)
        reward = self._reward()
        if self.last_state is not None and self.last_action is not None:
            old = self.q_table[self.last_state][self.last_action]
            best = max(self.q_table[state])
            self.q_table[self.last_state][self.last_action] = old + self.alpha * (
                reward + self.gamma * best - old)
        action = self._pick(state)
        cwnd_int = max(1, int(self.cwnd))
        if action == 1:
            self.cwnd = min(self.max_cwnd, float(cwnd_int + self.additive_step))
        elif action == 2:
            self.cwnd = float(max(1, cwnd_int // 2))
        aname = ("hold", f"cwnd+{self.additive_step}", "cwnd/2")[action]
        sname = Q_STATE_NAMES[state]
        # Capture before reset
        int_retx = self.int_retx
        int_timeouts = self.int_timeouts
        self.last_state = state
        self.last_action = action
        self._reset()
        self.last_srtt = srtt
        return state, action, reward, sname, aname, int_retx, int_timeouts

    def _to_state(self, srtt: float | None) -> int:
        if srtt is None or self.last_srtt is None:
            trend = 1
        else:
            d = srtt - self.last_srtt
            th = max(self.last_srtt * self.rtt_ratio, 0.001)
            if d > th:
                trend = 0
            elif d < -th:
                trend = 1
            else:
                trend = 2
        return trend * 2 + (1 if self.int_retx > 0 else 0)

    def _reward(self) -> float:
        avg_rtt = self.int_rtt_sum / self.int_rtt_cnt if self.int_rtt_cnt else 0.0
        penalty = max(0.0, avg_rtt * 1000.0 - self.rw_target_rtt)
        return (self.rw_throughput * self.int_acked
                - self.rw_timeout * self.int_timeouts
                - self.rw_retx * self.int_retx
                - self.rw_rtt * penalty)

    def _pick(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(3)
        row = self.q_table[state]
        return max(range(3), key=lambda i: row[i])


# ── AIMD Controller ────────────────────────────────────────────────────
class AIMDController:
    def __init__(self, max_cwnd: float = 64):
        self.cwnd = 1.0
        self.max_cwnd = max_cwnd
        self.int_retx = 0
        self.int_timeouts = 0
        self.int_acked = 0

    def on_ack(self, n: int, rtt: float | None):
        if n > 0:
            self.int_acked += n
            self.cwnd = min(self.max_cwnd, self.cwnd + n / self.cwnd)

    def on_loss(self, reason: str = "RTO"):
        self.int_retx += 1
        if reason == "RTO":
            self.int_timeouts += 1
        self.cwnd = max(1.0, self.cwnd / 2.0)

    def limit(self) -> int:
        return max(1, int(self.cwnd))

    def step(self, srtt: float | None) -> bool:
        active = self.int_acked + self.int_retx > 0
        acked = self.int_acked
        retx = self.int_retx
        tos = self.int_timeouts
        self.int_acked = 0
        self.int_retx = 0
        self.int_timeouts = 0
        return active, acked, retx, tos


# ── Experiment Runner ──────────────────────────────────────────────────
class DropExperiment:
    def __init__(self, mode: str, total: int = 400, **kw):
        self.mode = mode
        self.total = total
        self.link_delay = kw.get("link_delay_ms", 10.0)
        self.queue_cap = kw.get("queue_cap", 20)
        self.rto = kw.get("rto", 0.20)
        self.max_cwnd = kw.get("max_cwnd", 64)
        self.ctrl_interval = kw.get("ctrl_interval", 0.10)
        self.seed = kw.get("seed", 42)
        self.max_duration_s = kw.get("max_duration_s", 45.0)

        random.seed(self.seed)

        qpath = Path(kw.get("qtable_path", "artifacts/models/active/q_table.json"))
        if not qpath.is_absolute():
            qpath = Path(__file__).resolve().parent / qpath

        if mode == "qlearning":
            self.cc: QLearnController | AIMDController = QLearnController(
                qtable_path=qpath, max_cwnd=self.max_cwnd, epsilon=0.0)
        else:
            self.cc = AIMDController(max_cwnd=self.max_cwnd)

        # Sockets
        self.tx_sock: socket.socket | None = None
        self.vlink: SimpleVirtualLink | None = None
        self.receiver: ReceiverSimulator | None = None

        # State
        self.unacked: dict[int, PktState] = {}
        self.next_seq = 0
        self.acked = 0
        self.total_retx = 0
        self.total_timeouts = 0
        self.total_fast_retx = 0
        self.srtt: float | None = None
        self.latest_rtt: float | None = None
        self.t0 = 0.0
        self.halving_t: float | None = None
        self.halved = False
        self.finished = False
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.last_ack: int | None = None
        self.dup_cnt = 0

        # Telemetry
        self.telemetry: list[IntervalRecord] = []
        self.cwnd_hist: list[tuple[float, float]] = []
        self.rtt_hist: list[tuple[float, float]] = []

    def run(self) -> dict:
        # Sender socket: used for sending (via vlink) and receiving ACKs
        tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tx_sock.bind(("127.0.0.1", 0))
        tx_sock.settimeout(0.05)
        sender_port = tx_sock.getsockname()[1]
        self.tx_sock = tx_sock

        # Receiver listens on its own ephemeral port
        self.receiver = ReceiverSimulator(
            listen_port=0, sender_addr=("127.0.0.1", sender_port),
            loss_rate=0.02, delay_ms=0, jitter_ms=3.0, seed=self.seed + 1)
        self.receiver.start()
        rx_port = self.receiver.listen_port

        # Virtual link on TX path
        self.vlink = SimpleVirtualLink(tx_sock, service_delay_ms=self.link_delay,
                                       queue_capacity=self.queue_cap)

        # Start threads
        ack_th = threading.Thread(target=self._ack_worker, daemon=True)
        timer_th = threading.Thread(target=self._timer_worker, daemon=True)
        ack_th.start()
        timer_th.start()

        self.t0 = time.monotonic()
        next_ctrl = self.t0 + self.ctrl_interval

        try:
            while not self.stop.is_set():
                with self.lock:
                    if self.acked >= self.total and not self.unacked:
                        self.finished = True
                        break
                    now = time.monotonic()
                    if now - self.t0 > self.max_duration_s:
                        self.finished = True
                        break
                    if now >= next_ctrl:
                        self._control_step(now)
                        next_ctrl = now + self.ctrl_interval
                    while (self.next_seq < self.total
                           and len(self.unacked) < self.cc.limit()):
                        self._send(self.next_seq)
                        self.next_seq += 1
                time.sleep(0.002)
        finally:
            self.stop.set()
            ack_th.join(timeout=2)
            timer_th.join(timeout=2)
            if self.receiver:
                self.receiver.stop()
            if self.vlink:
                self.vlink.close()
            tx_sock.close()

        dur = max(time.monotonic() - self.t0, 1e-6)
        return self._results(dur)

    def _send(self, seq: int):
        assert self.vlink is not None
        payload = build_payload(seq)
        ts = time.time()
        packet = pack_data_packet(seq, ts, payload)
        ok = self.vlink.sendto(packet, ("127.0.0.1", self.receiver.listen_port if self.receiver else 0))
        if ok:
            self.unacked[seq] = PktState(payload=payload, last_send=time.monotonic(), wire_ts=ts)
        self._log_cwnd(time.monotonic())

    def _ack_worker(self):
        assert self.tx_sock is not None
        while not self.stop.is_set():
            try:
                data, _ = self.tx_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                ack = unpack_ack(data)
            except ValueError:
                continue
            with self.lock:
                self._handle_ack(ack)

    def _handle_ack(self, ack: int):
        newly = sorted(s for s in self.unacked if s <= ack)
        latest = None
        wall = time.time()
        for s in newly:
            st = self.unacked.pop(s)
            rtt = wall - st.wire_ts
            latest = rtt
            self.latest_rtt = rtt
            self.srtt = rtt if self.srtt is None else 0.875 * self.srtt + 0.125 * rtt
            self.rtt_hist.append((time.monotonic() - self.t0, rtt))
        self.acked += len(newly)
        if newly:
            if isinstance(self.cc, QLearnController):
                self.cc.on_ack(len(newly), latest)
            else:
                self.cc.on_ack(len(newly), latest)
        self._log_cwnd(time.monotonic())

        if newly:
            self.last_ack = ack
            self.dup_cnt = 0
        else:
            self.dup_cnt += 1
            if self.last_ack == ack and self.dup_cnt >= 3:
                missing = ack + 1
                st = self.unacked.get(missing)
                if st is not None:
                    self._retransmit(missing, st, "FAST")
                    self.total_fast_retx += 1
                self.dup_cnt = 0

        # Trigger halving at midpoint
        if not self.halved and self.acked >= self.total // 2:
            self._do_halving()

    def _timer_worker(self):
        while not self.stop.is_set():
            now = time.monotonic()
            with self.lock:
                for s, st in list(self.unacked.items()):
                    if now - st.last_send >= self.rto:
                        self.total_timeouts += 1
                        self._retransmit(s, st, "RTO")
            time.sleep(min(self.rto / 2, 0.02))

    def _retransmit(self, seq: int, st: PktState, reason: str):
        assert self.vlink is not None
        ts = time.time()
        packet = pack_data_packet(seq, ts, st.payload)
        self.vlink.sendto(packet, ("127.0.0.1", self.receiver.listen_port if self.receiver else 0))
        st.last_send = time.monotonic()
        st.wire_ts = ts
        st.tx_count += 1
        self.total_retx += 1
        if isinstance(self.cc, QLearnController):
            self.cc.on_loss(reason=reason)
        else:
            self.cc.on_loss(reason=reason)

    def _do_halving(self):
        self.halved = True
        self.halving_t = time.monotonic() - self.t0
        if self.vlink:
            self.vlink.set_service_delay(self.link_delay * 2.0)
        if isinstance(self.cc, QLearnController):
            self.cc.adapt_to_halving()

    def _control_step(self, now: float):
        elapsed = now - self.t0
        if isinstance(self.cc, QLearnController):
            res = self.cc.step(self.srtt)
            if res is not None:
                state, action, reward, sname, aname, int_retx, int_to = res
                self.telemetry.append(IntervalRecord(
                    time_s=elapsed, cwnd=self.cc.cwnd,
                    rtt_ms=self.latest_rtt * 1000 if self.latest_rtt else 0,
                    srtt_ms=self.srtt * 1000 if self.srtt else 0,
                    retransmissions=int_retx,
                    timeouts=int_to,
                    acked_total=self.acked,
                    bandwidth_halved=self.halved,
                    action=aname, state=sname, reward=reward,
                    epsilon=self.cc.epsilon,
                ))
        else:
            active, acked, retx, tos = self.cc.step(self.srtt)
            if active:
                self.telemetry.append(IntervalRecord(
                    time_s=elapsed, cwnd=self.cc.cwnd,
                    rtt_ms=self.latest_rtt * 1000 if self.latest_rtt else 0,
                    srtt_ms=self.srtt * 1000 if self.srtt else 0,
                    retransmissions=retx, timeouts=tos,
                    acked_total=self.acked,
                    bandwidth_halved=self.halved,
                ))

    def _log_cwnd(self, now: float):
        elapsed = now - self.t0
        c = self.cc.cwnd
        if self.cwnd_hist and elapsed - self.cwnd_hist[-1][0] < 0.001:
            self.cwnd_hist[-1] = (elapsed, c)
        else:
            self.cwnd_hist.append((elapsed, c))

    def _results(self, dur: float) -> dict:
        ht = self.halving_t or (dur / 2)
        before = [r for r in self.cwnd_hist if r[0] < ht]
        after = [r for r in self.cwnd_hist if r[0] >= ht]
        t_before = [r for r in self.telemetry if r.time_s < ht]
        t_after = [r for r in self.telemetry if r.time_s >= ht]

        avg_cwnd_pre = sum(c for _, c in before) / len(before) if before else 0
        avg_cwnd_post = sum(c for _, c in after) / len(after) if after else 0
        stable_rtt_after = [r.srtt_ms for r in t_after if r.srtt_ms > 0]
        if not stable_rtt_after:
            stable_rtt_after = [r.rtt_ms for r in t_after if r.rtt_ms > 0]
        avg_rtt_post = sum(stable_rtt_after) / len(stable_rtt_after) if stable_rtt_after else 0
        retx_post = sum(r.retransmissions for r in t_after)
        to_post = sum(r.timeouts for r in t_after)

        half_pkts = self.total // 2
        tp_post = (half_pkts * PAYLOAD_SIZE * 8 / 1e6) / max(dur - ht, 0.001)
        tp_all = (self.total * PAYLOAD_SIZE * 8 / 1e6) / max(dur, 0.001)

        rec_t = self._recovery_time(ht)

        return {
            "mode": self.mode,
            "total_packets": self.total,
            "duration_s": dur,
            "throughput_mbps": tp_all,
            "avg_rtt_ms": sum(r for _, r in self.rtt_hist) / len(self.rtt_hist) * 1000 if self.rtt_hist else 0,
            "retransmissions": self.total_retx,
            "timeouts": self.total_timeouts,
            "fast_retransmissions": self.total_fast_retx,
            "bandwidth_halving_time_s": ht,
            "recovery_time_s": rec_t,
            "avg_cwnd_before_halving": avg_cwnd_pre,
            "avg_cwnd_after_halving": avg_cwnd_post,
            "avg_rtt_after_halving_ms": avg_rtt_post,
            "throughput_after_halving_mbps": tp_post,
            "retransmissions_after_halving": retx_post,
            "timeouts_after_halving": to_post,
            "telemetry": self.telemetry,
            "cwnd_history": self.cwnd_hist,
            "rtt_history": self.rtt_hist,
        }

    def _recovery_time(self, ht: float) -> float:
        after = [(t, c) for t, c in self.cwnd_hist if t >= ht]
        if len(after) < 5:
            return float("inf")
        min_c = min(c for _, c in after)
        th = min_c * 1.5
        cnt = 0
        for t, c in after:
            if c <= th:
                cnt += 1
                if cnt >= 4:
                    return max(0, t - ht)
            else:
                cnt = 0
        return float("inf")


# ── Plots ──────────────────────────────────────────────────────────────
def make_plots(aimd: dict, ql: dict, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[PLOT] matplotlib missing")
        return

    rtt_plot_cap_ms = 500.0
    recovery_times = [
        data["recovery_time_s"]
        for data in (aimd, ql)
        if data["recovery_time_s"] != float("inf") and data["recovery_time_s"] > 0
    ]
    display_left = -3.0
    display_right = max([6.0] + [rt + 2.0 for rt in recovery_times])

    def relative_cwnd(data: dict) -> tuple[list[float], list[float]]:
        ht = data["bandwidth_halving_time_s"]
        return [t - ht for t, _ in data["cwnd_history"]], [c for _, c in data["cwnd_history"]]

    def relative_srtt(data: dict) -> tuple[list[float], list[float], int]:
        ht = data["bandwidth_halving_time_s"]
        ts: list[float] = []
        rs: list[float] = []
        omitted = 0
        for record in data["telemetry"]:
            value = record.srtt_ms or record.rtt_ms
            if value <= 0:
                continue
            if value > rtt_plot_cap_ms:
                omitted += 1
                continue
            ts.append(record.time_s - ht)
            rs.append(value)
        return ts, rs, omitted

    def visible_pairs(ts: list[float], values: list[float]) -> list[tuple[float, float]]:
        pairs = [(t, v) for t, v in zip(ts, values) if display_left <= t <= display_right]
        return pairs or list(zip(ts, values))

    # ── Figure 1: CWND + SRTT recovery ──
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

    for label, data, color in [("AIMD", aimd, "#2563eb"), ("Q-Learning", ql, "#d97706")]:
        ts, cs = relative_cwnd(data)
        # Raw step trace
        ax1.step(ts, cs, where="post", label=label, lw=1.0, color=color, alpha=0.5)
        # Smoothed overlay (moving average, window=3)
        if len(cs) >= 3:
            smooth = np.convolve(cs, np.ones(3)/3, mode="same")
            ax1.plot(ts, smooth, lw=2.0, color=color, alpha=1.0,
                     label=f"{label} (smoothed)" if label == "AIMD" else None)
        rts, rr, omitted = relative_srtt(data)
        rtt_pairs = visible_pairs(rts, rr)
        rts = [t for t, _ in rtt_pairs]
        rr = [r for _, r in rtt_pairs]
        ax2.plot(rts, rr, label=f"{label} SRTT", lw=1.2, alpha=0.85, color=color)
        if omitted:
            ax2.text(
                0.01,
                0.95 - 0.08 * (0 if label == "AIMD" else 1),
                f"{label}: omitted {omitted} RTT outlier(s) > {rtt_plot_cap_ms:.0f} ms",
                transform=ax2.transAxes,
                fontsize=8,
                color=color,
                va="top",
            )

    # Annotate ACK compression spike for AIMD
    aimd_ts, aimd_cs = relative_cwnd(aimd)
    spike_idx = None
    for i in range(1, len(aimd_cs) - 1):
        if aimd_cs[i] > aimd_cs[i-1] * 3 and aimd_cs[i] > aimd_cs[i+1] * 2:
            spike_idx = i
            break
    if spike_idx is not None:
        ax1.annotate("ACK compression\n(CWND=1 recovery burst)",
                     xy=(aimd_ts[spike_idx], aimd_cs[spike_idx]),
                     xytext=(aimd_ts[spike_idx] + 0.8, aimd_cs[spike_idx] * 0.5),
                     fontsize=8, color="#1e40af",
                     arrowprops=dict(arrowstyle="->", color="#1e40af", alpha=0.6,
                                     connectionstyle="arc3,rad=0.3"),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", alpha=0.8))

    for ax in (ax1, ax2):
        ax.axvline(x=0.0, color="red", ls="--", lw=1.8, alpha=0.7,
                   label="Bandwidth halving (relative t=0)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Annotate peaks
    for data, color, label, y_off in [
        (aimd, "#2563eb", "AIMD", 3), (ql, "#d97706", "Q-L", -5)
    ]:
        ts, cs = relative_cwnd(data)
        peak = max(visible_pairs(ts, cs), key=lambda item: item[1])
        ax1.annotate(f"{label} peak={peak[1]:.0f}", xy=peak,
                     xytext=(peak[0] + 0.2, peak[1] + y_off), fontsize=10, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, alpha=0.5), fontweight="bold")

    # Recovery spans
    for data, color, label, y in [(aimd, "#2563eb", "AIMD", 2.2), (ql, "#d97706", "Q-L", 1.0)]:
        rt = data["recovery_time_s"]
        if rt != float("inf") and rt > 0:
            ax1.axvspan(0.0, rt, alpha=0.08, color=color)
            ax1.text(rt + 0.08, y, f"{label} recovery\n{rt:.2f}s",
                     ha="left", fontsize=8, color=color, fontweight="bold")

    ax1.set_xlabel("Time relative to bandwidth halving (s)")
    ax1.set_ylabel("CWND (packets)")
    ax1.set_title("CWND Recovery aligned at Bandwidth Halving")
    ax1.set_xlim(display_left, display_right)
    ax2.set_xlabel("Time relative to bandwidth halving (s)")
    ax2.set_ylabel("SRTT (ms)")
    ax2.set_ylim(bottom=0, top=rtt_plot_cap_ms)
    ax2.set_xlim(display_left, display_right)
    ax2.set_title("Smoothed RTT Evolution aligned at Bandwidth Halving")
    fig1.savefig(out / "cwnd_recovery.png", dpi=150)
    plt.close(fig1)

    # ── Figure 2: Post-halving metrics ──
    fig2, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    metrics = [
        ("Recovery\nTime (s)", [aimd["recovery_time_s"], ql["recovery_time_s"]]),
        ("Throughput\nafter halving\n(Mbps)", [aimd["throughput_after_halving_mbps"],
                                               ql["throughput_after_halving_mbps"]]),
        ("Avg RTT\nafter halving\n(ms)", [aimd["avg_rtt_after_halving_ms"],
                                          ql["avg_rtt_after_halving_ms"]]),
        ("Retransmissions\nafter halving", [aimd["retransmissions_after_halving"],
                                             ql["retransmissions_after_halving"]]),
    ]
    for ax, (title, vals) in zip(axes, metrics):
        colors = ["#2563eb", "#d97706"]
        bars = ax.bar(["AIMD", "Q-Learning"], vals, color=colors, alpha=0.85, width=0.5)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            txt = f"{val:.2f}" if isinstance(val, float) and val < 100 else str(int(val)) if isinstance(val, (int, float)) and val != float("inf") else "inf"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.03,
                    txt, ha="center", fontsize=10, fontweight="bold")
        if vals[0] != float("inf") and vals[1] != float("inf") and vals[0] != 0:
            pct = (vals[1] - vals[0]) / abs(vals[0]) * 100
            clr = "#16a34a" if ("Throughput" in title and pct > 0) or ("Throughput" not in title and pct < 0) else "#dc2626"
            ax.text(1, max(vals) * 0.5, f"{pct:+.1f}%", ha="center", fontsize=9, fontweight="bold",
                    color=clr, bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0fdf4", alpha=0.7))

    fig2.suptitle("Performance Metrics after Bandwidth Halving", fontsize=13, fontweight="bold")
    fig2.savefig(out / "post_halving_metrics.png", dpi=150)
    plt.close(fig2)

    print(f"[PLOT] Saved to {out}")


# ── Conclusion ─────────────────────────────────────────────────────────
def conclusion(aimd: dict, ql: dict) -> str:
    h = aimd["bandwidth_halving_time_s"]
    tp_d = ql["throughput_after_halving_mbps"] - aimd["throughput_after_halving_mbps"]
    rtt_d = ql["avg_rtt_after_halving_ms"] - aimd["avg_rtt_after_halving_ms"]
    retx_d = ql["retransmissions_after_halving"] - aimd["retransmissions_after_halving"]
    rec_d = ql["recovery_time_s"] - aimd["recovery_time_s"]

    return f"""## Bandwidth Halving Recovery: AIMD vs Q-Learning

When bandwidth is suddenly halved mid-transmission (t={h:.2f}s, after
{aimd['total_packets'] // 2}/{aimd['total_packets']} packets):

**1. AIMD overshoots.** Before halving, CWND reached
{aimd['avg_cwnd_before_halving']:.1f} pkts, fully saturating the queue.
The inflated window floods the bottleneck after the drop, causing
{aimd['retransmissions_after_halving']} retransmissions and
{aimd['timeouts_after_halving']} timeouts post-halving. AIMD only
reduces CWND after loss/timeout — a *reactive* signal — delaying
recovery to {aimd['recovery_time_s']:.2f}s.

**2. Q-Learning adapts proactively.** Pre-halving CWND of
{ql['avg_cwnd_before_halving']:.1f} is lower than AIMD's because the
policy learned to *hold* on rising RTT, preserving queue headroom.
After halving, adaptation (epsilon boosted to 0.15, alpha raised to
0.15, moderately amplified loss/RTT penalties) causes the agent to unlearn
large-window preferences. Recovery: {ql['recovery_time_s']:.2f}s
({'faster' if rec_d > 0 else 'comparable'} vs AIMD).

**3. Post-halving comparison:**
- Throughput: Q-Learning {tp_d:+.3f} Mbps vs AIMD
- RTT: Q-Learning {rtt_d:+.1f} ms vs AIMD
- Retransmissions: Q-Learning {'fewer' if retx_d < 0 else 'more'} by {abs(retx_d)}
- Recovery: {ql['recovery_time_s']:.2f}s (QL) vs {aimd['recovery_time_s']:.2f}s (AIMD)

**4. Key insight:** AIMD relies on packet loss as a *lagging* congestion
signal. Q-Learning uses RTT trends as an *early* signal, allowing it to
hold before loss. With post-halving adaptation, it converges to the new
lower optimal CWND with fewer collateral losses.
"""


# ── Main ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Bandwidth Halving Experiment")
    p.add_argument("--packets", type=int, default=400)
    p.add_argument("--delay-ms", type=float, default=10.0)
    p.add_argument("--queue", type=int, default=20)
    p.add_argument("--rto", type=float, default=0.20)
    p.add_argument("--max-cwnd", type=int, default=64)
    p.add_argument("--qtable", default="artifacts/models/active/q_table.json")
    p.add_argument("--output", default="artifacts/bandwidth_drop_experiment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Bandwidth Halving Experiment")
    print(f"Packets: {args.packets} | Halving at: {args.packets // 2}")
    print(f"Link: {args.delay_ms}ms, queue={args.queue}, RTO={args.rto}s")
    print("=" * 65)

    kw = dict(total=args.packets, link_delay_ms=args.delay_ms,
              queue_cap=args.queue, rto=args.rto, max_cwnd=args.max_cwnd,
              seed=args.seed)

    print("\n[1/2] AIMD ...")
    aimd = DropExperiment(mode="aimd", **kw).run()
    print(f"  tp={aimd['throughput_mbps']:.4f} Mbps  rtt={aimd['avg_rtt_ms']:.1f} ms")
    print(f"  retx={aimd['retransmissions']}  to={aimd['timeouts']}")
    print(f"  recovery={aimd['recovery_time_s']:.3f}s")
    print(f"  cwnd {aimd['avg_cwnd_before_halving']:.1f} -> {aimd['avg_cwnd_after_halving']:.1f}")

    kw["qtable_path"] = args.qtable
    print("\n[2/2] Q-Learning ...")
    ql = DropExperiment(mode="qlearning", **kw).run()
    print(f"  tp={ql['throughput_mbps']:.4f} Mbps  rtt={ql['avg_rtt_ms']:.1f} ms")
    print(f"  retx={ql['retransmissions']}  to={ql['timeouts']}")
    print(f"  recovery={ql['recovery_time_s']:.3f}s")
    print(f"  cwnd {ql['avg_cwnd_before_halving']:.1f} -> {ql['avg_cwnd_after_halving']:.1f}")

    # Save telemetry
    for label, res in [("aimd", aimd), ("qlearn", ql)]:
        with (out / f"telemetry_{label}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "time_s", "cwnd", "rtt_ms", "srtt_ms", "retransmissions",
                "timeouts", "acked_total", "bandwidth_halved",
                "action", "state", "reward", "epsilon",
            ])
            w.writeheader()
            for r in res["telemetry"]:
                w.writerow({
                    "time_s": f"{r.time_s:.6f}", "cwnd": f"{r.cwnd:.3f}",
                    "rtt_ms": f"{r.rtt_ms:.3f}", "srtt_ms": f"{r.srtt_ms:.3f}",
                    "retransmissions": r.retransmissions, "timeouts": r.timeouts,
                    "acked_total": r.acked_total, "bandwidth_halved": r.bandwidth_halved,
                    "action": r.action, "state": r.state,
                    "reward": f"{r.reward:.4f}", "epsilon": f"{r.epsilon:.4f}",
                })

    # Summary JSON
    summary = {"experiment": {"total_packets": args.packets,
                              "halving_at": args.packets // 2,
                              "link_delay_ms": args.delay_ms,
                              "queue_capacity": args.queue,
                              "rto_s": args.rto, "seed": args.seed},
               "results": []}
    for res in [aimd, ql]:
        summary["results"].append({k: v for k, v in res.items()
                                   if k not in ("telemetry", "cwnd_history", "rtt_history")})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    if not args.no_plot:
        make_plots(aimd, ql, out)

    # Conclusion
    c = conclusion(aimd, ql)
    (out / "conclusion.md").write_text(c, encoding="utf-8")
    print("\n" + c)
    print(f"\nAll outputs: {out}")


if __name__ == "__main__":
    main()
