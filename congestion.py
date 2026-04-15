import json
import math
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IntervalSnapshot:
    started_at: float
    ended_at: float
    throughput_mbps: float
    avg_rtt_ms: float
    loss_count: int
    state: int
    action: int
    reward: float


class AIMDController:
    name = "aimd"

    def __init__(self) -> None:
        self.cwnd = 1.0

    def reset(self, training: bool = False) -> None:
        self.cwnd = 1.0

    def window_limit(self) -> int:
        return max(1, int(self.cwnd))

    def on_ack(self) -> None:
        self.cwnd += 1.0 / max(self.cwnd, 1.0)

    def on_loss(self) -> None:
        self.cwnd = max(1.0, self.cwnd / 2.0)

    def maybe_step(self, now: float, srtt: Optional[float]) -> Optional[IntervalSnapshot]:
        return None


class QLearningController:
    name = "q_learning"

    def __init__(
        self,
        initial_q_table: Optional[List[List[float]]] = None,
        epsilon: float = 0.35,
        epsilon_decay: float = 0.92,
        min_epsilon: float = 0.05,
        learning_rate: float = 0.25,
        discount: float = 0.9,
        max_cwnd: int = 48,
    ) -> None:
        self.q_table = initial_q_table or [[0.0, 0.0, 0.0] for _ in range(6)]
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.max_cwnd = max_cwnd
        self.cwnd = 1.0
        self.training = True
        self._prev_avg_rtt = None
        self._last_state = None
        self._last_action = None
        self._interval_start = None
        self._acked_bytes = 0
        self._rtt_samples = []
        self._losses = 0

    def reset(self, training: bool = True) -> None:
        self.cwnd = 1.0
        self.training = training
        self._prev_avg_rtt = None
        self._last_state = None
        self._last_action = None
        self._interval_start = None
        self._acked_bytes = 0
        self._rtt_samples = []
        self._losses = 0

    def window_limit(self) -> int:
        return max(1, int(self.cwnd))

    def record_ack(self, payload_bytes: int, rtt_seconds: float) -> None:
        self._acked_bytes += payload_bytes
        self._rtt_samples.append(rtt_seconds)

    def record_loss(self) -> None:
        self._losses += 1

    def on_ack(self) -> None:
        return None

    def on_loss(self) -> None:
        return None

    def maybe_step(self, now: float, srtt: Optional[float]) -> Optional[IntervalSnapshot]:
        if self._interval_start is None:
            self._interval_start = now
            return None

        control_interval = min(max(srtt or 0.10, 0.08), 0.40)
        if now - self._interval_start < control_interval:
            return None

        avg_rtt = (
            sum(self._rtt_samples) / len(self._rtt_samples)
            if self._rtt_samples
            else (self._prev_avg_rtt or control_interval)
        )
        if self._prev_avg_rtt is None:
            trend_index = 1
        else:
            delta_ratio = (avg_rtt - self._prev_avg_rtt) / max(self._prev_avg_rtt, 1e-6)
            if delta_ratio > 0.1:
                trend_index = 2
            elif delta_ratio < -0.1:
                trend_index = 0
            else:
                trend_index = 1

        loss_flag = 1 if self._losses else 0
        state = trend_index * 2 + loss_flag
        interval_duration = max(now - self._interval_start, 1e-6)
        throughput_mbps = (self._acked_bytes * 8.0) / interval_duration / 1_000_000.0
        avg_rtt_ms = avg_rtt * 1000.0
        reward = (8.0 * throughput_mbps) - (0.015 * avg_rtt_ms) - (1.4 * self._losses)

        if self.training and self._last_state is not None and self._last_action is not None:
            best_future = max(self.q_table[state])
            current = self.q_table[self._last_state][self._last_action]
            updated = current + self.learning_rate * (
                reward + self.discount * best_future - current
            )
            self.q_table[self._last_state][self._last_action] = updated

        action = self._choose_action(state)
        self._apply_action(action)
        snapshot = IntervalSnapshot(
            started_at=self._interval_start,
            ended_at=now,
            throughput_mbps=throughput_mbps,
            avg_rtt_ms=avg_rtt_ms,
            loss_count=self._losses,
            state=state,
            action=action,
            reward=reward,
        )
        self._prev_avg_rtt = avg_rtt
        self._last_state = state
        self._last_action = action
        self._interval_start = now
        self._acked_bytes = 0
        self._rtt_samples = []
        self._losses = 0
        return snapshot

    def finish_episode(self) -> None:
        if self.training:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "epsilon": self.epsilon,
                    "q_table": self.q_table,
                },
                handle,
                indent=2,
            )

    def _choose_action(self, state: int) -> int:
        if self.training and random.random() < self.epsilon:
            return random.randrange(3)
        row = self.q_table[state]
        best_value = max(row)
        best_actions = [idx for idx, value in enumerate(row) if math.isclose(value, best_value)]
        return random.choice(best_actions)

    def _apply_action(self, action: int) -> None:
        if action == 1:
            self.cwnd = min(float(self.max_cwnd), self.cwnd + 1.0)
        elif action == 2:
            self.cwnd = max(1.0, self.cwnd / 2.0)
        else:
            self.cwnd = min(float(self.max_cwnd), self.cwnd)
