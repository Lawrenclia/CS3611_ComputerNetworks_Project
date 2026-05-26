import json
import random
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


Action = str

RTT_TRENDS = ("rtt_up", "rtt_down", "rtt_stable")
LOSS_FLAGS = ("loss", "no_loss")
Q_STATES = tuple(f"{trend}|{loss}" for trend in RTT_TRENDS for loss in LOSS_FLAGS)
Q_ACTIONS: tuple[Action, ...] = ("0", "1", "2")
ACTION_NAMES = {
    "0": "hold",
    "1": "cwnd+1",
    "2": "cwnd/2",
    "aimd_inc": "cwnd+1/cwnd",
    "aimd_halve": "cwnd/2",
}

STATE_INDICES = {s: i for i, s in enumerate(Q_STATES)}
ACTION_INDICES = {a: i for i, a in enumerate(Q_ACTIONS)}


@dataclass
class ControlDecision:
    event: str
    state: str
    action: Action
    old_window: int
    new_window: int
    reward: float
    q_value: float
    loss_ewma: float
    throughput: float = 0.0
    avg_rtt: float = 0.0
    loss_count: int = 0


class FixedWindowController:
    def __init__(self, window_size: int) -> None:
        self.current_window = max(1, window_size)

    def observe_ack(
        self,
        newly_acked: int,
        srtt: Optional[float],
        latest_rtt: Optional[float],
        inflight: int,
    ) -> None:
        return None

    def observe_loss(
        self,
        reason: str,
        srtt: Optional[float],
        latest_rtt: Optional[float],
        inflight: int,
    ) -> None:
        return None

    def summary(self) -> str:
        return f"cc=fixed window={self.current_window}"

    def close(self) -> None:
        return None


class AIMDCongestionController:
    def __init__(self, min_window: int = 1, max_window: int = 64) -> None:
        if min_window <= 0:
            raise ValueError("min_window must be positive")
        if max_window < min_window:
            raise ValueError("max_window must be >= min_window")

        self.min_window = min_window
        self.max_window = max_window
        self.cwnd = 1.0
        self.current_window = 1
        self.ack_events = 0
        self.loss_events = 0

    def observe_ack(
        self,
        newly_acked: int,
        srtt: Optional[float],
        latest_rtt: Optional[float],
        inflight: int,
    ) -> Optional[ControlDecision]:
        if newly_acked <= 0:
            return None

        old_window = self.current_window
        for _ in range(newly_acked):
            self.cwnd += 1.0 / max(self.cwnd, 1.0)
            self.cwnd = min(self.cwnd, float(self.max_window))
        self.current_window = self._integer_window()
        self.ack_events += newly_acked

        return ControlDecision(
            event="ACK",
            state="aimd",
            action="aimd_inc",
            old_window=old_window,
            new_window=self.current_window,
            reward=0.0,
            q_value=self.cwnd,
            loss_ewma=0.0,
            throughput=float(newly_acked),
            avg_rtt=latest_rtt or srtt or 0.0,
            loss_count=0,
        )

    def observe_loss(
        self,
        reason: str,
        srtt: Optional[float],
        latest_rtt: Optional[float],
        inflight: int,
    ) -> Optional[ControlDecision]:
        if reason != "RTO":
            return None

        old_window = self.current_window
        self.cwnd = max(float(self.min_window), self.cwnd / 2.0)
        self.current_window = self._integer_window()
        self.loss_events += 1

        return ControlDecision(
            event=reason,
            state="aimd",
            action="aimd_halve",
            old_window=old_window,
            new_window=self.current_window,
            reward=0.0,
            q_value=self.cwnd,
            loss_ewma=0.0,
            throughput=0.0,
            avg_rtt=latest_rtt or srtt or 0.0,
            loss_count=1,
        )

    def summary(self) -> str:
        return (
            "cc=aimd window={window} cwnd={cwnd:.3f} ack_events={acks} "
            "rto_loss_events={losses}"
        ).format(
            window=self.current_window,
            cwnd=self.cwnd,
            acks=self.ack_events,
            losses=self.loss_events,
        )

    def close(self) -> None:
        return None

    def _integer_window(self) -> int:
        return min(self.max_window, max(self.min_window, int(self.cwnd)))


class QLearningCongestionController:
    def __init__(
        self,
        initial_window: int,
        min_window: int,
        max_window: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        q_table_path: Optional[str] = None,
        seed: Optional[int] = None,
        reward_throughput_weight: float = 1.0,
        reward_rtt_weight: float = 0.02,
        reward_loss_weight: float = 3.0,
        rtt_trend_threshold: float = 0.10,
        min_cycle_seconds: float = 0.001,
    ) -> None:
        if min_window <= 0:
            raise ValueError("min_window must be positive")
        if max_window < min_window:
            raise ValueError("max_window must be >= min_window")

        self.min_window = min_window
        self.max_window = max_window
        self.current_window = self._clamp_window(initial_window)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = Path(q_table_path) if q_table_path else None
        self.rng = random.Random(seed)

        self.reward_throughput_weight = reward_throughput_weight
        self.reward_rtt_weight = reward_rtt_weight
        self.reward_loss_weight = reward_loss_weight
        self.rtt_trend_threshold = rtt_trend_threshold
        self.min_cycle_seconds = min_cycle_seconds

        self.q_table_matrix = np.zeros((len(Q_STATES), len(Q_ACTIONS)), dtype=np.float64)
        
        self.last_state: Optional[str] = None
        self.last_action: Optional[Action] = None
        self.previous_cycle_avg_rtt: Optional[float] = None
        self.loss_ewma = 0.0
        self.decisions = 0
        self.learning_updates = 0

        self.cycle_started_at = time.monotonic()
        self.cycle_acked = 0
        self.cycle_loss_count = 0
        self.cycle_rtt_sum = 0.0
        self.cycle_rtt_samples = 0
        self.latest_srtt: Optional[float] = None

        if self.q_table_path and self.q_table_path.exists():
            self._load_q_table(self.q_table_path)

    def observe_ack(
        self,
        newly_acked: int,
        srtt: Optional[float],
        latest_rtt: Optional[float],
        inflight: int,
    ) -> Optional[ControlDecision]:
        self.latest_srtt = srtt or self.latest_srtt
        self.cycle_acked += newly_acked
        sample = latest_rtt or srtt
        if sample is not None and sample >= 0:
            self.cycle_rtt_sum += sample
            self.cycle_rtt_samples += 1
        self.loss_ewma *= 0.85
        return self._maybe_finish_rtt_cycle("ACK")

    def observe_loss(
        self,
        reason: str,
        srtt: Optional[float],
        latest_rtt: Optional[float],
        inflight: int,
    ) -> Optional[ControlDecision]:
        self.latest_srtt = srtt or self.latest_srtt
        self.cycle_loss_count += 1
        self.loss_ewma = min(1.0, 0.80 * self.loss_ewma + 0.20)
        sample = latest_rtt or srtt
        if sample is not None and sample >= 0:
            self.cycle_rtt_sum += sample
            self.cycle_rtt_samples += 1
        return self._maybe_finish_rtt_cycle(reason)

    def summary(self) -> str:
        return (
            "cc=q-learning window={window} states=6 actions=3 decisions={decisions} "
            "updates={updates} epsilon={epsilon:.3f} loss_ewma={loss:.3f}"
        ).format(
            window=self.current_window,
            decisions=self.decisions,
            updates=self.learning_updates,
            epsilon=self.epsilon,
            loss=self.loss_ewma,
        )

    def close(self) -> None:
        if not self.q_table_path:
            return
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {
            state: {
                action: float(self.q_table_matrix[STATE_INDICES[state], ACTION_INDICES[action]])
                for action in Q_ACTIONS
            }
            for state in Q_STATES
        }
        self.q_table_path.write_text(
            json.dumps(dump_data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _maybe_finish_rtt_cycle(self, event: str) -> Optional[ControlDecision]:
        now = time.monotonic()
        cycle_seconds = self._current_cycle_seconds()
        if now - self.cycle_started_at < cycle_seconds:
            return None
        return self._finish_rtt_cycle(event, now)

    def _finish_rtt_cycle(self, event: str, now: float) -> ControlDecision:
        duration = max(now - self.cycle_started_at, 1e-6)
        avg_rtt = self._cycle_avg_rtt()
        throughput = self.cycle_acked / duration
        loss_count = self.cycle_loss_count
        state = self._build_state(avg_rtt, loss_count)
        reward = (
            self.reward_throughput_weight * throughput
            - self.reward_rtt_weight * avg_rtt * 1000.0
            - self.reward_loss_weight * loss_count
        )

        self._learn(reward, state)
        action = self._select_action(state)
        old_window = self.current_window
        self.current_window = self._next_window(action)
        q_value = self._q_values(state)[action]

        self.last_state = state
        self.last_action = action
        self.previous_cycle_avg_rtt = avg_rtt
        self.decisions += 1
        self._reset_cycle(now)

        return ControlDecision(
            event=event,
            state=state,
            action=action,
            old_window=old_window,
            new_window=self.current_window,
            reward=reward,
            q_value=q_value,
            loss_ewma=self.loss_ewma,
            throughput=throughput,
            avg_rtt=avg_rtt,
            loss_count=loss_count,
        )

    def _learn(self, reward: float, next_state: str) -> None:
        if self.last_state is None or self.last_action is None:
            return

        s_idx = STATE_INDICES[self.last_state]
        a_idx = ACTION_INDICES[self.last_action]
        next_s_idx = STATE_INDICES[next_state]

        # Numpy matrix update for Q-Learning Bellman equation
        old_value = self.q_table_matrix[s_idx, a_idx]
        next_best = np.max(self.q_table_matrix[next_s_idx])
        new_value = old_value + self.alpha * (reward + self.gamma * next_best - old_value)
        self.q_table_matrix[s_idx, a_idx] = new_value
        
        self.learning_updates += 1

    def _select_action(self, state: str) -> Action:
        candidates = self._valid_actions()
        if self.rng.random() < self.epsilon:
            return self.rng.choice(candidates)

        values = self._q_values(state)
        best_value = max(values[action] for action in candidates)
        best_actions = [action for action in candidates if values[action] == best_value]
        if len(best_actions) == len(candidates):
            return self._tie_break_action(state, candidates)
        return best_actions[0]

    def _tie_break_action(self, state: str, candidates: list[Action]) -> Action:
        rtt_trend, loss_flag = state.split("|")
        if loss_flag == "loss" or rtt_trend == "rtt_up":
            return "2" if "2" in candidates else "0"
        if rtt_trend == "rtt_down":
            return "1" if "1" in candidates else "0"
        return "1" if "1" in candidates else "0"

    def _valid_actions(self) -> list[Action]:
        actions = list(Q_ACTIONS)
        if self.current_window <= self.min_window:
            actions.remove("2")
        if self.current_window >= self.max_window:
            actions.remove("1")
        return actions

    def _next_window(self, action: Action) -> int:
        if action == "1":
            return self._clamp_window(self.current_window + 1)
        if action == "2":
            return self._clamp_window(max(self.min_window, self.current_window // 2))
        return self.current_window

    def _build_state(self, avg_rtt: float, loss_count: int) -> str:
        if self.previous_cycle_avg_rtt is None or self.previous_cycle_avg_rtt <= 0:
            trend = "rtt_stable"
        elif avg_rtt > self.previous_cycle_avg_rtt * (1.0 + self.rtt_trend_threshold):
            trend = "rtt_up"
        elif avg_rtt < self.previous_cycle_avg_rtt * (1.0 - self.rtt_trend_threshold):
            trend = "rtt_down"
        else:
            trend = "rtt_stable"

        loss_flag = "loss" if loss_count > 0 else "no_loss"
        return f"{trend}|{loss_flag}"

    def _cycle_avg_rtt(self) -> float:
        if self.cycle_rtt_samples > 0:
            return self.cycle_rtt_sum / self.cycle_rtt_samples
        if self.latest_srtt is not None:
            return self.latest_srtt
        if self.previous_cycle_avg_rtt is not None:
            return self.previous_cycle_avg_rtt
        return self.min_cycle_seconds

    def _current_cycle_seconds(self) -> float:
        sample = self.latest_srtt or self.previous_cycle_avg_rtt or self.min_cycle_seconds
        return max(self.min_cycle_seconds, sample)

    def _reset_cycle(self, now: float) -> None:
        self.cycle_started_at = now
        self.cycle_acked = 0
        self.cycle_loss_count = 0
        self.cycle_rtt_sum = 0.0
        self.cycle_rtt_samples = 0

    def _q_values(self, state: str) -> dict[Action, float]:
        row = self.q_table_matrix[STATE_INDICES[state]]
        return {action: float(row[ACTION_INDICES[action]]) for action in Q_ACTIONS}

    def _clamp_window(self, window: int) -> int:
        return min(self.max_window, max(self.min_window, int(window)))

    def _load_q_table(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("q-table must be a JSON object")
        for state in Q_STATES:
            values = raw.get(state, {})
            if not isinstance(values, dict):
                continue
            s_idx = STATE_INDICES[state]
            for action in Q_ACTIONS:
                val = float(values.get(action, 0.0))
                self.q_table_matrix[s_idx, ACTION_INDICES[action]] = val
