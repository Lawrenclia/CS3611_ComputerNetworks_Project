from __future__ import annotations

import argparse
import csv
import json
import random
import site
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from protocol import PAYLOAD_SIZE, build_payload, pack_data_packet, unpack_ack
from virtual_link import VirtualFunnelLink

Q_STATE_NAMES = (
    "rtt_up|no_loss",
    "rtt_up|loss",
    "rtt_down|no_loss",
    "rtt_down|loss",
    "rtt_stable|no_loss",
    "rtt_stable|loss",
)
Q_EXPANDED_CWND_BUCKETS = ("cwnd_low", "cwnd_mid", "cwnd_high")
Q_EXPANDED_STATE_NAMES = tuple(
    f"{state}|{bucket}"
    for state in Q_STATE_NAMES
    for bucket in Q_EXPANDED_CWND_BUCKETS
)
Q_ACTION_KEYS = ("0", "1", "2")
Q_ACTION_NAMES = ("hold", "cwnd+1", "cwnd/2")
DQN_STATE_FEATURES = (
    "rtt_ms",
    "rtt_trend_percent",
    "loss_percent",
    "timeout_percent",
    "cwnd",
    "ack_ratio",
)
DQN_ACTION_MULTIPLIERS = (0.70, 0.90, 1.00, 1.10, 1.25)
DQN_ACTION_NAMES = ("halve", "gentle_decrease", "hold", "probe_plus", "probe_fast")
DQN_MIN_PROBE_INCREASE = 0.50
DQN_MAX_INCREASE_PER_STEP = 1.00


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
        reward_throughput_weight: float,
        reward_rtt_weight: float,
        reward_timeout_weight: float,
        reward_retx_weight: float,
        reward_cwnd_weight: float,
        reward_target_rtt_ms: float | None,
        qtable_file: str | None,
        dqn_model_file: str | None,
        dqn_lr: float,
        dqn_batch_size: int,
        dqn_replay_capacity: int,
        dqn_target_update: int,
        dqn_eval: bool,
        q_eval: bool,
        verbose: bool,
    ) -> None:
        self.mode = mode
        self.cwnd = max(1.0, float(initial_cwnd))
        self.max_cwnd = max(1.0, float(max_cwnd))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reward_throughput_weight = reward_throughput_weight
        self.reward_rtt_weight = reward_rtt_weight
        self.reward_timeout_weight = reward_timeout_weight
        self.reward_retx_weight = reward_retx_weight
        self.reward_cwnd_weight = reward_cwnd_weight
        self.reward_target_rtt_ms = reward_target_rtt_ms
        self.qtable_file = Path(qtable_file) if qtable_file else None
        self.dqn_model_file = Path(dqn_model_file) if dqn_model_file else None
        self.dqn_lr = dqn_lr
        self.dqn_batch_size = dqn_batch_size
        self.dqn_replay_capacity = dqn_replay_capacity
        self.dqn_target_update = dqn_target_update
        self.dqn_eval = dqn_eval
        self.q_eval = q_eval
        self.verbose = verbose

        self.q_table = [[0.0, 0.0, 0.0] for _ in Q_STATE_NAMES]
        self.last_state = 0
        self.last_action = 0
        self.last_srtt = None
        self.interval_acked = 0
        self.interval_losses = 0
        self.interval_retransmissions = 0
        self.interval_timeouts = 0
        self.interval_fast_retransmissions = 0
        self.interval_rtt_sum = 0.0
        self.interval_rtt_count = 0
        self._load_q_table()

        self.torch = None
        self.dqn_policy = None
        self.dqn_target = None
        self.dqn_optimizer = None
        self.dqn_loss_fn = None
        self.dqn_replay = deque(maxlen=max(1, dqn_replay_capacity))
        self.dqn_last_state = None
        self.dqn_last_action = None
        self.dqn_steps = 0
        if self.mode == "dqn":
            self._init_dqn()

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

    def on_loss(self, reason: str = "RTO") -> None:
        self.interval_losses += 1
        self.interval_retransmissions += 1
        if reason == "RTO":
            self.interval_timeouts += 1
        elif reason == "FAST":
            self.interval_fast_retransmissions += 1
        if self.mode == "aimd":
            self.cwnd = max(1.0, self.cwnd / 2.0)

    def maybe_step_qlearning(self, srtt: float | None) -> tuple[str, object, float, str] | None:
        if self.mode == "dqn":
            return self._step_dqn(srtt)
        if self.mode != "qlearning":
            self._reset_interval()
            return None

        state = self._state_from_interval(srtt)
        reward = self._reward()
        if not self.q_eval:
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
        return "qlearning", (state, action), reward, f"cwnd={self.cwnd:.2f}"

    def save(self) -> None:
        if self.mode == "dqn":
            self._save_dqn()
            return
        if self.mode != "qlearning" or self.qtable_file is None or self.q_eval:
            return
        try:
            self.qtable_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "metadata": {
                    "state_features": ["rtt_trend", "loss_flag"],
                    "state_count": len(Q_STATE_NAMES),
                    "actions": {
                        action_key: action_name
                        for action_key, action_name in zip(Q_ACTION_KEYS, Q_ACTION_NAMES)
                    },
                }
            }
            for state_index, state_name in enumerate(Q_STATE_NAMES):
                data[state_name] = {
                    action_key: self.q_table[state_index][action_index]
                    for action_index, action_key in enumerate(Q_ACTION_KEYS)
                }
            self.qtable_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            if self.verbose:
                print(f"[SENDER][QLEARN] failed to save q-table to {self.qtable_file}: {exc}", flush=True)

    def _choose_action(self, state: int) -> int:
        if not self.q_eval and random.random() < self.epsilon:
            return random.randrange(len(Q_ACTION_KEYS))
        row = self.q_table[state]
        return max(range(len(Q_ACTION_KEYS)), key=lambda i: row[i])

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
        loss_flag = 1 if self.interval_retransmissions > 0 else 0
        return trend * 2 + loss_flag

    def _reward(self) -> float:
        avg_rtt = (
            self.interval_rtt_sum / self.interval_rtt_count
            if self.interval_rtt_count
            else 0.0
        )
        avg_rtt_ms = avg_rtt * 1000.0
        if self.reward_target_rtt_ms is None:
            rtt_penalty_ms = avg_rtt_ms
        else:
            rtt_penalty_ms = max(0.0, avg_rtt_ms - self.reward_target_rtt_ms)
        return (
            self.reward_throughput_weight * self.interval_acked
            - self.reward_timeout_weight * self.interval_timeouts
            - self.reward_retx_weight * self.interval_retransmissions
            - self.reward_rtt_weight * rtt_penalty_ms
        )

    def _step_dqn(self, srtt: float | None) -> tuple[str, object, float, str]:
        state = self._continuous_state(srtt)
        reward = self._reward()
        # CWND efficiency bonus: reward maintaining higher CWND (only for DQN)
        # Prevents the "lazy policy" where DQN stays at minimal CWND forever.
        reward += self.reward_cwnd_weight * (self.cwnd / max(1.0, self.max_cwnd))
        if (
            not self.dqn_eval
            and self.dqn_last_state is not None
            and self.dqn_last_action is not None
        ):
            self.dqn_replay.append(
                (self.dqn_last_state, self.dqn_last_action, reward, state)
            )
            self._train_dqn_batch()

        action = self._choose_dqn_action(state)
        old_cwnd = self.cwnd
        multiplier = DQN_ACTION_MULTIPLIERS[action]
        self.cwnd = self._apply_dqn_action(multiplier)
        if not self.dqn_eval:
            self.dqn_last_state = state
            self.dqn_last_action = action
        self.dqn_steps += 1
        if not self.dqn_eval and self.dqn_steps % self.dqn_target_update == 0:
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())

        detail = (
            "state=[rtt_ms={rtt:.3f}, rtt_trend_pct={trend:.2f}, loss_pct={loss:.2f}, "
            "timeout_pct={timeout:.2f}, cwnd={old:.2f}, ack_ratio={ack_ratio:.2f}] "
            "action={action}({action_name}) multiplier={mult:.2f} cwnd={new:.2f} "
            "replay={replay} eval={eval_mode}"
        ).format(
            rtt=state[0],
            trend=state[1],
            loss=state[2],
            timeout=state[3],
            old=old_cwnd,
            ack_ratio=state[5],
            action=action,
            action_name=DQN_ACTION_NAMES[action],
            mult=multiplier,
            new=self.cwnd,
            replay=len(self.dqn_replay),
            eval_mode=self.dqn_eval,
        )
        self._reset_interval()
        self.last_srtt = srtt
        return "dqn", (state, action, multiplier), reward, detail

    def _apply_dqn_action(self, multiplier: float) -> float:
        if multiplier > 1.0:
            desired_cwnd = max(
                self.cwnd + DQN_MIN_PROBE_INCREASE,
                self.cwnd * multiplier,
            )
            next_cwnd = min(desired_cwnd, self.cwnd + DQN_MAX_INCREASE_PER_STEP)
        else:
            next_cwnd = self.cwnd * multiplier
        return min(self.max_cwnd, max(1.0, next_cwnd))

    def _continuous_state(self, srtt: float | None) -> tuple[float, ...]:
        avg_rtt = (
            self.interval_rtt_sum / self.interval_rtt_count
            if self.interval_rtt_count
            else (srtt or self.last_srtt or 0.0)
        )
        if self.last_srtt and self.last_srtt > 0:
            rtt_trend_percent = ((avg_rtt - self.last_srtt) / self.last_srtt) * 100.0
        else:
            rtt_trend_percent = 0.0
        total_events = self.interval_acked + self.interval_losses
        timeout_percent = (
            (self.interval_timeouts / total_events) * 100.0
            if total_events > 0
            else 0.0
        )
        ack_ratio = self.interval_acked / max(1.0, self.cwnd)
        return (
            float(avg_rtt * 1000.0),
            float(rtt_trend_percent),
            float(self._interval_loss_rate() * 100.0),
            float(timeout_percent),
            float(self.cwnd),
            float(ack_ratio),
        )

    def _normalized_dqn_state(self, state: tuple[float, ...]):
        rtt_ms, rtt_trend_pct, loss_pct, timeout_pct, cwnd, ack_ratio = state
        return self.torch.tensor(
            [
                min(rtt_ms / 1000.0, 10.0),
                min(max(rtt_trend_pct / 100.0, -1.0), 1.0),
                min(max(loss_pct / 100.0, 0.0), 1.0),
                min(max(timeout_pct / 100.0, 0.0), 1.0),
                min(max((cwnd / max(1.0, self.max_cwnd)) ** 0.5, 0.0), 1.0),
                min(max(ack_ratio / 2.0, 0.0), 1.0),
            ],
            dtype=self.torch.float32,
        )

    def _interval_loss_rate(self) -> float:
        total = self.interval_acked + self.interval_losses
        if total <= 0:
            return 0.0
        return self.interval_losses / total

    def _choose_dqn_action(self, state: tuple[float, ...]) -> int:
        if not self.dqn_eval and random.random() < self.epsilon:
            return random.randrange(len(DQN_ACTION_MULTIPLIERS))
        with self.torch.no_grad():
            q_values = self.dqn_policy(self._normalized_dqn_state(state).unsqueeze(0))
        return int(self.torch.argmax(q_values, dim=1).item())

    def _train_dqn_batch(self) -> None:
        if len(self.dqn_replay) < self.dqn_batch_size:
            return
        batch = random.sample(self.dqn_replay, self.dqn_batch_size)
        states, actions, rewards, next_states = zip(*batch)
        state_tensor = self.torch.stack([self._normalized_dqn_state(state) for state in states])
        action_tensor = self.torch.tensor(actions, dtype=self.torch.int64).unsqueeze(1)
        reward_tensor = self.torch.tensor(rewards, dtype=self.torch.float32)
        next_tensor = self.torch.stack(
            [self._normalized_dqn_state(state) for state in next_states]
        )

        q_values = self.dqn_policy(state_tensor).gather(1, action_tensor).squeeze(1)
        with self.torch.no_grad():
            next_best = self.dqn_target(next_tensor).max(1).values
            target = reward_tensor + self.gamma * next_best
        loss = self.dqn_loss_fn(q_values, target)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

    def _init_dqn(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.append(user_site)
            try:
                import torch
                import torch.nn as nn
            except ImportError:
                raise SystemExit(
                    "DQN mode requires PyTorch. Install it first, for example: "
                    "python -m pip install torch"
                ) from exc

        class DQNNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(len(DQN_STATE_FEATURES), 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, len(DQN_ACTION_MULTIPLIERS)),
                )

            def forward(self, x):
                return self.net(x)

        self.torch = torch
        self.dqn_policy = DQNNet()
        self.dqn_target = DQNNet()
        self.dqn_optimizer = torch.optim.Adam(self.dqn_policy.parameters(), lr=self.dqn_lr)
        self.dqn_loss_fn = nn.SmoothL1Loss()
        if self.dqn_model_file and self.dqn_model_file.exists():
            checkpoint = torch.load(self.dqn_model_file, map_location="cpu")
            if self._dqn_checkpoint_compatible(checkpoint):
                try:
                    self.dqn_policy.load_state_dict(checkpoint["policy_state_dict"])
                    if "optimizer_state_dict" in checkpoint and not self.dqn_eval:
                        self.dqn_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    self.dqn_steps = int(checkpoint.get("steps", 0))
                except RuntimeError as exc:
                    if self.verbose:
                        print(f"[SENDER][DQN] ignore incompatible model weights: {exc}", flush=True)
            elif self.verbose:
                print(
                    f"[SENDER][DQN] ignore old model architecture: {self.dqn_model_file}",
                    flush=True,
                )
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_target.eval()

    def _dqn_checkpoint_compatible(self, checkpoint: object) -> bool:
        if not isinstance(checkpoint, dict) or "policy_state_dict" not in checkpoint:
            return False
        state_features = checkpoint.get("state_features")
        if list(state_features or []) != list(DQN_STATE_FEATURES):
            return False
        action_multipliers = checkpoint.get("action_multipliers", checkpoint.get("actions"))
        try:
            loaded_actions = tuple(float(value) for value in action_multipliers)
        except (TypeError, ValueError):
            return False
        return loaded_actions == DQN_ACTION_MULTIPLIERS

    def _save_dqn(self) -> None:
        if self.dqn_eval or self.dqn_model_file is None or self.dqn_policy is None:
            return
        self.dqn_model_file.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "policy_state_dict": self.dqn_policy.state_dict(),
                "optimizer_state_dict": self.dqn_optimizer.state_dict(),
                "steps": self.dqn_steps,
                "action_multipliers": DQN_ACTION_MULTIPLIERS,
                "action_names": DQN_ACTION_NAMES,
                "state_features": DQN_STATE_FEATURES,
                "max_cwnd": self.max_cwnd,
            },
            self.dqn_model_file,
        )

    def _reset_interval(self) -> None:
        self.interval_acked = 0
        self.interval_losses = 0
        self.interval_retransmissions = 0
        self.interval_timeouts = 0
        self.interval_fast_retransmissions = 0
        self.interval_rtt_sum = 0.0
        self.interval_rtt_count = 0

    def _load_q_table(self) -> None:
        if self.mode != "qlearning" or self.qtable_file is None or not self.qtable_file.exists():
            return
        try:
            data = json.loads(self.qtable_file.read_text(encoding="utf-8"))
            rows = self._q_table_rows_from_json(data)
            if rows is not None:
                self.q_table = rows
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            if self.verbose:
                print("[SENDER][QLEARN] ignore invalid q-table file", flush=True)

    def _q_table_rows_from_json(self, data: object) -> list[list[float]] | None:
        if not isinstance(data, dict):
            return None

        rows = data.get("q_table")
        if isinstance(rows, list):
            parsed_rows = self._parse_q_rows(rows)
            if parsed_rows is None:
                return None
            if len(parsed_rows) == len(Q_STATE_NAMES):
                return parsed_rows
            if len(parsed_rows) == len(Q_EXPANDED_STATE_NAMES):
                return self._collapse_expanded_q_rows(parsed_rows)

        parsed_rows = []
        for state_name in Q_STATE_NAMES:
            state_values = data.get(state_name)
            if not isinstance(state_values, dict):
                parsed_rows = []
                break
            parsed_rows.append(
                [
                    float(state_values.get(action_key, 0.0))
                    for action_key in Q_ACTION_KEYS
                ]
            )
        if parsed_rows:
            return parsed_rows

        expanded_rows = []
        for state_name in Q_EXPANDED_STATE_NAMES:
            state_values = data.get(state_name)
            if not isinstance(state_values, dict):
                return None
            expanded_rows.append(
                [
                    float(state_values.get(action_key, 0.0))
                    for action_key in Q_ACTION_KEYS
                ]
            )
        return self._collapse_expanded_q_rows(expanded_rows)

    def _parse_q_rows(self, rows: list[object]) -> list[list[float]] | None:
        parsed_rows = []
        for row in rows:
            if not isinstance(row, list) or len(row) != len(Q_ACTION_KEYS):
                return None
            parsed_rows.append([float(value) for value in row])
        return parsed_rows

    def _collapse_expanded_q_rows(self, expanded_rows: list[list[float]]) -> list[list[float]]:
        rows = []
        bucket_count = len(Q_EXPANDED_CWND_BUCKETS)
        for state_index in range(len(Q_STATE_NAMES)):
            start = state_index * bucket_count
            bucket_rows = expanded_rows[start : start + bucket_count]
            rows.append(
                [
                    sum(row[action_index] for row in bucket_rows) / bucket_count
                    for action_index in range(len(Q_ACTION_KEYS))
                ]
            )
        return rows


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
        link_bandwidth_drop_after_packets: int | None = None,
        link_bandwidth_drop_factor: float = 0.5,
        cc_mode: str = "fixed",
        max_cwnd: float = 100.0,
        epsilon: float = 0.10,
        q_alpha: float = 0.30,
        q_gamma: float = 0.85,
        reward_throughput_weight: float = 1.0,
        reward_rtt_weight: float = 0.015,
        reward_timeout_weight: float = 10.0,
        reward_retx_weight: float = 2.0,
        reward_cwnd_weight: float = 0.0,
        reward_target_rtt_ms: float | None = None,
        qtable_file: str | None = "artifacts/models/active/q_table.json",
        dqn_model_file: str | None = "artifacts/models/active/dqn_model.pt",
        dqn_lr: float = 0.001,
        dqn_batch_size: int = 16,
        dqn_replay_capacity: int = 1024,
        dqn_target_update: int = 10,
        dqn_eval: bool = False,
        q_eval: bool = False,
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
        self.link_bandwidth_drop_after_packets = link_bandwidth_drop_after_packets
        self.link_bandwidth_drop_factor = link_bandwidth_drop_factor
        initial_cwnd = 1.0 if cc_mode == "aimd" else self.window_size
        self.controller = CongestionController(
            mode=cc_mode,
            initial_cwnd=initial_cwnd,
            max_cwnd=max_cwnd,
            epsilon=epsilon,
            alpha=q_alpha,
            gamma=q_gamma,
            reward_throughput_weight=reward_throughput_weight,
            reward_rtt_weight=reward_rtt_weight,
            reward_timeout_weight=reward_timeout_weight,
            reward_retx_weight=reward_retx_weight,
            reward_cwnd_weight=reward_cwnd_weight,
            reward_target_rtt_ms=reward_target_rtt_ms,
            qtable_file=qtable_file,
            dqn_model_file=dqn_model_file,
            dqn_lr=dqn_lr,
            dqn_batch_size=dqn_batch_size,
            dqn_replay_capacity=dqn_replay_capacity,
            dqn_target_update=dqn_target_update,
            dqn_eval=dqn_eval,
            q_eval=q_eval,
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
                bandwidth_drop_after_packets=self.link_bandwidth_drop_after_packets,
                bandwidth_drop_factor=self.link_bandwidth_drop_factor,
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
        self.controller.on_loss(reason=reason)
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
        mode, payload, reward, detail = result
        if mode == "qlearning":
            state, action = payload
            self._log(
                "QLEARN",
                "state={state} action={action}({action_name}) reward={reward:.3f} cwnd={cwnd:.2f}".format(
                    state=Q_STATE_NAMES[state],
                    action=action,
                    action_name=Q_ACTION_NAMES[action],
                    reward=reward,
                    cwnd=self.controller.cwnd,
                ),
            )
            return
        self._log("DQN", f"{detail} reward={reward:.3f}")

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
        "--cc",
        dest="cc_mode",
        choices=("fixed", "aimd", "qlearning", "q-learning", "dqn"),
        default="fixed",
        help="congestion control mode: fixed, AIMD, Q-Learning, or DQN",
    )
    parser.add_argument("--max-cwnd", "--max-window", dest="max_cwnd", type=float, default=100.0)
    parser.add_argument("--epsilon", "--q-epsilon", dest="epsilon", type=float, default=0.10)
    parser.add_argument("--q-alpha", type=float, default=0.30)
    parser.add_argument("--q-gamma", type=float, default=0.85)
    parser.add_argument(
        "--reward-throughput-weight",
        "--reward-ack-weight",
        dest="reward_throughput_weight",
        type=float,
        default=1.0,
        help="reward weight for packets ACKed in one control interval",
    )
    parser.add_argument(
        "--reward-timeout-weight",
        type=float,
        default=10.0,
        help="reward penalty per RTO timeout event",
    )
    parser.add_argument(
        "--reward-retx-weight",
        "--reward-loss-weight",
        dest="reward_retx_weight",
        type=float,
        default=2.0,
        help="reward penalty per retransmission event",
    )
    parser.add_argument(
        "--reward-rtt-weight",
        type=float,
        default=0.015,
        help="reward penalty per millisecond of RTT, or excess RTT when --reward-target-rtt-ms is set",
    )
    parser.add_argument(
        "--reward-target-rtt-ms",
        type=float,
        default=None,
        help="only penalize RTT above this target; unset penalizes total average RTT",
    )
    parser.add_argument(
        "--reward-cwnd-weight",
        type=float,
        default=0.0,
        help="reward bonus per unit of CWND utilization (cwnd/max_cwnd); encourages DQN to avoid lazy policies",
    )
    parser.add_argument("--qtable-file", "--q-table", dest="qtable_file", default="artifacts/models/active/q_table.json")
    parser.add_argument(
        "--q-eval",
        action="store_true",
        help="run Q-Learning in evaluation mode: greedy actions without Bellman updates or Q-table overwrite",
    )
    parser.add_argument("--dqn-model-file", default="artifacts/models/active/dqn_model.pt")
    parser.add_argument("--dqn-lr", type=float, default=0.001)
    parser.add_argument("--dqn-batch-size", type=int, default=16)
    parser.add_argument("--dqn-replay-capacity", type=int, default=1024)
    parser.add_argument("--dqn-target-update", type=int, default=10)
    parser.add_argument(
        "--dqn-eval",
        action="store_true",
        help="run DQN in evaluation mode: no random exploration, online training, or model overwrite",
    )
    parser.add_argument("--metrics-file", default="artifacts/metrics/metrics.csv")
    parser.add_argument("--history-file", default="artifacts/metrics/history.csv")
    parser.add_argument("--plot-file", default=None)
    parser.add_argument("--rto", type=float, default=0.20)
    parser.add_argument("--link-queue-capacity", type=int, default=20)
    parser.add_argument("--link-service-delay-ms", type=float, default=10.0)
    parser.add_argument("--link-bandwidth-drop-after-packets", type=int, default=0)
    parser.add_argument("--link-bandwidth-drop-factor", type=float, default=0.5)
    parser.add_argument("--disable-virtual-link", action="store_true")
    parser.add_argument("--min-window", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--q-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-rounds", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-epsilon-decay", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-min-epsilon", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--reward-alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reward-beta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reward-gamma", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reward-delta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reward-queue-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cc_mode == "q-learning":
        args.cc_mode = "qlearning"
    if args.q_seed is not None:
        random.seed(args.q_seed)
    if args.packets < 0:
        raise SystemExit("--packets must be non-negative")
    if args.start_seq < 0:
        raise SystemExit("--start-seq must be non-negative")
    if args.start_seq + max(args.packets - 1, 0) > 2_147_483_647:
        raise SystemExit("--start-seq + --packets exceeds signed ACK range")
    if args.repeat_rounds <= 0:
        raise SystemExit("--repeat-rounds must be positive")
    if not 0 < args.repeat_epsilon_decay <= 1:
        raise SystemExit("--repeat-epsilon-decay must be in (0, 1]")
    if not 0 <= args.repeat_min_epsilon <= 1:
        raise SystemExit("--repeat-min-epsilon must be in [0, 1]")
    final_start_seq = args.start_seq + (args.repeat_rounds - 1) * args.packets
    if final_start_seq + max(args.packets - 1, 0) > 2_147_483_647:
        raise SystemExit("repeated sequence range exceeds signed ACK range")
    if args.local_port + args.repeat_rounds - 1 > 65535:
        raise SystemExit("repeated local port range exceeds 65535")
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
    if args.reward_alpha is not None:
        args.reward_throughput_weight = args.reward_alpha
    if args.reward_beta is not None:
        args.reward_timeout_weight = args.reward_beta
    if args.reward_gamma is not None:
        args.reward_retx_weight = args.reward_gamma
    if args.reward_delta is not None:
        args.reward_rtt_weight = args.reward_delta
    if args.reward_throughput_weight < 0:
        raise SystemExit("--reward-throughput-weight must be non-negative")
    if args.reward_timeout_weight < 0:
        raise SystemExit("--reward-timeout-weight must be non-negative")
    if args.reward_retx_weight < 0:
        raise SystemExit("--reward-retx-weight must be non-negative")
    if args.reward_rtt_weight < 0:
        raise SystemExit("--reward-rtt-weight must be non-negative")
    if args.reward_cwnd_weight < 0:
        raise SystemExit("--reward-cwnd-weight must be non-negative")
    if args.reward_target_rtt_ms is not None and args.reward_target_rtt_ms < 0:
        raise SystemExit("--reward-target-rtt-ms must be non-negative")
    if args.dqn_lr <= 0:
        raise SystemExit("--dqn-lr must be positive")
    if args.dqn_batch_size <= 0:
        raise SystemExit("--dqn-batch-size must be positive")
    if args.dqn_replay_capacity <= 0:
        raise SystemExit("--dqn-replay-capacity must be positive")
    if args.dqn_target_update <= 0:
        raise SystemExit("--dqn-target-update must be positive")
    if args.link_queue_capacity <= 0:
        raise SystemExit("--link-queue-capacity must be positive")
    if args.link_service_delay_ms < 0:
        raise SystemExit("--link-service-delay-ms must be non-negative")
    if args.link_bandwidth_drop_after_packets < 0:
        raise SystemExit("--link-bandwidth-drop-after-packets must be non-negative")
    if not 0 < args.link_bandwidth_drop_factor <= 1:
        raise SystemExit("--link-bandwidth-drop-factor must be in (0, 1]")

    for round_index in range(args.repeat_rounds):
        epsilon = max(
            args.repeat_min_epsilon,
            args.epsilon * (args.repeat_epsilon_decay ** round_index),
        )
        sender = ReliableSender(
            target_host=args.target_host,
            target_port=args.target_port,
            local_host=args.local_host,
            local_port=args.local_port + round_index,
            total_packets=args.packets,
            window_size=args.window_size,
            rto=args.rto,
            verbose=not args.quiet,
            start_seq=args.start_seq + round_index * args.packets,
            use_virtual_link=not args.disable_virtual_link,
            link_queue_capacity=args.link_queue_capacity,
            link_service_delay_ms=args.link_service_delay_ms,
            link_bandwidth_drop_after_packets=(
                args.link_bandwidth_drop_after_packets or None
            ),
            link_bandwidth_drop_factor=args.link_bandwidth_drop_factor,
            cc_mode=args.cc_mode,
            max_cwnd=args.max_cwnd,
            epsilon=epsilon,
            q_alpha=args.q_alpha,
            q_gamma=args.q_gamma,
            reward_throughput_weight=args.reward_throughput_weight,
            reward_rtt_weight=args.reward_rtt_weight,
            reward_timeout_weight=args.reward_timeout_weight,
            reward_retx_weight=args.reward_retx_weight,
            reward_cwnd_weight=args.reward_cwnd_weight,
            reward_target_rtt_ms=args.reward_target_rtt_ms,
            qtable_file=args.qtable_file,
            dqn_model_file=args.dqn_model_file,
            dqn_lr=args.dqn_lr,
            dqn_batch_size=args.dqn_batch_size,
            dqn_replay_capacity=args.dqn_replay_capacity,
            dqn_target_update=args.dqn_target_update,
            dqn_eval=args.dqn_eval,
            q_eval=args.q_eval,
            metrics_file=args.metrics_file,
            history_file=args.history_file,
            plot_file=args.plot_file if round_index == args.repeat_rounds - 1 else None,
        )
        sender.run()


if __name__ == "__main__":
    main()
