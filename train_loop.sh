#!/bin/bash
# Q-Learning 100 rounds training loop
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT=9001
PACKETS=120
ROUNDS=100
INIT_EPSILON=0.3
EPSILON_DECAY=0.95
MIN_EPSILON=0.05

echo "[LOOP] Starting receiver on port $PORT..."
python3 "$ROOT/receiver.py" --port $PORT --initial-seq 0 &
RECEIVER_PID=$!
trap 'kill $RECEIVER_PID 2>/dev/null; wait $RECEIVER_PID 2>/dev/null' EXIT INT TERM
sleep 1

echo "[LOOP] Starting $ROUNDS rounds of Q-Learning training..."
for ((i=0; i<ROUNDS; i++)); do
    EPSILON=$(python3 -c "print(max($MIN_EPSILON, $INIT_EPSILON * ($EPSILON_DECAY ** $i)))")
    START_SEQ=$((i * PACKETS))
    echo "[LOOP] Round $((i+1))/$ROUNDS start_seq=$START_SEQ epsilon=$EPSILON"
    python3 "$ROOT/sender.py" \
        --target-port $PORT \
        --packets $PACKETS \
        --start-seq $START_SEQ \
        --cc-mode qlearning \
        --window-size 4 \
        --max-cwnd 32 \
        --epsilon $EPSILON \
        --rto 0.2 \
        --qtable-file "$ROOT/artifacts/models/active/q_table.json" \
        --metrics-file "$ROOT/artifacts/training/qlearning_metrics.csv" \
        --history-file "$ROOT/artifacts/training/qlearning_history.csv" \
        --quiet
    if [ $? -ne 0 ]; then
        echo "[LOOP] Round $((i+1)) FAILED!"
        kill $RECEIVER_PID 2>/dev/null
        wait $RECEIVER_PID 2>/dev/null
        exit 1
    fi
done

echo "[LOOP] Q-Learning training complete. Killing receiver..."
kill $RECEIVER_PID 2>/dev/null
wait $RECEIVER_PID 2>/dev/null
echo "[LOOP] Done!"
