#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT=9001
PACKETS=120
ROUNDS=100
INIT_EPSILON=0.5
EPSILON_DECAY=0.96
MIN_EPSILON=0.05

echo "[LOOP] Starting receiver on port $PORT..."
python3 "$ROOT/receiver.py" --port $PORT --initial-seq 0 &
RECEIVER_PID=$!
trap 'kill $RECEIVER_PID 2>/dev/null; wait $RECEIVER_PID 2>/dev/null' EXIT INT TERM
sleep 1

echo "[LOOP] Starting $ROUNDS rounds of DQN training..."
for ((i=0; i<ROUNDS; i++)); do
    EPSILON=$(python3 -c "print(max($MIN_EPSILON, $INIT_EPSILON * ($EPSILON_DECAY ** $i)))")
    START_SEQ=$((i * PACKETS))
    echo "[LOOP] Round $((i+1))/$ROUNDS start_seq=$START_SEQ epsilon=$EPSILON"
    python3 "$ROOT/sender.py" \
        --target-port $PORT \
        --packets $PACKETS \
        --start-seq $START_SEQ \
        --cc-mode dqn \
        --window-size 4 \
        --max-cwnd 32 \
        --epsilon $EPSILON \
        --rto 0.2 \
        --dqn-model-file "$ROOT/dqn_model.pt" \
        --dqn-lr 0.0005 \
        --dqn-batch-size 32 \
        --dqn-replay-capacity 2048 \
        --metrics-file "$ROOT/metrics.csv" \
        --history-file "$ROOT/history.csv" \
        --quiet
    if [ $? -ne 0 ]; then
        echo "[LOOP] Round $((i+1)) FAILED!"
        kill $RECEIVER_PID 2>/dev/null
        wait $RECEIVER_PID 2>/dev/null
        exit 1
    fi
done

echo "[LOOP] DQN training complete. Killing receiver..."
kill $RECEIVER_PID 2>/dev/null
wait $RECEIVER_PID 2>/dev/null
echo "[LOOP] Done!"
