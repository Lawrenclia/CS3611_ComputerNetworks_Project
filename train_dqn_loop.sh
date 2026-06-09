#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT=9001
PACKETS=240
ROUNDS=100
INIT_EPSILON=0.45
EPSILON_DECAY=0.97
MIN_EPSILON=0.03
MODEL="$ROOT/artifacts/models/active/dqn_model.pt"
BACKUP_DIR="$ROOT/artifacts/models/backups"

mkdir -p "$BACKUP_DIR"
if [ -f "$MODEL" ]; then
    BACKUP="$BACKUP_DIR/dqn_model_backup_$(date +%Y%m%d-%H%M%S).pt"
    cp "$MODEL" "$BACKUP"
    rm "$MODEL"
    echo "[LOOP] Backed up old DQN model to $BACKUP"
fi

echo "[LOOP] Starting receiver on port $PORT..."
python3 "$ROOT/receiver.py" --port $PORT --initial-seq 0 --loss-rate 0.04 --delay-ms 20 --jitter-ms 10 --seed 1 &
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
        --window-size 8 \
        --max-cwnd 80 \
        --epsilon $EPSILON \
        --q-gamma 0.90 \
        --rto 0.2 \
        --dqn-model-file "$MODEL" \
        --dqn-lr 0.0007 \
        --dqn-batch-size 32 \
        --dqn-replay-capacity 4096 \
        --dqn-target-update 20 \
        --reward-throughput-weight 2.4 \
        --reward-timeout-weight 7.0 \
        --reward-retx-weight 1.2 \
        --reward-rtt-weight 0.006 \
        --metrics-file "$ROOT/artifacts/training/dqn_metrics.csv" \
        --history-file "$ROOT/artifacts/training/dqn_history.csv" \
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
