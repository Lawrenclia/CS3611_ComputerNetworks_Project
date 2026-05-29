#!/bin/bash
# AIMD/Reno baseline benchmark
python3 receiver.py --port 9205 --loss-rate 0.08 --delay-ms 20 --jitter-ms 10 --seed 1 > rx1.log 2>&1 &
RX1_PID=$!
sleep 0.5
python3 sender.py --target-port 9205 --local-port 9305 --packets 120 --window-size 1 --rto 0.20 --cc-mode aimd --max-cwnd 32 > tx1.log 2>&1
kill $RX1_PID

# Q-Learning benchmark (using learned table, epsilon=0 for pure exploitation)
python3 receiver.py --port 9206 --loss-rate 0.08 --delay-ms 20 --jitter-ms 10 --seed 1 > rx2.log 2>&1 &
RX2_PID=$!
sleep 0.5
python3 sender.py --target-port 9206 --local-port 9306 --packets 120 --window-size 8 --rto 0.20 --cc-mode qlearning --epsilon 0.0 --qtable-file q_table.json > tx2.log 2>&1
kill $RX2_PID

echo "=== AIMD Baseline ==="
grep "\[DONE\]" tx1.log
echo "=== Q-Learning (Epsilon 0) ==="
grep "\[DONE\]" tx2.log
