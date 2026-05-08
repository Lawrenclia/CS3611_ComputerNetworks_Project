#!/bin/bash
# Fixed window benchmark
/Users/lawrenclia/Documents/files/计算机网络/project/.venv/bin/python receiver.py --port 9205 --loss-rate 0.08 --delay-ms 20 --jitter-ms 10 --seed 1 > rx1.log 2>&1 &
RX1_PID=$!
sleep 0.5
/Users/lawrenclia/Documents/files/计算机网络/project/.venv/bin/python sender.py --target-port 9205 --local-port 9305 --packets 120 --window-size 8 --rto 0.20 --cc fixed > tx1.log 2>&1
kill $RX1_PID

# Q-Learning benchmark (using learned table, epsilon=0 for pure exploitation)
/Users/lawrenclia/Documents/files/计算机网络/project/.venv/bin/python receiver.py --port 9206 --loss-rate 0.08 --delay-ms 20 --jitter-ms 10 --seed 1 > rx2.log 2>&1 &
RX2_PID=$!
sleep 0.5
/Users/lawrenclia/Documents/files/计算机网络/project/.venv/bin/python sender.py --target-port 9206 --local-port 9306 --packets 120 --window-size 8 --rto 0.20 --cc q-learning --q-epsilon 0.0 --q-table q_table.json > tx2.log 2>&1
kill $RX2_PID

echo "=== Fixed Window ==="
grep "\[DONE\]" tx1.log
echo "=== Q-Learning (Epsilon 0) ==="
grep "\[DONE\]" tx2.log
