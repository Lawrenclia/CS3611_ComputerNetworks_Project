# UDP Reliable Transport Demo

This project implements question 5 of the CS3611 computer networks assignment:

- UDP application-layer reliable transport with ACK and timeout retransmission.
- A virtual bottleneck link with finite queue and packet drops.
- AIMD congestion control as the baseline.
- Q-learning congestion control with 6 discrete states and 3 actions.
- Fast retransmit on 3 duplicate cumulative ACKs, plus receiver-side out-of-order buffering.
- SVG/CSV result export for CWND, throughput, and RTT comparison.

## Files

- `receiver.py`: receives data packets and immediately returns ACKs.
- `sender.py`: runs AIMD and Q-learning experiments and saves outputs.
- `virtual_link.py`: shapes traffic with a token-like dequeue interval and bounded queue.
- `congestion.py`: AIMD and Q-learning controllers.
- `protocol.py`: packet formats and helpers.
- `visualize.py`: exports CSV and SVG figures, and can also generate PNG plots when matplotlib is installed.

## Run

Start the receiver in one terminal:

```bash
python3 receiver.py --host 127.0.0.1 --port 9001 --initial-seq 0
```

Start the sender in another terminal:

```bash
python3 sender.py --target-host 127.0.0.1 --target-port 9001
```

Recommended environment setup:

```bash
python3 -m pip install -r requirements.txt
```

Outputs are written to `results/`:

- `comparison.png` (when matplotlib is installed)
- `comparison.svg`
- `metrics.csv`
- `summary.json`
- `q_table.json`

## Notes

- Keep the receiver running while the sender executes AIMD, Q-learning training episodes, and the final evaluation round.
- The receiver uses cumulative ACK semantics, so sequence numbers should be continuous across episodes in a single experiment.
- The assignment text writes `Sender.py` and `Receiver.py`; on many macOS setups, `sender.py` and `receiver.py` are case-insensitively equivalent.
