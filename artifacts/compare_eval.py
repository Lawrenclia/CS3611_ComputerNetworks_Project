"""Throwaway A/B harness: AIMD vs Q-Learning vs DQN under identical conditions.

Runs each controller in evaluation mode against a fresh receiver and reports
throughput / RTT / retransmissions plus CWND-dynamics stats so we can confirm
the DQN window actually varies and beats the baselines. Not part of the project.
"""
from __future__ import annotations
import csv, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DQN_MODEL = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "artifacts/models/active/dqn_smoke.pt")
SEED = sys.argv[2] if len(sys.argv) > 2 else "7"
PACKETS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
COND = dict(loss="0.02", delay="10", jitter="3", rto="0.20", maxcwnd="64", service="10", queue="20", seed=SEED)
OUT = ROOT / "artifacts/training/cmp_metrics.csv"
HIST = ROOT / "artifacts/training/cmp_history.csv"
for f in (OUT, HIST):
    f.unlink(missing_ok=True)

def run(mode, extra, rport, sport):
    rcv = subprocess.Popen([sys.executable, str(ROOT/"receiver.py"), "--port", str(rport),
        "--initial-seq", "0", "--loss-rate", COND["loss"], "--delay-ms", COND["delay"],
        "--jitter-ms", COND["jitter"], "--seed", COND["seed"]], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(0.4)
    try:
        cmd = [sys.executable, str(ROOT/"sender.py"), "--target-port", str(rport),
            "--local-port", str(sport), "--packets", str(PACKETS), "--rto", COND["rto"],
            "--cc-mode", mode, "--max-cwnd", COND["maxcwnd"], "--link-service-delay-ms", COND["service"],
            "--link-queue-capacity", COND["queue"], "--metrics-file", str(OUT),
            "--history-file", str(HIST), "--quiet", *extra]
        subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        rcv.terminate()
        try: rcv.wait(timeout=2)
        except subprocess.TimeoutExpired: rcv.kill()

run("aimd", ["--window-size", "1"], 18001, 18000)
run("qlearning", ["--window-size", "1", "--epsilon", "0.0", "--q-eval",
    "--qtable-file", str(ROOT/"artifacts/models/active/q_table.json")], 18011, 18010)
run("dqn", ["--window-size", "8", "--epsilon", "0.0", "--dqn-eval",
    "--dqn-model-file", DQN_MODEL], 18021, 18020)

rows = list(csv.DictReader(OUT.open(encoding="utf-8")))
hist = list(csv.DictReader(HIST.open(encoding="utf-8")))
print(f"\n{'mode':>10} {'thrpt_Mbps':>11} {'avg_rtt_ms':>11} {'retx':>6} {'timeouts':>9}  cwnd[min/mean/max/std/uniq]")
for r in rows:
    m = r["mode"]
    cw = [float(h["cwnd"]) for h in hist if h["run_id"] == r["run_id"]]
    if cw:
        stats = f"{min(cw):.1f}/{statistics.mean(cw):.1f}/{max(cw):.1f}/{statistics.pstdev(cw):.2f}/{len(set(round(x,1) for x in cw))}"
    else:
        stats = "n/a"
    print(f"{m:>10} {float(r['throughput_mbps']):>11.3f} {float(r['avg_rtt_ms']):>11.2f} {r['retransmissions']:>6} {r['timeout_events']:>9}  {stats}")
print(f"\nthroughput ceiling = 0.819 Mbps")
