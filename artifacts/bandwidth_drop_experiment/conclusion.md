## Bandwidth Halving Recovery: AIMD vs Q-Learning

When bandwidth is suddenly halved mid-transmission (t=2.00s, after
150/300 packets):

**1. AIMD overshoots.** Before halving, CWND reached
8.5 pkts, fully saturating the queue.
The inflated window floods the bottleneck after the drop, causing
32 retransmissions and
22 timeouts post-halving. AIMD only
reduces CWND after loss/timeout — a *reactive* signal — delaying
recovery to 0.11s.

**2. Q-Learning adapts proactively.** Pre-halving CWND of
3.8 is lower than AIMD's because the
policy learned to *hold* on rising RTT, preserving queue headroom.
After halving, adaptation (epsilon boosted to 0.15, alpha raised to
0.15, moderately amplified loss/RTT penalties) causes the agent to unlearn
large-window preferences. Recovery: 0.11s
(comparable vs AIMD).

**3. Post-halving comparison:**
- Throughput: Q-Learning +0.354 Mbps vs AIMD
- RTT: Q-Learning -39.6 ms vs AIMD
- Retransmissions: Q-Learning fewer by 23
- Recovery: 0.11s (QL) vs 0.11s (AIMD)

**4. Key insight:** AIMD relies on packet loss as a *lagging* congestion
signal. Q-Learning uses RTT trends as an *early* signal, allowing it to
hold before loss. With post-halving adaptation, it converges to the new
lower optimal CWND with fewer collateral losses.
