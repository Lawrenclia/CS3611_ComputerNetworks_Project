## Bandwidth Halving Recovery: AIMD vs Q-Learning

When bandwidth is suddenly halved mid-transmission (t=2.14s, after
150/300 packets):

**1. AIMD overshoots.** Before halving, CWND reached
8.5 pkts, fully saturating the queue.
The inflated window floods the bottleneck after the drop, causing
45 retransmissions and
29 timeouts post-halving. AIMD only
reduces CWND after loss/timeout — a *reactive* signal — delaying
recovery to 2.37s.

**2. Q-Learning adapts proactively.** Pre-halving CWND of
3.6 is lower than AIMD's because the
policy learned to *hold* on rising RTT, preserving queue headroom.
After halving, adaptation (epsilon boosted to 0.15, alpha raised to
0.15, moderately amplified loss/RTT penalties) causes the agent to unlearn
large-window preferences. Recovery: 0.20s
(comparable vs AIMD).

**3. Post-halving comparison:**
- Throughput: Q-Learning +0.385 Mbps vs AIMD
- RTT: Q-Learning -47.3 ms vs AIMD
- Retransmissions: Q-Learning fewer by 37
- Recovery: 0.20s (QL) vs 2.37s (AIMD)

**4. Key insight:** AIMD relies on packet loss as a *lagging* congestion
signal. Q-Learning uses RTT trends as an *early* signal, allowing it to
hold before loss. With post-halving adaptation, it converges to the new
lower optimal CWND with fewer collateral losses.
