## Bandwidth Halving Recovery: AIMD vs Q-Learning

When bandwidth is suddenly halved mid-transmission (t=2.53s, after
150/300 packets):

**1. AIMD overshoots.** Before halving, CWND reached
7.5 pkts, fully saturating the queue.
The inflated window floods the bottleneck after the drop, causing
72 retransmissions and
46 timeouts post-halving. AIMD only
reduces CWND after loss/timeout — a *reactive* signal — delaying
recovery to 3.72s.

**2. Q-Learning adapts proactively.** Pre-halving CWND of
5.2 is lower than AIMD's because the
policy learned to *hold* on rising RTT, preserving queue headroom.
After halving, adaptation (epsilon boosted to 0.15, alpha raised to
0.15, moderately amplified loss/RTT penalties) causes the agent to unlearn
large-window preferences. Recovery: 1.98s
(comparable vs AIMD).

**3. Post-halving comparison:**
- Throughput: Q-Learning -0.051 Mbps vs AIMD
- RTT: Q-Learning +42.5 ms vs AIMD
- Retransmissions: Q-Learning fewer by 57
- Recovery: 1.98s (QL) vs 3.72s (AIMD)

**4. Key insight:** AIMD relies on packet loss as a *lagging* congestion
signal. Q-Learning uses RTT trends as an *early* signal, allowing it to
hold before loss. With post-halving adaptation, it converges to the new
lower optimal CWND with fewer collateral losses.
