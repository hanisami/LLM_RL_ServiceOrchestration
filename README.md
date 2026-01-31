# LLM + RL Service Orchestration under Demand Shift

Code and experiment artifacts for our paper on **LLM-guided reinforcement learning (RL)** for **service orchestration under demand shifts** (non-stationary workloads, changing resource availability, and fluctuating QoS/SLO constraints).

---

## 1) What this repository does

We study orchestration decisions (e.g., **placement**, **scaling**, **consolidation**, **admission**, **migration**) in an **edge/cloud** setting where **demand shifts** cause non-stationarity.

At each discrete time step *t*:

- **Telemetry → Context encoder:** raw measurements (utilization/QoS/SLA signals) are mapped into a compact state vector `s_t = f_enc(o_t)` capturing current conditions + temporal dynamics (trend/volatility/risk).
- **RL policy:** chooses an orchestration action `a_t`.
- **LLM-guidance (optional / configurable):** used as a structured advisor for *risk-aware orchestration*, generating constraints, candidate actions, or rationale signals that can be integrated into reward shaping / action pruning / safety checks.
- **Reward:** combines SLO/SLA adherence, provisioning cost, backlog, and penalties for excessive orchestration churn.

The code is organized so you can:
- run **training** and **evaluation** under workload changes,
- reproduce plots/tables in the paper,
- swap **LLM provider** or disable LLM guidance to compare baselines.

