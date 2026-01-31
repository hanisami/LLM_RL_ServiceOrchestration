from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from context_provider import LLMContext


def discretize(x: float, bins: List[float]) -> int:
    for i, th in enumerate(bins):
        if x < th:
            return i
    return len(bins)


def encode_state(obs: Dict[str, float], context: Optional[LLMContext], use_llm: bool) -> Tuple:
    cpu = float(obs.get("cpu_util", 0.0))
    mem = float(obs.get("mem_util", 0.0))
    backlog = float(obs.get("backlog", 0.0))
    slo = float(obs.get("slo_miss", 0.0))
    hosts = float(obs.get("active_hosts", 0.0))
    pending = float(obs.get("pending_scale", 0.0))

    cpu_b = discretize(cpu, [0.4, 0.7, 0.85])
    mem_b = discretize(mem, [0.4, 0.7, 0.85])
    backlog_b = discretize(backlog, [0.05, 0.20, 0.40, 0.70])
    slo_b = discretize(slo, [0.05, 0.15, 0.30, 0.60])
    hosts_b = discretize(hosts, [0.25, 0.50, 0.75])
    pending_b = discretize(pending, [-0.5, -0.1, 0.1, 0.5])

    if (not use_llm) or (context is None):
        return (cpu_b, mem_b, backlog_b, slo_b, hosts_b, pending_b)

    # Context adds *history-derived* signals (trend/volatility/risk) that are not
    # visible from a single observation snapshot.
    dom_id = {"cpu": 0, "mem": 1, "balanced": 2}.get(context.dominant_resource, 2)
    outlook_id = {"decreasing": 0, "stable": 1, "increasing": 2, "spike_likely": 3}.get(context.demand_outlook, 1)

    sla_risk = 1 if "sla_risk" in (context.risk_flags or []) else 0
    instab_risk = 1 if "instability_risk" in (context.risk_flags or []) else 0

    return (cpu_b, mem_b, backlog_b, slo_b, hosts_b, pending_b, dom_id, outlook_id, sla_risk, instab_risk)
