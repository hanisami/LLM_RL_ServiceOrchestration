from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from workload_source import WorkloadSource


class NonStationaryEnv:
    """
    Trace-driven (or synthetic) capacity-control environment with provisioning delay.

    Action:
      0=consolidate (schedule -1 host)
      1=balanced (schedule 0)
      2=add_headroom (schedule +1 host)

    Key: scaling is delayed by `scale_delay` steps.
    """

    ACTIONS = {0: "consolidate", 1: "balanced", 2: "add_headroom"}

    def __init__(
        self,
        seed: int = 1,
        episode_len: int = 200,
        max_hosts: int = 12,
        init_hosts: int = 6,
        workload: Optional[WorkloadSource] = None,
        scale_delay: int = 10,
        w_slo: float = 15.0,
        w_cost: float = 0.5,
        w_backlog: float = 1.5,
        w_scale: float = 0.2,
    ):
        self.rng = random.Random(seed)
        self.episode_len = episode_len
        self.max_hosts = max_hosts
        self.init_hosts = init_hosts
        self.workload = workload

        self.scale_delay = max(0, int(scale_delay))
        self.pending: Deque[int] = deque([0] * self.scale_delay, maxlen=self.scale_delay)

        # state
        self.t = 0
        self.active_hosts = init_hosts
        self.backlog = 0.0
        self.cpu_util = 0.0
        self.mem_util = 0.0
        self.slo_miss = 0.0

        self.regimes: List[str] = []  # only used in dummy mode

        # reward weights
        self.w_slo = float(w_slo)
        self.w_cost = float(w_cost)
        self.w_backlog = float(w_backlog)
        self.w_scale = float(w_scale)

    def reset(self) -> Dict[str, float]:
        self.t = 0
        self.active_hosts = self.init_hosts
        self.backlog = 0.0

        if self.scale_delay > 0:
            self.pending = deque([0] * self.scale_delay, maxlen=self.scale_delay)

        if self.workload is not None:
            self.workload.reset(self.episode_len)
        else:
            phases = ["normal", "cpu_heavy", "mem_heavy", "bursty"]
            phase_len = max(1, self.episode_len // len(phases))
            self.regimes = []
            for p in phases:
                self.regimes += [p] * phase_len
            while len(self.regimes) < self.episode_len:
                self.regimes.append(phases[-1])

        self._update_telemetry(demand_cpu=0.0, demand_mem=0.0)
        return self._obs()

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}")

        # 1) apply any scheduled scaling that becomes effective now
        applied_delta = 0
        if self.scale_delay > 0:
            applied_delta = self.pending.popleft()
            self.active_hosts = int(max(1, min(self.max_hosts, self.active_hosts + applied_delta)))

        # 2) schedule new scaling decision (delayed), or apply immediately if delay=0
        scheduled_delta = 0
        if action == 0:
            scheduled_delta = -1
        elif action == 2:
            scheduled_delta = +1

        if self.scale_delay > 0:
            self.pending.append(scheduled_delta)
        else:
            self.active_hosts = int(max(1, min(self.max_hosts, self.active_hosts + scheduled_delta)))

        # 3) demand generation
        if self.workload is not None:
            cpu_u, mem_u = self.workload.get(self.t)
            demand_cpu = cpu_u * float(self.max_hosts)
            demand_mem = mem_u * float(self.max_hosts)
            regime = "trace"
        else:
            regime = self.regimes[self.t]
            demand_cpu, demand_mem = self._generate_demand(regime)

        # 4) capacity model + backlog
        cpu_cap = float(self.active_hosts)
        mem_cap = float(self.active_hosts)

        served_cpu = min(cpu_cap, demand_cpu + 0.5 * self.backlog)
        served_mem = min(mem_cap, demand_mem + 0.5 * self.backlog)

        unmet = max(0.0, (demand_cpu - served_cpu) + (demand_mem - served_mem))
        self.backlog = max(0.0, self.backlog * 0.85 + unmet)

        # 5) update telemetry + SLO proxy
        self._update_telemetry(demand_cpu=demand_cpu, demand_mem=demand_mem)

        # 6) reward
        scale_pen = abs(scheduled_delta)
        reward = (
            - self.w_slo * self.slo_miss
            - self.w_cost * (self.active_hosts / self.max_hosts)
            - self.w_backlog * self.backlog_norm()
            - self.w_scale * scale_pen
        )

        self.t += 1
        done = self.t >= self.episode_len

        info = {
            "regime_hidden": regime,
            "active_hosts": self.active_hosts,
            "pending_sum": int(sum(self.pending)) if self.scale_delay > 0 else 0,
            "applied_delta": int(applied_delta),
            "scheduled_delta": int(scheduled_delta),
            "backlog": float(self.backlog),
            "cpu_util": float(self.cpu_util),
            "mem_util": float(self.mem_util),
            "slo_miss": float(self.slo_miss),
        }
        return self._obs(), float(reward), done, info

    def backlog_norm(self) -> float:
        return min(1.0, self.backlog / 10.0)

    def _obs(self) -> Dict[str, float]:
        pending_sum = float(sum(self.pending)) if self.scale_delay > 0 else 0.0
        pending_norm = max(-1.0, min(1.0, pending_sum / max(1.0, float(self.scale_delay))))
        return {
            "cpu_util": float(self.cpu_util),
            "mem_util": float(self.mem_util),
            "backlog": float(self.backlog_norm()),
            "slo_miss": float(min(1.0, self.slo_miss)),
            "active_hosts": float(self.active_hosts) / float(self.max_hosts),
            "pending_scale": float(pending_norm),
        }

    def _generate_demand(self, regime: str) -> Tuple[float, float]:
        base = 0.8 + 0.4 * math.sin(2.0 * math.pi * (self.t / max(1, self.episode_len)))
        noise = self.rng.uniform(-0.1, 0.1)

        if regime == "normal":
            cpu = (base + noise) * 5.0
            mem = (base + noise) * 5.0
        elif regime == "cpu_heavy":
            cpu = (base + noise) * 7.0
            mem = (base + noise) * 4.0
        elif regime == "mem_heavy":
            cpu = (base + noise) * 4.0
            mem = (base + noise) * 7.0
        elif regime == "bursty":
            spike = 0.0
            if self.rng.random() < 0.15:
                spike = self.rng.uniform(2.0, 6.0)
            cpu = (base + noise) * 5.0 + spike
            mem = (base + noise) * 5.0 + spike
        else:
            cpu = (base + noise) * 5.0
            mem = (base + noise) * 5.0

        return max(0.0, cpu), max(0.0, mem)

    def _update_telemetry(self, demand_cpu: float, demand_mem: float) -> None:
        cpu_cap = float(self.active_hosts)
        mem_cap = float(self.active_hosts)

        self.cpu_util = min(1.0, (demand_cpu + 0.3 * self.backlog) / max(1e-6, cpu_cap))
        self.mem_util = min(1.0, (demand_mem + 0.3 * self.backlog) / max(1e-6, mem_cap))

        miss = 0.0
        if self.backlog > 2.0:
            miss += (self.backlog - 2.0) * 0.15
        miss += max(0.0, self.cpu_util - 0.85) * 0.5
        miss += max(0.0, self.mem_util - 0.85) * 0.5
        self.slo_miss = min(1.0, miss)
