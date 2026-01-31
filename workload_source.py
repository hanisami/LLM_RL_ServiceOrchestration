from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class WorkloadSource:
    """
    Produces (cpu_util, mem_util) in [0,1] per time step.
    The env will map utilization into demand relative to capacity.
    """
    def reset(self, episode_len: int) -> None:
        raise NotImplementedError

    def get(self, t: int) -> Tuple[float, float]:
        raise NotImplementedError


@dataclass
class TimePoint:
    cpu_util: float
    mem_util: float


class CSVTimeSeriesWorkload(WorkloadSource):
    """
    Reads a preprocessed CSV timeseries:
      columns: t,cpu_util,mem_util
    If mem_util is missing in the CSV, it will be synthesized as correlated with cpu.
    If the series is shorter than episode_len, it will loop.
    """

    def __init__(
        self,
        csv_path: str,
        loop: bool = True,
        synth_mem_if_missing: bool = True,
        seed: int = 0,
    ):
        self.csv_path = csv_path
        self.loop = loop
        self.synth_mem_if_missing = synth_mem_if_missing
        self.rng = random.Random(seed)
        self.series: List[TimePoint] = []
        self.episode_len = 0

    def reset(self, episode_len: int) -> None:
        self.episode_len = episode_len
        if not self.series:
            self.series = self._load_csv(self.csv_path)

    def get(self, t: int) -> Tuple[float, float]:
        if not self.series:
            raise RuntimeError("CSVTimeSeriesWorkload not initialized. Call reset() first.")
        idx = t
        if idx >= len(self.series):
            if self.loop:
                idx = idx % len(self.series)
            else:
                idx = len(self.series) - 1
        p = self.series[idx]
        return (p.cpu_util, p.mem_util)

    def _load_csv(self, path: str) -> List[TimePoint]:
        out: List[TimePoint] = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            has_mem = "mem_util" in r.fieldnames if r.fieldnames else False
            for row in r:
                cpu = float(row["cpu_util"])
                if has_mem:
                    mem = float(row["mem_util"])
                else:
                    if not self.synth_mem_if_missing:
                        mem = cpu
                    else:
                        # correlated + small noise
                        mem = _clip(0.70 * cpu + 0.15 + self.rng.uniform(-0.05, 0.05))
                out.append(TimePoint(cpu_util=_clip(cpu), mem_util=_clip(mem)))

        if not out:
            raise ValueError(f"CSV timeseries is empty: {path}")
        return out


class SyntheticAzureLikeWorkload(WorkloadSource):
    """
    Generates utilization patterns similar to public cloud VM aggregates:
    - diurnal seasonality
    - bursty spikes
    - regime switching (normal / cpu-heavy / mem-heavy / bursty)
    The output is in [0,1] and is stable for RL experiments.
    """

    def __init__(
        self,
        seed: int = 0,
        step_minutes: int = 5,
        switch_prob: float = 0.02,
        spike_prob: float = 0.08,
    ):
        self.rng = random.Random(seed)
        self.step_minutes = step_minutes
        self.switch_prob = switch_prob
        self.spike_prob = spike_prob

        self.episode_len = 0
        self.regime = "normal"

        # AR(1) noise state
        self.cpu_ar = 0.0
        self.mem_ar = 0.0

    def reset(self, episode_len: int) -> None:
        self.episode_len = episode_len
        self.regime = "normal"
        self.cpu_ar = 0.0
        self.mem_ar = 0.0

    def get(self, t: int) -> Tuple[float, float]:
        # Regime switching
        if self.rng.random() < self.switch_prob:
            self.regime = self.rng.choice(["normal", "cpu_heavy", "mem_heavy", "bursty"])

        # Diurnal baseline (5-min steps -> 288 steps per day)
        steps_per_day = int((24 * 60) / self.step_minutes)
        phase = 2.0 * math.pi * ((t % steps_per_day) / max(1, steps_per_day))
        diurnal = 0.45 + 0.18 * math.sin(phase) + 0.07 * math.sin(2 * phase)

        # AR(1) correlated noise
        self.cpu_ar = 0.85 * self.cpu_ar + self.rng.uniform(-0.05, 0.05)
        self.mem_ar = 0.85 * self.mem_ar + self.rng.uniform(-0.05, 0.05)

        cpu = diurnal + self.cpu_ar
        mem = diurnal + self.mem_ar

        # Regime bias
        if self.regime == "cpu_heavy":
            cpu += 0.10
            mem -= 0.05
        elif self.regime == "mem_heavy":
            mem += 0.10
            cpu -= 0.05

        # Bursty spikes (heavy-tailed-ish)
        if self.regime == "bursty" and self.rng.random() < self.spike_prob:
            spike = min(0.45, (self.rng.random() ** 0.25) * 0.45)  # skew to larger
            cpu += spike
            mem += 0.70 * spike

        # Keep mem correlated with cpu slightly (common in aggregates)
        mem = 0.65 * mem + 0.35 * (0.8 * cpu + 0.1)

        return (_clip(cpu), _clip(mem))
