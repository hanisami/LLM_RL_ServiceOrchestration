from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class LLMContext:
    regime: str                  # normal/cpu_heavy/mem_heavy/bursty
    dominant_resource: str       # cpu/mem/balanced
    demand_outlook: str          # decreasing/stable/increasing/spike_likely
    risk_flags: List[str]        # sla_risk, instability_risk
    raw_json: Dict[str, Any]


class ContextProvider(Protocol):
    def get_context(self, window: List[Dict[str, float]]) -> LLMContext: ...


ALLOWED_REGIME = ["normal", "cpu_heavy", "mem_heavy", "bursty"]
ALLOWED_DOM = ["cpu", "mem", "balanced"]
ALLOWED_OUTLOOK = ["decreasing", "stable", "increasing", "spike_likely"]
ALLOWED_RISKS = ["sla_risk", "instability_risk"]


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return json.loads(text[start:end + 1])


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _normalize_choice(val: Any, options: List[str], default: str) -> str:
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        for x in val:
            if isinstance(x, str) and x in options:
                return x
        return default
    if isinstance(val, str):
        v = val.strip().strip('"').strip("'").lower()
        if v in options:
            return v
        for opt in options:
            if opt in v:
                return opt
        if v.startswith("[") and v.endswith("]"):
            try:
                arr = json.loads(v)
                if isinstance(arr, list):
                    for x in arr:
                        if isinstance(x, str) and x in options:
                            return x
            except Exception:
                pass
    return default


def _normalize_risks(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        out = [x for x in val if isinstance(x, str) and x in ALLOWED_RISKS]
        return sorted(list(set(out)))
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",")]
        out = [p for p in parts if p in ALLOWED_RISKS]
        return sorted(list(set(out)))
    return []


class MockContextProvider:
    """
    Deterministic "LLM-like" context based on window statistics.
    Notably: detects volatility ramps (std + high-quantile presence),
    but can revert to stable when variance drops.
    """

    def get_context(self, window: List[Dict[str, float]]) -> LLMContext:
        cpu = [w["cpu_util"] for w in window]
        mem = [w["mem_util"] for w in window]
        backlog = [w["backlog"] for w in window]
        slo = [w["slo_miss"] for w in window]

        avg_cpu = _mean(cpu)
        avg_mem = _mean(mem)
        std_cpu = _std(cpu)
        std_mem = _std(mem)

        # high tail frequency, but not too sensitive
        high_cpu_frac = sum(1 for x in cpu if x > 0.88) / max(1, len(cpu))
        high_mem_frac = sum(1 for x in mem if x > 0.88) / max(1, len(mem))

        if (std_cpu > 0.10) or (std_mem > 0.10) or (high_cpu_frac > 0.10) or (high_mem_frac > 0.10):
            regime = "bursty"
        elif avg_cpu - avg_mem > 0.10:
            regime = "cpu_heavy"
        elif avg_mem - avg_cpu > 0.10:
            regime = "mem_heavy"
        else:
            regime = "normal"

        if avg_cpu - avg_mem > 0.07:
            dom = "cpu"
        elif avg_mem - avg_cpu > 0.07:
            dom = "mem"
        else:
            dom = "balanced"

        # Outlook:
        # spike_likely only when variance/high-tail is meaningfully present
        if (std_cpu > 0.10) or (std_mem > 0.10) or (high_cpu_frac > 0.10) or (high_mem_frac > 0.10):
            outlook = "spike_likely"
        else:
            cpu_trend = cpu[-1] - cpu[0] if len(cpu) >= 2 else 0.0
            mem_trend = mem[-1] - mem[0] if len(mem) >= 2 else 0.0
            if cpu_trend > 0.07 or mem_trend > 0.07:
                outlook = "increasing"
            elif cpu_trend < -0.07 and mem_trend < -0.07:
                outlook = "decreasing"
            else:
                outlook = "stable"

        risks: List[str] = []
        if _mean(slo) > 0.15 or _mean(backlog) > 0.30:
            risks.append("sla_risk")
        if avg_cpu > 0.92 and avg_mem > 0.92:
            risks.append("instability_risk")

        raw = {
            "regime": regime,
            "dominant_resource": dom,
            "demand_outlook": outlook,
            "risk_flags": risks,
        }
        return LLMContext(regime=regime, dominant_resource=dom, demand_outlook=outlook, risk_flags=risks, raw_json=raw)


class OpenAIContextProvider:
    """
    OpenAI-compatible endpoint provider (works with Ollama by setting OPENAI_BASE_URL).
    """

    def __init__(self, model: str = "llama3.2"):
        if OpenAI is None:
            raise RuntimeError("openai package missing. Install: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        base_url = os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model

    def get_context(self, window: List[Dict[str, float]]) -> LLMContext:
        prompt = self._build_prompt(window)
        parsed = self._call_json(prompt)

        regime = _normalize_choice(parsed.get("regime"), ALLOWED_REGIME, "normal")
        dom = _normalize_choice(parsed.get("dominant_resource"), ALLOWED_DOM, "balanced")
        outlook = _normalize_choice(parsed.get("demand_outlook"), ALLOWED_OUTLOOK, "stable")
        risks = _normalize_risks(parsed.get("risk_flags"))

        return LLMContext(regime=regime, dominant_resource=dom, demand_outlook=outlook, risk_flags=risks, raw_json=parsed)

    def _build_prompt(self, window: List[Dict[str, float]]) -> str:
        cpu = [w["cpu_util"] for w in window]
        mem = [w["mem_util"] for w in window]
        backlog = [w["backlog"] for w in window]
        slo = [w["slo_miss"] for w in window]

        avg_cpu = _mean(cpu)
        avg_mem = _mean(mem)
        std_cpu = _std(cpu)
        std_mem = _std(mem)

        high_cpu_frac = sum(1 for x in cpu if x > 0.88) / max(1, len(cpu))
        high_mem_frac = sum(1 for x in mem if x > 0.88) / max(1, len(mem))

        tail = window[-8:] if len(window) >= 8 else window
        tail_lines = "\n".join(
            f"{i}:cpu={p['cpu_util']:.3f},mem={p['mem_util']:.3f},b={p['backlog']:.3f},slo={p['slo_miss']:.3f}"
            for i, p in enumerate(tail)
        )

        return (
            "Return ONLY minified JSON. No prose. No markdown.\n"
            'Schema: {"regime":"<one>","dominant_resource":"<one>","demand_outlook":"<one>","risk_flags":["..."]}\n'
            f"Allowed regime: {ALLOWED_REGIME}\n"
            f"Allowed dominant_resource: {ALLOWED_DOM}\n"
            f"Allowed demand_outlook: {ALLOWED_OUTLOOK}\n"
            f"Allowed risk_flags: subset of {ALLOWED_RISKS}\n"
            "Rules:\n"
            "- If std_cpu>0.10 OR std_mem>0.10 OR high_cpu_frac>0.10 OR high_mem_frac>0.10 then demand_outlook MUST be \"spike_likely\".\n"
            "- Otherwise choose decreasing/stable/increasing based on trend.\n"
            f"features: avg_cpu={avg_cpu:.3f}, avg_mem={avg_mem:.3f}, std_cpu={std_cpu:.3f}, std_mem={std_mem:.3f}, "
            f"high_cpu_frac={high_cpu_frac:.3f}, high_mem_frac={high_mem_frac:.3f}, avg_backlog={_mean(backlog):.3f}, avg_slo={_mean(slo):.3f}\n"
            f"recent_points:\n{tail_lines}\n"
        )

    def _call_json(self, user_text: str) -> Dict[str, Any]:
        # Try Responses API
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "Output strict minified JSON only."},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
                max_output_tokens=180,
            )
            text = getattr(resp, "output_text", "") or ""
            if text.strip():
                return _extract_first_json_object(text)
        except Exception:
            pass

        # Fallback to Chat Completions (needed for some Ollama builds)
        try:
            cc = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output strict minified JSON only."},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
                max_tokens=180,
            )
            text = (cc.choices[0].message.content or "").strip()
            if text:
                return _extract_first_json_object(text)
        except Exception:
            pass

        return {"regime": "normal", "dominant_resource": "balanced", "demand_outlook": "stable", "risk_flags": []}
