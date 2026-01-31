from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import time

from env import NonStationaryEnv
from rl_agent import QLearningAgent
from encoding import encode_state
from context_provider import ContextProvider, LLMContext
from logging_utils import write_csv
from workload_source import WorkloadSource


def _context_action_bias(obs: Dict[str, float], ctx: Optional[LLMContext]) -> List[float]:
    """Convert context into a lightweight action prior.


    IMPORTANT: This is not a hand-crafted controller that replaces RL.
    It is a soft bias that helps the agent break ties and learn faster in
    non-stationary regimes.

    Actions: 0=consolidate, 1=balanced, 2=add_headroom
    """
    # default: prefer "balanced" over "consolidate" when uncertain.
    bias = [0.0, 0.35, 0.0]

    cpu = float(obs.get("cpu_util", 0.0))
    mem = float(obs.get("mem_util", 0.0))
    backlog = float(obs.get("backlog", 0.0))
    slo = float(obs.get("slo_miss", 0.0))
    pending = float(obs.get("pending_scale", 0.0))

    # If we already have pending positive scale, discourage more adds to reduce oscillations.
    if pending > 0.30:
        bias[2] -= 0.35
    if pending < -0.30:
        bias[0] -= 0.20

    if ctx is None:
        # Still allow a weak on-the-spot bias when backlog/SLO are high.
        if slo > 0.20 or backlog > 0.30:
            bias[2] += 0.50
        return bias

    risks = ctx.risk_flags or []
    outlook = (ctx.demand_outlook or "stable").lower()
    dom = (ctx.dominant_resource or "balanced").lower()

    # Risk-driven bias
    if "sla_risk" in risks:
        bias[2] += 0.75
        bias[1] += 0.15
    if "instability_risk" in risks:
        # When both resources are saturated, prioritize headroom.
        bias[2] += 0.90

    # Outlook-driven bias
    if outlook == "spike_likely":
        bias[2] += 0.85
        bias[1] += 0.10
    elif outlook == "increasing":
        bias[2] += 0.55
    elif outlook == "decreasing":
        # Only consolidate when we are comfortably below thresholds.
        if cpu < 0.55 and mem < 0.55 and backlog < 0.10 and slo < 0.05 and pending <= 0.0:
            bias[0] += 0.60
        else:
            bias[1] += 0.10

    # Dominant-resource nudges
    if dom == "cpu" and cpu > 0.80:
        bias[2] += 0.25
    if dom == "mem" and mem > 0.80:
        bias[2] += 0.25

    return bias


def _run_one_episode(
    env: NonStationaryEnv,
    agent: QLearningAgent,
    provider: Optional[ContextProvider],
    llm_interval: int,
    llm_window_size: int,
    warmup_steps: int,
    train: bool,
    step_log: Optional[List[Dict]],
    episode_index: int,
) -> Dict[str, float]:
    use_llm_context = provider is not None
    change_step = env.episode_len // 2

    obs = env.reset()
    done = False

    hist: Deque[Dict[str, float]] = deque(maxlen=llm_window_size)
    context: Optional[LLMContext] = None
    llm_calls_in_ep = 0

    # warmup (same as before)
    for _ in range(warmup_steps):
        hist.append({
            "cpu_util": float(obs["cpu_util"]),
            "mem_util": float(obs["mem_util"]),
            "backlog": float(obs["backlog"]),
            "slo_miss": float(obs["slo_miss"]),
        })
        obs, _, done, _ = env.step(1)
        if done:
            break

    # initial LLM call (optional)
    if use_llm_context and len(hist) > 0:
        t0 = time.perf_counter()
        context = provider.get_context(list(hist))
        llm_latency_init = (time.perf_counter() - t0) * 1000.0
        llm_calls_in_ep += 1
    else:
        llm_latency_init = 0.0

    ep_reward = 0.0
    pre_slo_sum = post_slo_sum = 0.0
    pre_steps = post_steps = 0

    steps = 0
    while not done:
        llm_called = 0
        llm_latency_ms_step = 0.0

        if use_llm_context and (steps > 0) and (steps % max(1, llm_interval) == 0) and len(hist) > 0:
            t0 = time.perf_counter()
            context = provider.get_context(list(hist))
            llm_latency_ms_step = (time.perf_counter() - t0) * 1000.0
            llm_called = 1
            llm_calls_in_ep += 1

        s = encode_state(obs, context, use_llm=use_llm_context)

        # If LLM/context is enabled, apply a soft action-bias derived from the context.
        # This ensures the "RL_CTX" run is not a no-op and can exploit window-based signals.
        if use_llm_context:
            bias = _context_action_bias(obs, context)
            a = agent.act_biased(s, bias) if train else agent.act_greedy_biased(s, bias)
        else:
            bias = None
            a = agent.act(s) if train else agent.act_greedy(s)

        obs2, r, done, info = env.step(a)

        if train:
            s2 = encode_state(obs2, context, use_llm=use_llm_context)
            agent.update(s, a, r, s2, done)

        ep_reward += r

        if steps < change_step:
            pre_slo_sum += info["slo_miss"]; pre_steps += 1
        else:
            post_slo_sum += info["slo_miss"]; post_steps += 1

        if step_log is not None:
            row = {
                "mode": "train" if train else "eval",
                "episode": episode_index,
                "step": steps,
                "phase": "pre" if steps < change_step else "post",
                "cpu_util": float(obs2["cpu_util"]),
                "mem_util": float(obs2["mem_util"]),
                "backlog": float(obs2["backlog"]),
                "slo_miss": float(obs2["slo_miss"]),
                "active_hosts_norm": float(obs2["active_hosts"]),
                "pending_scale": float(obs2.get("pending_scale", 0.0)),
                "action": int(a),
                "reward": float(r),
                "llm_called": int(llm_called),
                "llm_latency_ms": float(llm_latency_ms_step),
                "applied_delta": int(info.get("applied_delta", 0)),
                "scheduled_delta": int(info.get("scheduled_delta", 0)),
                "pending_sum": int(info.get("pending_sum", 0)),
            }
            if bias is not None:
                row["ctx_bias_a0"] = float(bias[0])
                row["ctx_bias_a1"] = float(bias[1])
                row["ctx_bias_a2"] = float(bias[2])
            if context is not None:
                row["ctx_regime"] = context.regime
                row["ctx_dom"] = context.dominant_resource
                row["ctx_outlook"] = context.demand_outlook
                row["ctx_spike_bit"] = 1 if context.demand_outlook == "spike_likely" else 0
                row["ctx_sla_risk"] = 1 if "sla_risk" in context.risk_flags else 0
                row["ctx_instability_risk"] = 1 if "instability_risk" in context.risk_flags else 0
            step_log.append(row)

        hist.append({
            "cpu_util": float(obs2["cpu_util"]),
            "mem_util": float(obs2["mem_util"]),
            "backlog": float(obs2["backlog"]),
            "slo_miss": float(obs2["slo_miss"]),
        })

        obs = obs2
        steps += 1

    return {
        "ep_reward": float(ep_reward),
        "pre_mean_slo_miss": float(pre_slo_sum / max(1, pre_steps)),
        "post_mean_slo_miss": float(post_slo_sum / max(1, post_steps)),
        "llm_calls": float(llm_calls_in_ep),
        "llm_init_latency_ms": float(llm_latency_init),
    }


def run_experiment(
    name: str,
    provider: Optional[ContextProvider],
    workload: Optional[WorkloadSource],
    seed: int,
    train_episodes: int = 120,
    eval_episodes: int = 30,
    episode_len: int = 800,
    warmup_steps: int = 10,
    llm_interval: int = 100,
    llm_window_size: int = 60,
    verbose_every: int = 20,
    csv_train_out: Optional[str] = None,
    csv_eval_out: Optional[str] = None,
    step_csv_out: Optional[str] = None,
) -> Dict[str, float]:
    env = NonStationaryEnv(seed=seed, episode_len=episode_len, workload=workload)
    agent = QLearningAgent(n_actions=3, seed=seed + 10)

    step_rows: List[Dict] = []

    # ---- TRAIN ----
    train_rows: List[Dict] = []
    train_rewards: List[float] = []
    for ep in range(1, train_episodes + 1):
        out = _run_one_episode(
            env=env,
            agent=agent,
            provider=provider,
            llm_interval=llm_interval,
            llm_window_size=llm_window_size,
            warmup_steps=warmup_steps,
            train=True,
            step_log=step_rows if step_csv_out else None,
            episode_index=ep,
        )
        agent.end_episode()
        train_rows.append({"episode": ep, **out})
        train_rewards.append(out["ep_reward"])

        if verbose_every and (ep % verbose_every == 0):
            avgR = sum(train_rewards[-verbose_every:]) / float(verbose_every)
            print(f"[{name}][TRAIN] ep {ep:3d}/{train_episodes} avg_reward(last{verbose_every})={avgR:.2f} eps={agent.eps:.3f}")

    if csv_train_out:
        write_csv(csv_train_out, train_rows)

    # ---- EVAL (greedy, no learning) ----
    agent.set_eps(0.0)
    eval_rows: List[Dict] = []
    eval_rewards: List[float] = []
    eval_post_slo: List[float] = []

    for k in range(1, eval_episodes + 1):
        ep_id = train_episodes + k  # continue numbering
        out = _run_one_episode(
            env=env,
            agent=agent,
            provider=provider,
            llm_interval=llm_interval,
            llm_window_size=llm_window_size,
            warmup_steps=warmup_steps,
            train=False,
            step_log=step_rows if step_csv_out else None,
            episode_index=ep_id,
        )
        eval_rows.append({"episode": k, **out})
        eval_rewards.append(out["ep_reward"])
        eval_post_slo.append(out["post_mean_slo_miss"])

    if csv_eval_out:
        write_csv(csv_eval_out, eval_rows)
    if step_csv_out:
        write_csv(step_csv_out, step_rows)

    return {
        "train_avg_reward": sum(r["ep_reward"] for r in train_rows) / max(1, len(train_rows)),
        "eval_avg_reward": sum(eval_rewards) / max(1, len(eval_rewards)),
        "eval_post_mean_slo": sum(eval_post_slo) / max(1, len(eval_post_slo)),
        "eval_llm_calls_per_ep": sum(r["llm_calls"] for r in eval_rows) / max(1, len(eval_rows)),
    }
