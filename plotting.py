# llm_rl_poc/plotting.py
from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    return s.rolling(window=window, min_periods=1).mean()


def _align_on_episode(b: pd.DataFrame, l: pd.DataFrame) -> pd.DataFrame:
    """
    Robust alignment: merge on 'episode' so plots do not assume identical indexing/length.
    """
    if "episode" not in b.columns or "episode" not in l.columns:
        raise ValueError("CSV must contain an 'episode' column.")
    out = pd.merge(
        b.sort_values("episode"),
        l.sort_values("episode"),
        on="episode",
        how="inner",
        suffixes=("_base", "_ctx"),
    )
    return out


def _plot_two_curves(
    x: pd.Series,
    y_base: pd.Series,
    y_ctx: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_prefix: str,
    smooth_window: int = 10,
    show_raw: bool = True,
) -> None:
    plt.figure()

    # Optional raw traces (faint) to show variance without dominating the figure
    if show_raw:
        plt.plot(x, y_base, alpha=0.25, linewidth=1.0, label="Baseline RL (raw)")
        plt.plot(x, y_ctx, alpha=0.25, linewidth=1.0, label="RL + LLM (raw)")

    # Smoothed traces (primary comparison)
    plt.plot(x, _rolling_mean(y_base, smooth_window), linewidth=2.0, label=f"Baseline RL (MA-{smooth_window})")
    plt.plot(x, _rolling_mean(y_ctx, smooth_window), linewidth=2.0, label=f"RL + LLM (MA-{smooth_window})")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_prefix + ".png", dpi=300)
    plt.savefig(out_prefix + ".pdf")
    plt.close()


def generate_all_figures(
    baseline_train_csv: str,
    llm_train_csv: str,
    baseline_eval_csv: str,
    llm_eval_csv: str,
    out_dir: str = "figures",
    smooth_window: int = 10,
    show_raw: bool = True,
) -> None:
    _ensure_dir(out_dir)

    b_tr = pd.read_csv(baseline_train_csv)
    l_tr = pd.read_csv(llm_train_csv)
    b_ev = pd.read_csv(baseline_eval_csv)
    l_ev = pd.read_csv(llm_eval_csv)

    tr = _align_on_episode(b_tr, l_tr)
    ev = _align_on_episode(b_ev, l_ev)

    # --- TRAIN figures ---
    _plot_two_curves(
        x=tr["episode"],
        y_base=tr["ep_reward_base"],
        y_ctx=tr["ep_reward_ctx"],
        title="TRAIN Episode Reward",
        xlabel="Training Episode",
        ylabel="Reward",
        out_prefix=os.path.join(out_dir, "train_reward"),
        smooth_window=smooth_window,
        show_raw=show_raw,
    )

    _plot_two_curves(
        x=tr["episode"],
        y_base=tr["post_mean_slo_miss_base"],
        y_ctx=tr["post_mean_slo_miss_ctx"],
        title="TRAIN Post-Change Mean SLO Miss",
        xlabel="Training Episode",
        ylabel="SLO miss (post-change)",
        out_prefix=os.path.join(out_dir, "train_post_slo"),
        smooth_window=smooth_window,
        show_raw=show_raw,
    )

    # --- EVAL figures ---
    _plot_two_curves(
        x=ev["episode"],
        y_base=ev["ep_reward_base"],
        y_ctx=ev["ep_reward_ctx"],
        title="EVAL Episode Reward",
        xlabel="Evaluation Episode",
        ylabel="Reward",
        out_prefix=os.path.join(out_dir, "eval_reward"),
        smooth_window=max(1, smooth_window // 2),  # eval is shorter; use smaller MA by default
        show_raw=show_raw,
    )

    _plot_two_curves(
        x=ev["episode"],
        y_base=ev["post_mean_slo_miss_base"],
        y_ctx=ev["post_mean_slo_miss_ctx"],
        title="EVAL Post-Change Mean SLO Miss",
        xlabel="Evaluation Episode",
        ylabel="SLO miss (post-change)",
        out_prefix=os.path.join(out_dir, "eval_post_slo"),
        smooth_window=max(1, smooth_window // 2),
        show_raw=show_raw,
    )

    # LLM calls (EVAL) — only for ctx run
    if "llm_calls" in ev.columns:
        plt.figure()
        plt.plot(ev["episode"], ev["llm_calls_ctx"], linewidth=2.0, label="LLM calls per eval-episode")
        plt.xlabel("Evaluation Episode")
        plt.ylabel("Calls")
        plt.title("LLM Call Frequency (EVAL)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eval_llm_calls.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "eval_llm_calls.pdf"))
        plt.close()


def generate_llm_diagnostics(step_llm_csv: str, out_dir: str = "figures", episode_to_plot: int = 1) -> None:
    """
    Publication-friendly step diagnostics:
    - Always plots a valid episode (falls back to the first available if requested one is missing).
    - Adds action and bias timelines for interpretability.
    """
    _ensure_dir(out_dir)
    d = pd.read_csv(step_llm_csv)

    if "episode" not in d.columns:
        raise ValueError("Step CSV must contain an 'episode' column.")

    if episode_to_plot not in set(d["episode"].unique()):
        # Fallback to first available episode to avoid empty plots
        episode_to_plot = int(sorted(d["episode"].unique())[0])

    ep = d[d["episode"] == episode_to_plot].copy()
    ep = ep.sort_values("step")

    outlook_map = {"decreasing": 0, "stable": 1, "increasing": 2, "spike_likely": 3}
    if "ctx_outlook" in ep.columns:
        ep["ctx_outlook_num"] = ep["ctx_outlook"].map(outlook_map).fillna(1)

    calls = ep[ep.get("llm_called", 0) == 1]

    # Capacity timeline
    plt.figure()
    plt.plot(ep["step"], ep["active_hosts_norm"], linewidth=2.0, label="Active hosts (norm)")
    if not calls.empty:
        plt.scatter(calls["step"], calls["active_hosts_norm"], label="LLM call", marker="x")
    plt.xlabel("Step")
    plt.ylabel("Active hosts (norm)")
    plt.title(f"Capacity Timeline (episode={episode_to_plot})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diag_capacity_timeline.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "diag_capacity_timeline.pdf"))
    plt.close()

    # Action timeline (discrete)
    if "action" in ep.columns:
        plt.figure()
        plt.step(ep["step"], ep["action"], where="post", linewidth=2.0, label="Action")
        if not calls.empty:
            plt.scatter(calls["step"], calls["action"], label="LLM call", marker="x")
        plt.xlabel("Step")
        plt.ylabel("Action (0/1/2)")
        plt.title(f"Action Timeline (episode={episode_to_plot})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "diag_action_timeline.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "diag_action_timeline.pdf"))
        plt.close()

    # Bias timelines (if present)
    bias_cols = ["ctx_bias_a0", "ctx_bias_a1", "ctx_bias_a2"]
    if all(c in ep.columns for c in bias_cols):
        plt.figure()
        plt.plot(ep["step"], ep["ctx_bias_a0"], linewidth=2.0, label="Bias a0")
        plt.plot(ep["step"], ep["ctx_bias_a1"], linewidth=2.0, label="Bias a1")
        plt.plot(ep["step"], ep["ctx_bias_a2"], linewidth=2.0, label="Bias a2")
        if not calls.empty:
            plt.scatter(calls["step"], calls["ctx_bias_a1"], label="LLM call", marker="x")
        plt.xlabel("Step")
        plt.ylabel("Bias")
        plt.title(f"Context Bias Timeline (episode={episode_to_plot})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "diag_bias_timeline.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "diag_bias_timeline.pdf"))
        plt.close()

    # Outlook timeline (if present)
    if "ctx_outlook_num" in ep.columns:
        plt.figure()
        plt.plot(ep["step"], ep["ctx_outlook_num"], linewidth=2.0, label="Demand outlook (0..3)")
        if not calls.empty:
            plt.scatter(calls["step"], calls["ctx_outlook_num"], label="LLM call", marker="x")
        plt.xlabel("Step")
        plt.ylabel("Outlook (0=decr,1=stable,2=incr,3=spike)")
        plt.title(f"LLM Outlook Timeline (episode={episode_to_plot})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "diag_outlook_timeline.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, "diag_outlook_timeline.pdf"))
        plt.close()
