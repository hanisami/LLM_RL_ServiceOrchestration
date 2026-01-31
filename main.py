from __future__ import annotations

import argparse
import json
import os

from experiment import run_experiment
from context_provider import MockContextProvider, OpenAIContextProvider
from plotting import generate_all_figures, generate_llm_diagnostics
from logging_utils import ensure_dir
from workload_source import CSVTimeSeriesWorkload, SyntheticAzureLikeWorkload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--provider", choices=["mock", "openai"], default="mock")
    p.add_argument("--model", default="llama3.2")

    p.add_argument("--train-episodes", type=int, default=120)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--episode-len", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--workload", choices=["synthetic_azure", "csv"], default="synthetic_azure")
    p.add_argument("--trace-csv", default="")

    p.add_argument("--llm-interval", type=int, default=100)
    p.add_argument("--llm-window", type=int, default=60)

    # for step plots (this uses the global episode id in the step log)
    p.add_argument("--plot-episode-id", type=int, default=1)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dir("runs")
    ensure_dir("figures")

    def make_workload():
        if args.workload == "synthetic_azure":
            return SyntheticAzureLikeWorkload(seed=args.seed)
        if not args.trace_csv:
            raise SystemExit("ERROR: --trace-csv is required when --workload csv")
        return CSVTimeSeriesWorkload(csv_path=args.trace_csv, seed=args.seed)

    # Outputs
    base_train_csv = os.path.join("runs", f"baseline_train_{args.workload}.csv")
    base_eval_csv = os.path.join("runs", f"baseline_eval_{args.workload}.csv")

    llm_train_csv = os.path.join("runs", f"llm_train_{args.provider}_{args.workload}.csv")
    llm_eval_csv = os.path.join("runs", f"llm_eval_{args.provider}_{args.workload}.csv")

    step_csv = os.path.join("runs", f"steps_{args.provider}_{args.workload}.csv")

    print("\n=== RUN 1: RL baseline (train + eval) ===")
    res_base = run_experiment(
        name="RL_BASE",
        provider=None,
        workload=make_workload(),
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        episode_len=args.episode_len,
        llm_interval=args.llm_interval,
        llm_window_size=args.llm_window,
        csv_train_out=base_train_csv,
        csv_eval_out=base_eval_csv,
        step_csv_out=None,
    )

    provider = MockContextProvider() if args.provider == "mock" else OpenAIContextProvider(model=args.model)

    print(f"\n=== RUN 2: RL + context ({args.provider}) (train + eval) ===")
    res_llm = run_experiment(
        name="RL_CTX",
        provider=provider,
        workload=make_workload(),
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        episode_len=args.episode_len,
        llm_interval=args.llm_interval,
        llm_window_size=args.llm_window,
        csv_train_out=llm_train_csv,
        csv_eval_out=llm_eval_csv,
        step_csv_out=step_csv,
    )

    print("\n=== FINAL SUMMARY ===")
    print("Baseline:", json.dumps(res_base, indent=2))
    print("RL+LLM  :", json.dumps(res_llm, indent=2))

    print("\nGenerating figures (EVAL curves)...")
    # generate_all_figures(baseline_eval_csv=base_eval_csv, llm_eval_csv=llm_eval_csv, out_dir="figures")
    generate_all_figures(
        baseline_train_csv=base_train_csv,
        llm_train_csv=llm_train_csv,
        baseline_eval_csv=base_eval_csv,
        llm_eval_csv=llm_eval_csv,
        out_dir="figures",
        smooth_window=10,
        show_raw=True,
    )

    if os.path.exists(step_csv):
        generate_llm_diagnostics(step_llm_csv=step_csv, out_dir="figures", episode_to_plot=args.plot_episode_id)

    print("Done.")
    print("Logs: runs/*.csv")
    print("Figures: figures/*.png")


if __name__ == "__main__":
    main()
