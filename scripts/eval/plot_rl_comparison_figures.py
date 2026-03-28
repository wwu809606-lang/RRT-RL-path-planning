import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = [
        "task_id",
        "method",
        "success",
        "iterations",
        "first_path_iter",
        "n_exp_invalid",
        "n_inc_invalid",
        "n_prog_invalid",
        "invalid_ratio",
        "runtime_sec",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"结果表缺少必要字段: {missing}")
    return df


def method_mean(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby("method", as_index=False)[metric]
        .mean()
        .sort_values(by=metric, ascending=True)
    )


def pivot_by_task(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot(index="task_id", columns="method", values=metric)


def save_bar_mean(df: pd.DataFrame, metric: str, ylabel: str, outpath: str):
    mean_df = method_mean(df, metric)

    plt.figure(figsize=(7, 5))
    plt.bar(mean_df["method"], mean_df[metric])
    plt.ylabel(ylabel)
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_success_rate(df: pd.DataFrame, outpath: str):
    summary = df.groupby("method", as_index=False)["success"].mean()

    plt.figure(figsize=(7, 5))
    plt.bar(summary["method"], summary["success"])
    plt.ylabel("Success Rate")
    plt.xlabel("Method")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_task_scatter(df: pd.DataFrame, metric: str, ylabel: str, outpath: str):
    pivot = pivot_by_task(df, metric)

    plt.figure(figsize=(9, 5))
    for method in pivot.columns:
        x = np.arange(len(pivot))
        y = pivot[method].values
        plt.scatter(x, y, s=18, label=method)
        y_mean = np.nanmean(y)
        plt.plot([x.min(), x.max()], [y_mean, y_mean], linewidth=1.5)

    plt.xlabel("Task ID")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_pairwise_improvement_bar(
    df: pd.DataFrame,
    baseline_method: str,
    rl_method: str,
    metric: str,
    ylabel: str,
    outpath: str,
):
    pivot = pivot_by_task(df, metric)

    if baseline_method not in pivot.columns or rl_method not in pivot.columns:
        return

    improvement = pivot[baseline_method] - pivot[rl_method]

    plt.figure(figsize=(9, 5))
    x = np.arange(len(improvement))
    plt.bar(x, improvement.values)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Task ID")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/rl_eval/per_task_results.csv",
        help="逐任务结果 CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rl_eval/figures",
        help="输出图片目录",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="baseline_rrt",
        help="baseline 方法名",
    )
    parser.add_argument(
        "--rl-name",
        type=str,
        default="rrt_rl",
        help="RL 方法名",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    df = load_results(args.input)

    # 1. 均值柱状图
    save_bar_mean(
        df=df,
        metric="first_path_iter",
        ylabel="Mean First Path Iteration",
        outpath=os.path.join(args.output_dir, "mean_first_path_iter.png"),
    )
    save_bar_mean(
        df=df,
        metric="invalid_ratio",
        ylabel="Mean Invalid Ratio",
        outpath=os.path.join(args.output_dir, "mean_invalid_ratio.png"),
    )
    save_bar_mean(
        df=df,
        metric="n_prog_invalid",
        ylabel="Mean Number of Progress Invalid Expansions",
        outpath=os.path.join(args.output_dir, "mean_n_prog_invalid.png"),
    )
    save_bar_mean(
        df=df,
        metric="runtime_sec",
        ylabel="Mean Runtime (s)",
        outpath=os.path.join(args.output_dir, "mean_runtime_sec.png"),
    )

    # 2. 成功率
    save_success_rate(
        df=df,
        outpath=os.path.join(args.output_dir, "success_rate.png"),
    )

    # 3. 逐任务散点图
    save_task_scatter(
        df=df,
        metric="first_path_iter",
        ylabel="First Path Iteration",
        outpath=os.path.join(args.output_dir, "scatter_first_path_iter.png"),
    )
    save_task_scatter(
        df=df,
        metric="invalid_ratio",
        ylabel="Invalid Ratio",
        outpath=os.path.join(args.output_dir, "scatter_invalid_ratio.png"),
    )
    save_task_scatter(
        df=df,
        metric="n_prog_invalid",
        ylabel="Number of Progress Invalid Expansions",
        outpath=os.path.join(args.output_dir, "scatter_n_prog_invalid.png"),
    )

    # 4. 逐任务改进量
    save_pairwise_improvement_bar(
        df=df,
        baseline_method=args.baseline_name,
        rl_method=args.rl_name,
        metric="first_path_iter",
        ylabel="Baseline - RL (First Path Iteration)",
        outpath=os.path.join(args.output_dir, "improvement_first_path_iter.png"),
    )
    save_pairwise_improvement_bar(
        df=df,
        baseline_method=args.baseline_name,
        rl_method=args.rl_name,
        metric="invalid_ratio",
        ylabel="Baseline - RL (Invalid Ratio)",
        outpath=os.path.join(args.output_dir, "improvement_invalid_ratio.png"),
    )
    save_pairwise_improvement_bar(
        df=df,
        baseline_method=args.baseline_name,
        rl_method=args.rl_name,
        metric="n_prog_invalid",
        ylabel="Baseline - RL (Progress Invalid Expansions)",
        outpath=os.path.join(args.output_dir, "improvement_n_prog_invalid.png"),
    )

    print(f"[Saved] figures -> {args.output_dir}")


if __name__ == "__main__":
    main()