import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置全局字体为 serif（Times 风格）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times', 'DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


# =========================================================
# 配置
# =========================================================
CSV_PATH = "results/rrt_multi_seed/multi_seed_results.csv"
OUT_DIR = "results/figures/baseline"
DPI = 300
PAD_INCHES = 0.03


# =========================================================
# 通用保存函数
# =========================================================
def save_fig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 输出SVG和PDF矢量图
    svg_path = out_path.with_suffix('.svg')
    pdf_path = out_path.with_suffix('.pdf')
    fig.savefig(svg_path, format='svg', bbox_inches="tight", pad_inches=PAD_INCHES)
    fig.savefig(pdf_path, format='pdf', bbox_inches="tight", pad_inches=PAD_INCHES)
    plt.close(fig)
    print(f"[Done] saved: {svg_path}")
    print(f"[Done] saved: {pdf_path}")


# =========================================================
# 读取数据
# =========================================================
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # 成功样本
    df_success = df[df["success"] == 1].copy()

    # 过滤掉无效的 first_path_iter
    if "first_path_iter" in df_success.columns:
        df_success = df_success[df_success["first_path_iter"] >= 0].copy()

    # 按 task_seed 排序，方便画折线图
    if "task_seed" in df_success.columns:
        df_success = df_success.sort_values("task_seed").reset_index(drop=True)

    return df, df_success


# =========================================================
# 图1：first_path_iter 折线图
# =========================================================
def plot_first_path_iter_line(df_success: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

    x = df_success["task_seed"].to_numpy()
    y = df_success["first_path_iter"].to_numpy()

    ax.plot(x, y, marker="o", linewidth=1.4, markersize=3.8)

    ax.set_xlabel("Task Seed")
    ax.set_ylabel("First Path Iteration")
    # ax.set_title("First Feasible Path Iteration across Tasks")

    if len(x) > 0:
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)

    ax.margins(x=0.01, y=0.08)
    ax.grid(True, alpha=0.25)

    save_fig(fig, out_dir / "baseline_first_path_iter_line.png")


# =========================================================
# 图2：三类无效扩展均值柱状图
# =========================================================
def plot_invalid_types_bar(df_success: pd.DataFrame, out_dir: Path):
    metrics = ["n_exp_invalid", "n_inc_invalid", "n_prog_invalid"]
    labels = ["Expansion Invalid", "Increment Invalid", "Progress Invalid"]
    means = [df_success[m].mean() for m in metrics]

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

    bars = ax.bar(labels, means, width=0.62)

    ax.set_ylabel("Mean Count")
    # ax.set_title("Mean Counts of Invalid Expansion Types")
    ax.margins(x=0.04, y=0.10)
    ax.grid(True, axis="y", alpha=0.25)

    # 数值标注
    ymax = max(means) if len(means) > 0 else 0.0
    offset = max(1.0, ymax * 0.02)
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    save_fig(fig, out_dir / "baseline_invalid_types_bar.png")


# =========================================================
# 图3：invalid_ratio 折线图
# =========================================================
def plot_invalid_ratio_line(df_success: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

    x = df_success["task_seed"].to_numpy()
    y = df_success["invalid_ratio"].to_numpy()

    ax.plot(x, y, marker="o", linewidth=1.4, markersize=3.8)

    ax.set_xlabel("Task Seed")
    ax.set_ylabel("Invalid Ratio")
    # ax.set_title("Invalid Ratio across Tasks")

    if len(x) > 0:
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)

    ax.margins(x=0.01, y=0.08)
    ax.grid(True, alpha=0.25)

    save_fig(fig, out_dir / "baseline_invalid_ratio_line.png")


# =========================================================
# 图4：invalid_ratio vs first_path_iter 散点图
# =========================================================
def plot_invalid_ratio_vs_first_path_iter(df_success: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)

    x = df_success["invalid_ratio"].to_numpy()
    y = df_success["first_path_iter"].to_numpy()

    ax.scatter(x, y, s=28)

    ax.set_xlabel("Invalid Ratio")
    ax.set_ylabel("First Path Iteration")
    # ax.set_title("Invalid Ratio vs First Feasible Path Iteration")

    ax.margins(x=0.05, y=0.08)
    ax.grid(True, alpha=0.25)

    save_fig(fig, out_dir / "baseline_invalid_ratio_vs_first_path_iter.png")


# =========================================================
# 统计表
# =========================================================
def save_summary_stats(df: pd.DataFrame, df_success: pd.DataFrame, out_dir: Path):
    summary = pd.DataFrame({
        "metric": [
            "first_path_iter",
            "n_exp_invalid",
            "n_inc_invalid",
            "n_prog_invalid",
            "invalid_ratio",
            "success_rate",
        ],
        "mean": [
            df_success["first_path_iter"].mean(),
            df_success["n_exp_invalid"].mean(),
            df_success["n_inc_invalid"].mean(),
            df_success["n_prog_invalid"].mean(),
            df_success["invalid_ratio"].mean(),
            df["success"].mean(),
        ],
        "std": [
            df_success["first_path_iter"].std(ddof=1),
            df_success["n_exp_invalid"].std(ddof=1),
            df_success["n_inc_invalid"].std(ddof=1),
            df_success["n_prog_invalid"].std(ddof=1),
            df_success["invalid_ratio"].std(ddof=1),
            df["success"].std(ddof=1),
        ],
    })

    summary.to_csv(out_dir / "baseline_summary_stats.csv", index=False)


# =========================================================
# 主函数
# =========================================================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, df_success = load_data(CSV_PATH)

    plot_first_path_iter_line(df_success, out_dir)
    plot_invalid_types_bar(df_success, out_dir)
    plot_invalid_ratio_line(df_success, out_dir)
    plot_invalid_ratio_vs_first_path_iter(df_success, out_dir)
    save_summary_stats(df, df_success, out_dir)

    print(f"绘图完成，输出目录：{out_dir}")


if __name__ == "__main__":
    main()