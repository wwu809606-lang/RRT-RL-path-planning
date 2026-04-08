import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from stable_baselines3 import PPO

from src.planners.rrt_3d import RRTPlanner3D, point_in_collision_3d
from src.envs.rrt_rl_env import RRTRLEnv
from src.rewards.reward_fn import RewardConfig


XYZ = Tuple[float, float, float]

BUILDING_FILE = "data/processed/buildings_keep_15_recommended.geojson"
DEFAULT_MODEL = "results/rl/best_model/best_model.zip"


# =========================================================
# 1. 建筑读取与预处理
# =========================================================
def load_buildings(infile: str):
    gdf = gpd.read_file(infile)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    # 转米制投影
    gdf = gdf.to_crs(epsg=3857)

    # 原始范围
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # 平移到局部坐标系
    gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)

    # 几何过滤
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # 高度字段
    gdf["height_m"] = gdf["height_m"].astype(float)
    gdf = gdf[np.isfinite(gdf["height_m"])].copy()

    if len(gdf) == 0:
        raise ValueError("没有有效建筑数据。")

    meta = {
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": float(xmax - xmin),
        "y_max": float(ymax - ymin),
        "z_min": 0.0,
        "z_max": max(120.0, float(gdf["height_m"].max()) + 20.0),
    }
    return gdf, meta


def gdf_to_obstacles(gdf) -> List[Dict[str, Any]]:
    """
    转成 RRTPlanner3D 使用的障碍格式：
    {
        "polygon": shapely polygon,
        "height": float
    }
    """
    obstacles = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row["height_m"])

        if geom.geom_type == "Polygon":
            obstacles.append({"polygon": geom, "height": h})
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                obstacles.append({"polygon": poly, "height": h})

    return obstacles


# =========================================================
# 2. 随机起点终点（距离更远）
# =========================================================
def sample_free_point(
    obstacles,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> XYZ:
    while True:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(max(z_min, 20.0), min(z_max, 80.0))
        p = (x, y, z)
        if not point_in_collision_3d(p, obstacles):
            return p


def sample_start_goal(
    obstacles,
    meta: Dict[str, float],
    min_dist: float = 600.0,
    max_dist: float = 1600.0,
    max_trials: int = 1000,
) -> Tuple[XYZ, XYZ]:
    """
    采样相距较远的起终点。
    """
    for _ in range(max_trials):
        start = sample_free_point(
            obstacles,
            meta["x_min"], meta["x_max"],
            meta["y_min"], meta["y_max"],
            meta["z_min"], meta["z_max"],
        )
        goal = sample_free_point(
            obstacles,
            meta["x_min"], meta["x_max"],
            meta["y_min"], meta["y_max"],
            meta["z_min"], meta["z_max"],
        )

        d = ((start[0] - goal[0]) ** 2 +
             (start[1] - goal[1]) ** 2 +
             (start[2] - goal[2]) ** 2) ** 0.5

        if min_dist <= d <= max_dist:
            return start, goal

    raise RuntimeError("无法采样到满足距离要求的起终点，请调小 min_dist 或增大 max_trials。")


# =========================================================
# 3. baseline：手动跑并记录 invalid edges
# =========================================================
def run_baseline(planner_kwargs: Dict[str, Any], start: XYZ, goal: XYZ):
    t0 = time.perf_counter()

    planner = RRTPlanner3D(**planner_kwargs)
    planner.reset_tree(start, goal)

    invalid_edges = []

    for _ in range(planner.max_iter):
        rnd = planner.sample_free(goal)
        res = planner.extend_once(rnd)

        if res["status"] in ["expansion_invalid", "increment_invalid"]:
            invalid_edges.append({
                "from": res["nearest_xyz"],
                "to": res["new_xyz"],
                "status": res["status"],
            })

        if res["goal_reached"]:
            elapsed = time.perf_counter() - t0

            path_xyz = np.array(planner.extract_path(planner.goal_node_idx), dtype=float)
            tree_edges = planner.export_tree_edges()
            result = {
                "success": True,
                "iterations": planner.iter_count,
                "first_path_iter": planner.first_path_iter,
                "n_exp_invalid": planner.n_exp_invalid,
                "n_inc_invalid": planner.n_inc_invalid,
                "n_prog_invalid": planner.n_prog_invalid,
                "invalid_ratio": planner.get_search_state()["invalid_ratio"],
                "runtime_sec": elapsed,
            }
            return path_xyz, tree_edges, invalid_edges, result

        if planner.iter_count >= planner.max_iter:
            break

    raise RuntimeError("Baseline RRT failed on this task.")


# =========================================================
# 4. RL：运行并记录 invalid edges
# =========================================================
def run_rl(
    model_path: str,
    planner_kwargs: Dict[str, Any],
    start: XYZ,
    goal: XYZ,
    k_candidates: int = 8,
):
    t0 = time.perf_counter()

    task_list = [{"start": start, "goal": goal}]

    env = RRTRLEnv(
        planner_kwargs=planner_kwargs,
        task_list=task_list,
        k_candidates=k_candidates,
        reward_cfg=RewardConfig(),
    )

    model = PPO.load(model_path)

    # 不传固定 seed，保证每次运行都不同
    obs, _ = env.reset()

    terminated = False
    truncated = False
    invalid_edges = []

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))

        step_result = info["step_result"]

        if step_result["status"] in ["expansion_invalid", "increment_invalid"]:
            invalid_edges.append({
                "from": step_result["nearest_xyz"],
                "to": step_result["new_xyz"],
                "status": step_result["status"],
            })

    planner = env.planner
    if planner is None or planner.goal_node_idx is None:
        raise RuntimeError("RL failed on this task.")

    elapsed = time.perf_counter() - t0

    path_xyz = np.array(planner.extract_path(planner.goal_node_idx), dtype=float)
    tree_edges = planner.export_tree_edges()

    state = planner.get_search_state()
    result = {
        "success": state["success"],
        "iterations": state["iter_count"],
        "first_path_iter": state["first_path_iter"],
        "n_exp_invalid": state["n_exp_invalid"],
        "n_inc_invalid": state["n_inc_invalid"],
        "n_prog_invalid": state["n_prog_invalid"],
        "invalid_ratio": state["invalid_ratio"],
        "runtime_sec": elapsed,
    }

    return path_xyz, tree_edges, invalid_edges, result


# =========================================================
# 5. 绘图函数
# =========================================================
def plot_buildings(ax, gdf):
    for _, row in gdf.iterrows():
        geom = row.geometry

        if geom.geom_type == "Polygon":
            coords = np.asarray(geom.exterior.coords)
            ax.add_patch(
                MplPolygon(
                    coords,
                    closed=True,
                    facecolor="#d9d9d9",
                    edgecolor="#a0a0a0",
                    linewidth=0.35,
                    alpha=0.75,
                    zorder=1,
                )
            )

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = np.asarray(poly.exterior.coords)
                ax.add_patch(
                    MplPolygon(
                        coords,
                        closed=True,
                        facecolor="#d9d9d9",
                        edgecolor="#a0a0a0",
                        linewidth=0.35,
                        alpha=0.75,
                        zorder=1,
                    )
                )


def plot_tree_edges(ax, tree_edges, color="#9ecae1", linewidth=0.30, alpha=0.18, label=None):
    for i, e in enumerate(tree_edges):
        p1 = e["from"]
        p2 = e["to"]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
            label=label,
        )
        # 只给第一条边添加label，避免图例重复
        label = None


def plot_invalid_edges(ax, invalid_edges, color="red", linewidth=1.15, alpha=1, label=None):
    """
    无效扩展画得更明显。
    """
    if len(invalid_edges) == 0:
        print(f"[Warning] No invalid edges to plot for label: {label}")
        return

    print(f"[Plotting] {len(invalid_edges)} invalid edges for {label if label else 'unlabeled'}")

    for i, e in enumerate(invalid_edges):
        p1 = e["from"]
        p2 = e["to"]

        if p1 is None or p2 is None:
            continue

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle="--",
            zorder=4,  # 提高层级，确保在树上面
            label=label,
        )
        # 只给第一条边添加label，避免图例重复
        label = None


def plot_path(ax, path_xyz, color, label, linewidth=2.8, linestyle="-", alpha=1):
    ax.plot(
        path_xyz[:, 0],
        path_xyz[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=5,
        alpha=alpha,
    )
    ax.scatter(
        path_xyz[:, 0],
        path_xyz[:, 1],
        color=color,
        s=10,
        zorder=6,
        alpha=alpha,
    )


def setup_axis(ax, meta, title):
    ax.set_xlim(meta["x_min"], meta["x_max"])
    ax.set_ylim(meta["y_min"], meta["y_max"])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)


def add_start_goal(ax, start, goal):
    ax.scatter(start[0], start[1], s=85, marker="o", color="#4A90E2", zorder=7, label="Start")
    ax.scatter(goal[0], goal[1], s=110, marker="*", color="#cb181d", zorder=7, label="Goal")



def plot_compare(
    gdf,
    meta,
    start,
    goal,
    base_path,
    base_edges,
    base_invalid_edges,
    base_result,
    rl_path,
    rl_edges,
    rl_invalid_edges,
    rl_result,
    outfile,
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 绘制建筑
    plot_buildings(ax, gdf)

    # 绘制baseline的树（中蓝色，更明显）
    plot_tree_edges(ax, base_edges, color="#4292c6", linewidth=0.5, alpha=1, label="Baseline RRT Tree")

    # 绘制RL的树（深橙色，更明显）
    plot_tree_edges(ax, rl_edges, color="#e6550d", linewidth=0.5, alpha=1, label="RRT + RL Tree")

    # 绘制baseline的无效边（已禁用显示）
    # plot_invalid_edges(ax, base_invalid_edges, color="green", linewidth=1.2, alpha=1.0, label="Baseline Invalid Expansions")

    # 绘制RL的无效边（已禁用显示）
    # plot_invalid_edges(ax, rl_invalid_edges, color="magenta", linewidth=1.2, alpha=1.0, label="RL Invalid Expansions")

    # 绘制两条路径（使用不同颜色和线型）
    plot_path(ax, base_path, color="#08519c", label="Baseline RRT Path", linewidth=2.5, linestyle="-", alpha=1)
    plot_path(ax, rl_path, color="#d95f0e", label="RRT + RL Path", linewidth=2.5, linestyle="--", alpha=1)

    # 添加起终点
    add_start_goal(ax, start, goal)

    # 设置坐标轴
    ax.set_xlim(meta["x_min"], meta["x_max"])
    ax.set_ylim(meta["y_min"], meta["y_max"])
    ax.set_aspect("equal")
    # ax.set_title("Baseline RRT vs RRT + RL Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    # 输出矢量图（SVG和PDF）
    svg_outfile = outfile.rsplit('.', 1)[0] + '.svg'
    pdf_outfile = outfile.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(svg_outfile, format='svg', bbox_inches='tight')
    plt.savefig(pdf_outfile, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"[Done] saved: {svg_outfile}")
    print(f"[Done] saved: {pdf_outfile}")


def plot_compare_1x4(
    gdf,
    meta,
    results_list,  # List of dicts containing run results
    outfile,
):
    """
    绘制1x4子图，每个子图显示一次运行结果
    results_list: List of dicts with keys: start, goal, base_path, base_edges, rl_path, rl_edges
    """
    # 设置全局字体为 serif（Times 风格）
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times', 'DejaVu Serif', 'Liberation Serif', 'serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()

    subtitle_labels = ['a', 'b', 'c', 'd']

    for idx, result in enumerate(results_list):
        ax = axes[idx]

        # 绘制建筑
        plot_buildings(ax, gdf)

        # 绘制baseline的树
        plot_tree_edges(ax, result["base_edges"], color="#4292c6", linewidth=0.5, alpha=1,
                       label="Baseline RRT Tree" if idx == 0 else None)

        # 绘制RL的树
        plot_tree_edges(ax, result["rl_edges"], color="#e6550d", linewidth=0.5, alpha=1,
                       label="RRT + RL Tree" if idx == 0 else None)

        # 绘制两条路径
        plot_path(ax, result["base_path"], color="#08519c",
                 label="Baseline RRT Path" if idx == 0 else None,
                 linewidth=2.5, linestyle="-", alpha=1)
        plot_path(ax, result["rl_path"], color="#d95f0e",
                 label="RRT + RL Path" if idx == 0 else None,
                 linewidth=2.5, linestyle="--", alpha=1)

        # 添加起终点
        add_start_goal(ax, result["start"], result["goal"])

        # 设置坐标轴
        ax.set_xlim(meta["x_min"], meta["x_max"])
        ax.set_ylim(meta["y_min"], meta["y_max"])
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.25)

    # 在整个figure上方添加图例（水平平铺）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#4292c6', lw=0.5, label='Baseline RRT Tree'),
        Line2D([0], [0], color='#e6550d', lw=0.5, label='RRT + RL Tree'),
        Line2D([0], [0], color='#08519c', lw=2.5, linestyle='-', label='Baseline RRT Path'),
        Line2D([0], [0], color='#d95f0e', lw=2.5, linestyle='--', label='RRT + RL Path'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.98),
               ncol=4, fontsize=12, frameon=False)

    # 在每个子图外面下方添加小标题 (a, b, c, d)
    for idx, label in enumerate(subtitle_labels):
        # 使用子图坐标系，在x轴标签下方
        axes[idx].text(0.5, -0.12, f'({label})',
                      transform=axes[idx].transAxes,
                      fontsize=14, ha='center', va='top')

    # 调整子图间距，为上方图例和底部标题留出空间
    plt.subplots_adjust(wspace=0.2, top=0.94, bottom=0.15)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    # 输出矢量图（SVG和PDF）
    svg_outfile = outfile.rsplit('.', 1)[0] + '.svg'
    pdf_outfile = outfile.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(svg_outfile, format='svg', bbox_inches='tight')
    plt.savefig(pdf_outfile, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"[Done] saved: {svg_outfile}")
    print(f"[Done] saved: {pdf_outfile}")


# =========================================================
# 6. 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--outfile", type=str, default="results/vis/random_compare_2d_1x4.svg")
    parser.add_argument("--num-runs", type=int, default=4, help="Number of runs to plot (default: 4)")
    parser.add_argument("--max-tries-per-run", type=int, default=40, help="Max trials to find a successful run")
    parser.add_argument("--min-dist", type=float, default=1000.0)
    parser.add_argument("--max-dist", type=float, default=1600.0)
    args = parser.parse_args()

    # 不固定随机种子
    gdf, meta = load_buildings(BUILDING_FILE)
    obstacles = gdf_to_obstacles(gdf)

    planner_kwargs_base = dict(
        obstacles=obstacles,
        x_min=meta["x_min"],
        x_max=meta["x_max"],
        y_min=meta["y_min"],
        y_max=meta["y_max"],
        z_min=meta["z_min"],
        z_max=meta["z_max"],
        step_size=45.0,
        goal_sample_rate=0.06,
        max_iter=1200,
        goal_tolerance=35.0,
        resolution=5.0,
        duplicate_threshold=22.5,
        min_progress=6.75,
    )

    results_list = []

    for run_idx in range(args.num_runs):
        # 每次运行使用不同的随机种子
        planner_kwargs = planner_kwargs_base.copy()
        planner_kwargs["random_seed"] = random.randint(0, 10_000_000)

        print(f"\n[Run {run_idx + 1}/{args.num_runs}] Looking for valid start/goal...")

        success = False
        for trial in range(args.max_tries_per_run):
            start, goal = sample_start_goal(
                obstacles=obstacles,
                meta=meta,
                min_dist=args.min_dist,
                max_dist=args.max_dist,
            )

            print(f"[Run {run_idx + 1}, Trial {trial + 1}] start={start}, goal={goal}")

            try:
                base_path, base_edges, base_invalid_edges, base_result = run_baseline(
                    planner_kwargs, start, goal
                )
                rl_path, rl_edges, rl_invalid_edges, rl_result = run_rl(
                    args.model, planner_kwargs, start, goal
                )

                print("[Baseline]", base_result)
                print("[RL]", rl_result)
                print(f"[Debug] Baseline tree edges: {len(base_edges)}, invalid edges: {len(base_invalid_edges)}")
                print(f"[Debug] RL tree edges: {len(rl_edges)}, invalid edges: {len(rl_invalid_edges)}")

                results_list.append({
                    "start": start,
                    "goal": goal,
                    "base_path": base_path,
                    "base_edges": base_edges,
                    "rl_path": rl_path,
                    "rl_edges": rl_edges,
                })

                success = True
                break

            except RuntimeError as e:
                print(f"[Run {run_idx + 1}, Trial {trial + 1}] failed: {e}")

        if not success:
            print(f"[Warning] Run {run_idx + 1} failed after {args.max_tries_per_run} trials")
            if len(results_list) == 0:
                raise RuntimeError("Failed to find any successful runs. Please increase --max-tries-per-run.")

    # 绘制1x4子图
    if len(results_list) > 0:
        plot_compare_1x4(
            gdf=gdf,
            meta=meta,
            results_list=results_list,
            outfile=args.outfile,
        )
    else:
        raise RuntimeError("No successful runs to plot.")


if __name__ == "__main__":
    main()