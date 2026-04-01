import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

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


def plot_tree_edges(ax, tree_edges, color="#9ecae1", linewidth=0.30, alpha=0.18):
    for e in tree_edges:
        p1 = e["from"]
        p2 = e["to"]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )


def plot_invalid_edges(ax, invalid_edges, color="red", linewidth=1.15, alpha=1):
    """
    无效扩展画得更明显。
    """
    for e in invalid_edges:
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
            zorder=3,
        )


def plot_path(ax, path_xyz, color, label, linewidth=2.8, linestyle="-"):
    ax.plot(
        path_xyz[:, 0],
        path_xyz[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=5,
    )
    ax.scatter(
        path_xyz[:, 0],
        path_xyz[:, 1],
        color=color,
        s=10,
        zorder=6,
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ---------------- baseline ----------------
    ax = axes[0]
    plot_buildings(ax, gdf)
    plot_tree_edges(ax, base_edges, color="green", linewidth=0.28, alpha=1)
    plot_invalid_edges(ax, base_invalid_edges, color="green", linewidth=1.15, alpha=1)
    plot_path(ax, base_path, color="#08519c", label="Baseline Path", linewidth=2)
    add_start_goal(ax, start, goal)
    setup_axis(
        ax,
        meta,
        title=f"Baseline RRT",
    )
    ax.legend(loc="upper right")

    # ---------------- RL ----------------
    ax = axes[1]
    plot_buildings(ax, gdf)
    plot_tree_edges(ax, rl_edges, color="green", linewidth=0.28, alpha=1)
    plot_invalid_edges(ax, rl_invalid_edges, color="green", linewidth=1.15, alpha=1)
    plot_path(ax, rl_path, color="#08519c", label="RRT + RL Path", linewidth=2)
    add_start_goal(ax, start, goal)
    setup_axis(
        ax,
        meta,
        title=f"RRT + RL",
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=240, bbox_inches="tight")
    plt.show()
    print(f"[Done] saved: {outfile}")


# =========================================================
# 6. 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--outfile", type=str, default="results/vis/random_compare_2d.png")
    parser.add_argument("--max-tries", type=int, default=40)
    parser.add_argument("--min-dist", type=float, default=1000.0)
    parser.add_argument("--max-dist", type=float, default=1600.0)
    args = parser.parse_args()

    # 不固定随机种子
    gdf, meta = load_buildings(BUILDING_FILE)
    obstacles = gdf_to_obstacles(gdf)

    planner_kwargs = dict(
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
        random_seed=random.randint(0, 10_000_000),
        duplicate_threshold=22.5,
        min_progress=6.75,
    )

    for trial in range(args.max_tries):
        start, goal = sample_start_goal(
            obstacles=obstacles,
            meta=meta,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
        )

        print(f"[Trial {trial + 1}] start={start}, goal={goal}")

        try:
            base_path, base_edges, base_invalid_edges, base_result = run_baseline(
                planner_kwargs, start, goal
            )
            rl_path, rl_edges, rl_invalid_edges, rl_result = run_rl(
                args.model, planner_kwargs, start, goal
            )

            print("[Baseline]", base_result)
            print("[RL]", rl_result)

            plot_compare(
                gdf=gdf,
                meta=meta,
                start=start,
                goal=goal,
                base_path=base_path,
                base_edges=base_edges,
                base_invalid_edges=base_invalid_edges,
                base_result=base_result,
                rl_path=rl_path,
                rl_edges=rl_edges,
                rl_invalid_edges=rl_invalid_edges,
                rl_result=rl_result,
                outfile=args.outfile,
            )
            return

        except RuntimeError as e:
            print(f"[Trial {trial + 1}] failed: {e}")

    raise RuntimeError("多次随机采样后仍未找到双方都成功的任务，请增大 --max-tries。")


if __name__ == "__main__":
    main()