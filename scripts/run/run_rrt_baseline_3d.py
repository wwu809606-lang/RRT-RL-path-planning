import json
import sys
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np

from src.planners.rrt_3d import RRTPlanner3D
from src.utils.geometry import dist_3d


# =========================================================
# 配置区
# =========================================================
CONFIG = {
    "infile": "data/processed/buildings_keep_15_recommended.geojson",
    "result_dir": "results/rrt_multi_seed",

    # 多 seed
    "num_tasks": 50,
    "rrt_seed": 42,   # 固定RRT内部随机性（保证公平）

    # 空域范围
    "z_min": 20.0,
    "z_max": 220.0,

    # RRT 参数（保持固定）
    "step_size": 45.0,
    "goal_sample_rate": 0.06,
    "max_iter": 12000,
    "goal_tolerance": 35.0,
    "collision_resolution": 5.0,

    # 地图参数
    "obstacle_buffer": 8.0,
    "map_margin": 80.0,

    # 任务约束
    "min_start_goal_dist": 300.0,
}


# =========================================================
# 数据读取
# =========================================================
def load_buildings_local(infile):
    gdf = gpd.read_file(infile)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf = gdf.to_crs(epsg=3857)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if "height_m" not in gdf.columns:
        raise ValueError("缺少 height_m")

    gdf["height_m"] = gdf["height_m"].astype(float)
    gdf = gdf[np.isfinite(gdf["height_m"])].copy()

    meta = {
        "x_range": xmax - xmin,
        "y_range": ymax - ymin,
        "z_max_data": float(gdf["height_m"].max()),
    }
    return gdf, meta


def prepare_obstacles_3d(gdf, obstacle_buffer):
    obstacles = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row["height_m"])
        buffered = geom.buffer(obstacle_buffer)

        if buffered.geom_type == "Polygon":
            obstacles.append({"polygon": buffered, "height": h})
        elif buffered.geom_type == "MultiPolygon":
            for poly in buffered.geoms:
                obstacles.append({"polygon": poly, "height": h})

    return obstacles


# =========================================================
# 任务生成（核心）
# =========================================================
def point_in_collision_3d(p, obstacles):
    from shapely.geometry import Point
    pt2d = Point(p[0], p[1])
    z = p[2]

    for obs in obstacles:
        if z <= obs["height"] and obs["polygon"].intersects(pt2d):
            return True
    return False


def sample_start_goal(meta, z_min, z_max, obstacles, seed, min_dist):
    np.random.seed(seed)

    while True:
        p1 = (
            np.random.uniform(0, meta["x_range"]),
            np.random.uniform(0, meta["y_range"]),
            np.random.uniform(z_min, z_max),
        )
        p2 = (
            np.random.uniform(0, meta["x_range"]),
            np.random.uniform(0, meta["y_range"]),
            np.random.uniform(z_min, z_max),
        )

        if point_in_collision_3d(p1, obstacles):
            continue
        if point_in_collision_3d(p2, obstacles):
            continue
        if dist_3d(p1, p2) < min_dist:
            continue

        return p1, p2


# =========================================================
# 主函数
# =========================================================
def main():
    cfg = CONFIG
    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print("加载地图...")
    gdf, meta = load_buildings_local(cfg["infile"])
    obstacles = prepare_obstacles_3d(gdf, cfg["obstacle_buffer"])

    x_min = -cfg["map_margin"]
    x_max = meta["x_range"] + cfg["map_margin"]
    y_min = -cfg["map_margin"]
    y_max = meta["y_range"] + cfg["map_margin"]

    all_results = []

    print(f"开始 multi-seed 评估，共 {cfg['num_tasks']} 个任务")

    for task_seed in range(cfg["num_tasks"]):
        print(f"\n=== Task {task_seed} ===")

        start_xyz, goal_xyz = sample_start_goal(
            meta,
            cfg["z_min"],
            cfg["z_max"],
            obstacles,
            seed=task_seed,
            min_dist=cfg["min_start_goal_dist"],
        )

        planner = RRTPlanner3D(
            obstacles=obstacles,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=cfg["z_min"],
            z_max=cfg["z_max"],
            step_size=cfg["step_size"],
            goal_sample_rate=cfg["goal_sample_rate"],
            max_iter=cfg["max_iter"],
            goal_tolerance=cfg["goal_tolerance"],
            resolution=cfg["collision_resolution"],
            random_seed=cfg["rrt_seed"],  # 固定算法随机性
        )

        result = planner.plan(start_xyz, goal_xyz)

        row = {
            "task_seed": task_seed,
            "success": int(result["success"]),
            "first_path_iter": result["first_path_iter"] or -1,
            "n_exp_invalid": result["n_exp_invalid"],
            "n_inc_invalid": result["n_inc_invalid"],
            "n_prog_invalid": result["n_prog_invalid"],
            "invalid_ratio": result["invalid_ratio"],
        }

        all_results.append(row)

    # =========================================================
    # 保存 CSV
    # =========================================================
    csv_path = result_dir / "multi_seed_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nCSV 已保存: {csv_path}")

    # =========================================================
    # 统计
    # =========================================================
    def compute_stats(key):
        vals = [r[key] for r in all_results if r["success"] == 1 and r[key] >= 0]
        if len(vals) == 0:
            return None, None
        arr = np.array(vals)
        return arr.mean(), arr.std()

    print("\n===== 统计结果（只统计成功样本） =====")

    for k in [
        "first_path_iter",
        "n_exp_invalid",
        "n_inc_invalid",
        "n_prog_invalid",
        "invalid_ratio",
    ]:
        mean, std = compute_stats(k)
        if mean is not None:
            print(f"{k}: {mean:.2f} ± {std:.2f}")
        else:
            print(f"{k}: 无有效数据")

    success_rate = np.mean([r["success"] for r in all_results])
    print(f"\nSuccess Rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()