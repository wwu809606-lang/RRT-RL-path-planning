import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import yaml
from shapely.geometry import MultiPolygon, Polygon

from src.planners.rrt_3d import point_in_collision_3d
from src.utils.geometry import dist_3d


XYZ = Tuple[float, float, float]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_geometry_to_local_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    if gdf.crs.to_epsg() == 4326:
        gdf = gdf.to_crs(epsg=3857)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)
    return gdf


def extract_obstacles_and_bounds(building_geojson: str):
    gdf = gpd.read_file(building_geojson)
    gdf = normalize_geometry_to_local_xy(gdf)

    candidate_height_cols = ["height", "Height", "building_h", "elevation", "h"]
    height_col = None
    for col in candidate_height_cols:
        if col in gdf.columns:
            height_col = col
            break

    if height_col is None:
        raise ValueError(
            f"未找到建筑高度字段。请确认存在以下字段之一: {candidate_height_cols}"
        )

    obstacles: List[Dict[str, Any]] = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row[height_col])

        if geom is None or geom.is_empty:
            continue

        if isinstance(geom, Polygon):
            obstacles.append({"polygon": geom, "height": h})
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                if not poly.is_empty:
                    obstacles.append({"polygon": poly, "height": h})

    if len(obstacles) == 0:
        raise ValueError("未解析出任何障碍物。")

    xmin, ymin, xmax, ymax = gdf.total_bounds
    max_h = float(gdf[height_col].max())

    bounds = {
        "x_min": float(xmin),
        "x_max": float(xmax),
        "y_min": float(ymin),
        "y_max": float(ymax),
        "z_min": 0.0,
        "z_max": max(max_h + 80.0, 120.0),
    }

    return obstacles, bounds


def sample_free_point(
    rng: np.random.Generator,
    obstacles,
    bounds: Dict[str, float],
    z_low_ratio: float = 0.25,
    z_high_ratio: float = 0.85,
) -> XYZ:
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    z_min, z_max = bounds["z_min"], bounds["z_max"]

    z_low = z_min + z_low_ratio * (z_max - z_min)
    z_high = z_min + z_high_ratio * (z_max - z_min)

    while True:
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        z = float(rng.uniform(z_low, z_high))
        p = (x, y, z)

        if not point_in_collision_3d(p, obstacles):
            return p


def make_single_task(
    rng: np.random.Generator,
    obstacles,
    bounds: Dict[str, float],
    min_start_goal_dist: float,
    max_trials: int = 500,
) -> Dict[str, List[float]]:
    for _ in range(max_trials):
        start = sample_free_point(rng, obstacles, bounds)
        goal = sample_free_point(rng, obstacles, bounds)

        d = dist_3d(start, goal)
        if d >= min_start_goal_dist:
            return {
                "start": [float(start[0]), float(start[1]), float(start[2])],
                "goal": [float(goal[0]), float(goal[1]), float(goal[2])],
            }

    raise RuntimeError(
        "在给定最大尝试次数内未能生成满足最小起终点距离约束的任务，请降低 min_start_goal_dist。"
    )


def split_tasks(
    tasks: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    n = len(tasks)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_tasks = tasks[:n_train]
    val_tasks = tasks[n_train : n_train + n_val]
    test_tasks = tasks[n_train + n_val : n_train + n_val + n_test]

    return train_tasks, val_tasks, test_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_train.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=300,
        help="总任务数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--min-dist-ratio",
        type=float,
        default=0.35,
        help="最小起终点距离占空间对角线比例",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="训练集比例",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="验证集比例",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="测试集比例",
    )
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    cfg = load_yaml(args.config)
    building_geojson = cfg["data"]["building_geojson"]

    obstacles, inferred_bounds = extract_obstacles_and_bounds(building_geojson)

    planner_cfg = cfg["planner"]
    bounds = {
        "x_min": float(planner_cfg.get("x_min", inferred_bounds["x_min"])),
        "x_max": float(planner_cfg.get("x_max", inferred_bounds["x_max"])),
        "y_min": float(planner_cfg.get("y_min", inferred_bounds["y_min"])),
        "y_max": float(planner_cfg.get("y_max", inferred_bounds["y_max"])),
        "z_min": float(planner_cfg.get("z_min", inferred_bounds["z_min"])),
        "z_max": float(planner_cfg.get("z_max", inferred_bounds["z_max"])),
    }

    dx = bounds["x_max"] - bounds["x_min"]
    dy = bounds["y_max"] - bounds["y_min"]
    dz = bounds["z_max"] - bounds["z_min"]
    space_diag = float((dx * dx + dy * dy + dz * dz) ** 0.5)
    min_start_goal_dist = args.min_dist_ratio * space_diag

    rng = np.random.default_rng(args.seed)

    tasks = []
    for i in range(args.n_tasks):
        task = make_single_task(
            rng=rng,
            obstacles=obstacles,
            bounds=bounds,
            min_start_goal_dist=min_start_goal_dist,
        )
        tasks.append(task)

        if (i + 1) % 50 == 0 or (i + 1) == args.n_tasks:
            print(f"[Task Generation] {i+1}/{args.n_tasks}")

    rng.shuffle(tasks)

    train_tasks, val_tasks, test_tasks = split_tasks(
        tasks=tasks,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_path = cfg["data"]["train_tasks"]
    val_path = cfg["data"]["val_tasks"]
    test_path = cfg["data"]["test_tasks"]

    ensure_dir(str(Path(train_path).parent))

    save_json(train_tasks, train_path)
    save_json(val_tasks, val_path)
    save_json(test_tasks, test_path)

    stats = {
        "n_total": len(tasks),
        "n_train": len(train_tasks),
        "n_val": len(val_tasks),
        "n_test": len(test_tasks),
        "seed": args.seed,
        "min_start_goal_dist": min_start_goal_dist,
        "space_diag": space_diag,
        "bounds": bounds,
    }
    stats_path = str(Path(train_path).parent / "task_generation_stats.json")
    save_json(stats, stats_path)

    print("\n===== Task Dataset Summary =====")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[Saved] train: {train_path}")
    print(f"[Saved] val:   {val_path}")
    print(f"[Saved] test:  {test_path}")
    print(f"[Saved] stats: {stats_path}")


if __name__ == "__main__":
    main()