import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Polygon, MultiPolygon

from stable_baselines3 import PPO

from src.envs.rrt_rl_env import RRTRLEnv
from src.planners.rrt_3d import RRTPlanner3D
from src.rewards.reward_fn import RewardConfig


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_geometry_to_local_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    if gdf.crs.to_epsg() == 4326:
        gdf = gdf.to_crs(epsg=3857)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)
    return gdf


def extract_obstacles_from_geojson(building_geojson: str) -> List[Dict[str, Any]]:
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
            f"未找到建筑高度字段。请在 {building_geojson} 中确认存在以下字段之一: "
            f"{candidate_height_cols}"
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
        raise ValueError("未从 geojson 中解析出任何建筑物障碍。")

    return obstacles


def build_planner_kwargs(cfg: Dict[str, Any], obstacles) -> Dict[str, Any]:
    planner_cfg = cfg["planner"].copy()
    planner_cfg["obstacles"] = obstacles
    return planner_cfg


def build_reward_cfg(cfg: Dict[str, Any]) -> RewardConfig:
    reward_cfg = cfg.get("reward", {})
    return RewardConfig(
        goal_reward=reward_cfg.get("goal_reward", 20.0),
        valid_step_reward=reward_cfg.get("valid_step_reward", 0.2),
        progress_scale=reward_cfg.get("progress_scale", 0.05),
        expansion_invalid_penalty=reward_cfg.get("expansion_invalid_penalty", 1.0),
        increment_invalid_penalty=reward_cfg.get("increment_invalid_penalty", 0.4),
        progress_invalid_penalty=reward_cfg.get("progress_invalid_penalty", 0.8),
        timeout_penalty=reward_cfg.get("timeout_penalty", 0.0),
    )


def evaluate_baseline(
    planner_kwargs: Dict[str, Any],
    task: Dict[str, Any],
) -> Dict[str, Any]:
    planner = RRTPlanner3D(**planner_kwargs)

    start_xyz = tuple(task["start"])
    goal_xyz = tuple(task["goal"])

    t0 = time.perf_counter()
    result = planner.plan(start_xyz, goal_xyz)
    t1 = time.perf_counter()

    return {
        "success": bool(result["success"]),
        "iterations": int(result["iterations"]),
        "first_path_iter": result["first_path_iter"] if result["first_path_iter"] is not None else np.nan,
        "n_exp_invalid": int(result["n_exp_invalid"]),
        "n_inc_invalid": int(result["n_inc_invalid"]),
        "n_prog_invalid": int(result["n_prog_invalid"]),
        "invalid_ratio": float(result["invalid_ratio"]),
        "runtime_sec": float(t1 - t0),
        "path_len_nodes": len(result["path_xyz"]),
    }


def evaluate_rl_single_task(
    model: PPO,
    planner_kwargs: Dict[str, Any],
    task: Dict[str, Any],
    k_candidates: int,
    reward_cfg: RewardConfig,
    seed: int,
) -> Dict[str, Any]:
    env = RRTRLEnv(
        planner_kwargs=planner_kwargs,
        task_list=[task],
        k_candidates=k_candidates,
        reward_cfg=reward_cfg,
    )

    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    terminated = False
    truncated = False

    t0 = time.perf_counter()
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
    t1 = time.perf_counter()

    planner = env.planner
    assert planner is not None

    path_len_nodes = 0
    if planner.success and planner.goal_node_idx is not None:
        path_xyz = planner.extract_path(planner.goal_node_idx)
        path_len_nodes = len(path_xyz)

    invalid_ratio = (
        (planner.n_exp_invalid + planner.n_inc_invalid + planner.n_prog_invalid)
        / max(planner.iter_count, 1)
    )

    env.close()

    return {
        "success": bool(planner.success),
        "iterations": int(planner.iter_count),
        "first_path_iter": planner.first_path_iter if planner.first_path_iter is not None else np.nan,
        "n_exp_invalid": int(planner.n_exp_invalid),
        "n_inc_invalid": int(planner.n_inc_invalid),
        "n_prog_invalid": int(planner.n_prog_invalid),
        "invalid_ratio": float(invalid_ratio),
        "runtime_sec": float(t1 - t0),
        "path_len_nodes": int(path_len_nodes),
        "episode_reward": float(total_reward),
    }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for method_name, g in df.groupby("method"):
        success_rate = g["success"].mean()

        summary_rows.append(
            {
                "method": method_name,
                "n_tasks": len(g),
                "success_rate": success_rate,
                "iterations_mean": g["iterations"].mean(),
                "iterations_std": g["iterations"].std(ddof=0),
                "first_path_iter_mean": g["first_path_iter"].mean(),
                "first_path_iter_std": g["first_path_iter"].std(ddof=0),
                "n_exp_invalid_mean": g["n_exp_invalid"].mean(),
                "n_inc_invalid_mean": g["n_inc_invalid"].mean(),
                "n_prog_invalid_mean": g["n_prog_invalid"].mean(),
                "invalid_ratio_mean": g["invalid_ratio"].mean(),
                "invalid_ratio_std": g["invalid_ratio"].std(ddof=0),
                "runtime_sec_mean": g["runtime_sec"].mean(),
                "runtime_sec_std": g["runtime_sec"].std(ddof=0),
                "path_len_nodes_mean": g["path_len_nodes"].mean(),
            }
        )

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_train.yaml",
        help="与训练时相同的配置文件",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/rl/best_model/best_model.zip",
        help="训练好的 PPO 模型路径",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="覆盖配置文件中的测试任务路径；留空则读取 config[data][test_tasks]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rl_eval",
        help="评估结果输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="评估随机种子",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    cfg = load_yaml(args.config)

    building_geojson = cfg["data"]["building_geojson"]
    obstacles = extract_obstacles_from_geojson(building_geojson)

    planner_kwargs = build_planner_kwargs(cfg, obstacles)
    reward_cfg = build_reward_cfg(cfg)
    k_candidates = int(cfg["env"].get("k_candidates", 8))

    test_tasks_path = args.tasks if args.tasks else cfg["data"]["test_tasks"]
    test_tasks = load_json(test_tasks_path)
    if len(test_tasks) == 0:
        raise ValueError("测试任务集为空。")

    model = PPO.load(args.model_path)

    rows: List[Dict[str, Any]] = []

    for i, task in enumerate(test_tasks):
        baseline_metrics = evaluate_baseline(planner_kwargs, task)
        rows.append(
            {
                "task_id": i,
                "method": "baseline_rrt",
                "success": baseline_metrics["success"],
                "iterations": baseline_metrics["iterations"],
                "first_path_iter": baseline_metrics["first_path_iter"],
                "n_exp_invalid": baseline_metrics["n_exp_invalid"],
                "n_inc_invalid": baseline_metrics["n_inc_invalid"],
                "n_prog_invalid": baseline_metrics["n_prog_invalid"],
                "invalid_ratio": baseline_metrics["invalid_ratio"],
                "runtime_sec": baseline_metrics["runtime_sec"],
                "path_len_nodes": baseline_metrics["path_len_nodes"],
            }
        )

        rl_metrics = evaluate_rl_single_task(
            model=model,
            planner_kwargs=planner_kwargs,
            task=task,
            k_candidates=k_candidates,
            reward_cfg=reward_cfg,
            seed=args.seed + i,
        )
        rows.append(
            {
                "task_id": i,
                "method": "rrt_rl",
                "success": rl_metrics["success"],
                "iterations": rl_metrics["iterations"],
                "first_path_iter": rl_metrics["first_path_iter"],
                "n_exp_invalid": rl_metrics["n_exp_invalid"],
                "n_inc_invalid": rl_metrics["n_inc_invalid"],
                "n_prog_invalid": rl_metrics["n_prog_invalid"],
                "invalid_ratio": rl_metrics["invalid_ratio"],
                "runtime_sec": rl_metrics["runtime_sec"],
                "path_len_nodes": rl_metrics["path_len_nodes"],
                "episode_reward": rl_metrics["episode_reward"],
            }
        )

        print(
            f"[{i+1}/{len(test_tasks)}] "
            f"baseline success={baseline_metrics['success']}, "
            f"rl success={rl_metrics['success']}"
        )

    df = pd.DataFrame(rows)
    summary_df = summarize_results(df)

    per_task_path = os.path.join(args.output_dir, "per_task_results.csv")
    summary_path = os.path.join(args.output_dir, "comparison_summary.csv")

    df.to_csv(per_task_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n===== Summary =====")
    print(summary_df)
    print(f"\n[Saved] per-task results: {per_task_path}")
    print(f"[Saved] summary: {summary_path}")


if __name__ == "__main__":
    main()