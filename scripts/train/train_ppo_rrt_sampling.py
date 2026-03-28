import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import yaml
from shapely.geometry import Polygon, MultiPolygon

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.envs.rrt_rl_env import RRTRLEnv
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
    """
    与你前面的空域建模脚本保持一致：
    若是经纬度，则转 EPSG:3857；
    再整体平移到左下角为原点附近。
    """
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    # 若还是地理坐标，先投影
    if gdf.crs.to_epsg() == 4326:
        gdf = gdf.to_crs(epsg=3857)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)
    return gdf


def extract_obstacles_from_geojson(building_geojson: str) -> List[Dict[str, Any]]:
    """
    读取建筑物 geojson，生成 planner 所需 obstacles:
    [
        {"polygon": shapely_polygon, "height": float},
        ...
    ]
    高度字段默认优先尝试:
    height / Height / building_h / elevation / h
    """
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


def make_env_fn(
    planner_kwargs: Dict[str, Any],
    task_list: List[Dict[str, Any]],
    k_candidates: int,
    reward_cfg: RewardConfig,
    monitor_dir: str,
    rank: int,
):
    def _init():
        env = RRTRLEnv(
            planner_kwargs=planner_kwargs,
            task_list=task_list,
            k_candidates=k_candidates,
            reward_cfg=reward_cfg,
        )
        monitor_path = os.path.join(monitor_dir, f"train_env_{rank}.csv")
        env = Monitor(env, filename=monitor_path)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_train.yaml",
        help="训练配置文件路径",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    output_dir = cfg.get("output_dir", "results/rl")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    tb_dir = os.path.join(output_dir, "tensorboard")
    monitor_dir = os.path.join(output_dir, "monitor")

    for p in [output_dir, ckpt_dir, log_dir, tb_dir, monitor_dir]:
        ensure_dir(p)

    # 1) 读取障碍物
    building_geojson = cfg["data"]["building_geojson"]
    obstacles = extract_obstacles_from_geojson(building_geojson)

    # 2) 构造 planner / reward
    planner_kwargs = build_planner_kwargs(cfg, obstacles)
    reward_cfg = build_reward_cfg(cfg)

    # 3) 读取任务集
    train_tasks = load_json(cfg["data"]["train_tasks"])
    val_tasks = load_json(cfg["data"]["val_tasks"])

    if len(train_tasks) == 0:
        raise ValueError("训练任务集为空。")
    if len(val_tasks) == 0:
        raise ValueError("验证任务集为空。")

    # 4) 构建训练环境
    n_envs = int(cfg["train"].get("n_envs", 4))
    k_candidates = int(cfg["env"].get("k_candidates", 8))

    env_fns = [
        make_env_fn(
            planner_kwargs=planner_kwargs,
            task_list=train_tasks,
            k_candidates=k_candidates,
            reward_cfg=reward_cfg,
            monitor_dir=monitor_dir,
            rank=i,
        )
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    # 5) 构建验证环境
    eval_env = DummyVecEnv(
        [
            lambda: Monitor(
                RRTRLEnv(
                    planner_kwargs=planner_kwargs,
                    task_list=val_tasks,
                    k_candidates=k_candidates,
                    reward_cfg=reward_cfg,
                )
            )
        ]
    )
    eval_env = VecMonitor(eval_env)

    # 6) PPO 参数
    ppo_cfg = cfg["ppo"]

    model = PPO(
        policy=ppo_cfg.get("policy", "MlpPolicy"),
        env=train_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 1024)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        verbose=int(ppo_cfg.get("verbose", 1)),
        tensorboard_log=tb_dir,
        device=ppo_cfg.get("device", "auto"),
        seed=int(cfg["train"].get("seed", 42)),
    )

    # 7) 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg["train"].get("checkpoint_freq", 20000)) // max(n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_rrt_sampling",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=log_dir,
        eval_freq=int(cfg["train"].get("eval_freq", 10000)) // max(n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=int(cfg["train"].get("n_eval_episodes", 20)),
    )

    # 8) 训练
    total_timesteps = int(cfg["train"].get("total_timesteps", 300000))
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="ppo_rrt_sampling",
        progress_bar=True,
    )

    # 9) 保存最终模型
    final_model_path = os.path.join(output_dir, "ppo_rrt_sampling_final")
    model.save(final_model_path)

    # 10) 保存一份训练配置副本
    with open(os.path.join(output_dir, "used_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[Done] final model saved to: {final_model_path}.zip")
    print(f"[Done] best model dir: {os.path.join(output_dir, 'best_model')}")
    print(f"[Done] logs saved to: {output_dir}")


if __name__ == "__main__":
    main()