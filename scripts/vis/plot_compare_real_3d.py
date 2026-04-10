import argparse
import json
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
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from stable_baselines3 import PPO

from src.planners.rrt_3d import RRTPlanner3D, point_in_collision_3d
from src.envs.rrt_rl_env import RRTRLEnv
from src.rewards.reward_fn import RewardConfig


XYZ = Tuple[float, float, float]

BUILDING_FILE = "data/processed/buildings_keep_15_recommended.geojson"
DEFAULT_MODEL = "results/rl/best_model/best_model.zip"

# 设置全局字体为 serif（Times 风格）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times', 'DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


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
# 2. 随机起点终点
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
        z = random.uniform(max(z_min, 20.0), min(z_max, 150.0))  # 提高上限到150m
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
    采样相距较远的起终点，确保至少有一个点高度较高。
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

        # 确保至少有一个点高度 > 100m
        if max(start[2], goal[2]) < 100.0:
            continue

        d = ((start[0] - goal[0]) ** 2 +
             (start[1] - goal[1]) ** 2 +
             (start[2] - goal[2]) ** 2) ** 0.5

        if min_dist <= d <= max_dist:
            return start, goal

    raise RuntimeError("无法采样到满足距离要求的起终点，请调小 min_dist 或增大 max_trials。")


# =========================================================
# 3. baseline：运行（记录无效扩展）
# =========================================================
def run_baseline(planner_kwargs: Dict[str, Any], start: XYZ, goal: XYZ):
    t0 = time.perf_counter()

    planner = RRTPlanner3D(**planner_kwargs)
    planner.reset_tree(start, goal)

    invalid_edges = []  # 记录无效扩展

    for _ in range(planner.max_iter):
        rnd = planner.sample_free(goal)
        res = planner.extend_once(rnd)

        # 记录所有无效扩展
        if res["status"] in ["expansion_invalid", "increment_invalid", "progress_invalid"]:
            invalid_edges.append({
                "from": res["nearest_xyz"],
                "to": res["new_xyz"],
                "status": res["status"],
            })

        if res["goal_reached"]:
            elapsed = time.perf_counter() - t0

            path_xyz = np.array(planner.extract_path(planner.goal_node_idx), dtype=float)
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
            return path_xyz, invalid_edges, result

        if planner.iter_count >= planner.max_iter:
            break

    raise RuntimeError("Baseline RRT failed on this task.")


# =========================================================
# 4. RL：运行（记录无效扩展）
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
    invalid_edges = []  # 记录无效扩展

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))

        # 记录所有无效扩展
        step_result = info["step_result"]
        if step_result["status"] in ["expansion_invalid", "increment_invalid", "progress_invalid"]:
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

    return path_xyz, invalid_edges, result


# =========================================================
# 5. 3D绘图函数
# =========================================================
def plot_buildings_3d(ax, gdf):
    """绘制3D建筑（半透明）"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row["height_m"])

        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            coords = list(zip(x, y))

            # 顶面
            top_face = [(px, py, h) for px, py in coords]
            ax.add_collection3d(
                Poly3DCollection(
                    [top_face],
                    facecolors="#d9d9d9",
                    edgecolors="#a0a0a0",
                    linewidths=0.35,
                    alpha=0.35,  # 降低透明度，可以看到后面的路径
                    zorder=1,
                )
            )

            # 立面
            side_faces = []
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                side_faces.append([
                    (x1, y1, 0),
                    (x2, y2, 0),
                    (x2, y2, h),
                    (x1, y1, h)
                ])

            ax.add_collection3d(
                Poly3DCollection(
                    side_faces,
                    facecolors="#d9d9d9",
                    edgecolors="#a0a0a0",
                    linewidths=0.35,
                    alpha=0.35,  # 降低透明度
                    zorder=1,
                )
            )

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                coords = list(zip(x, y))

                # 顶面
                top_face = [(px, py, h) for px, py in coords]
                ax.add_collection3d(
                    Poly3DCollection(
                        [top_face],
                        facecolors="#d9d9d9",
                        edgecolors="#a0a0a0",
                        linewidths=0.35,
                        alpha=0.35,  # 降低透明度
                        zorder=1,
                    )
                )

                # 立面
                side_faces = []
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    side_faces.append([
                        (x1, y1, 0),
                        (x2, y2, 0),
                        (x2, y2, h),
                        (x1, y1, h)
                    ])

                ax.add_collection3d(
                    Poly3DCollection(
                        side_faces,
                        facecolors="#d9d9d9",
                        edgecolors="#a0a0a0",
                        linewidths=0.35,
                        alpha=0.35,  # 降低透明度
                        zorder=1,
                    )
                )


def plot_path_3d(ax, path_xyz, color, label, linewidth=2.5, alpha=1):
    """绘制3D路径"""
    ax.plot(
        path_xyz[:, 0],
        path_xyz[:, 1],
        path_xyz[:, 2],
        color=color,
        linewidth=linewidth,
        linestyle="-",  # 始终使用实线
        label=label,
        zorder=5,
        alpha=alpha,
    )
    # 绘制路径点
    ax.scatter(
        path_xyz[:, 0],
        path_xyz[:, 1],
        path_xyz[:, 2],
        color=color,
        s=8,
        zorder=6,
        alpha=alpha,
    )


def plot_invalid_edges_3d(ax, invalid_edges, color, linewidth=1.0, alpha=0.6):
    """绘制3D无效扩展边"""
    if len(invalid_edges) == 0:
        return

    for edge in invalid_edges:
        p1 = edge["from"]
        p2 = edge["to"]
        if p1 is None or p2 is None:
            continue

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color,
            linewidth=linewidth,
            linestyle="-",  # 细实线
            alpha=alpha,
            zorder=3,  # 在建筑上面，在路径下面
        )


def add_start_goal_3d(ax, start, goal):
    """添加起终点标记"""
    ax.scatter(start[0], start[1], start[2], s=100, marker="o", color="#4A90E2", zorder=7, label="Start")
    ax.scatter(goal[0], goal[1], goal[2], s=150, marker="*", color="#cb181d", zorder=7, label="Goal")


# =========================================================
# 5.5 Blender导出函数
# =========================================================
def geometry_to_coords(geom):
    """将 shapely 几何转换为坐标列表"""
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        return list(zip(x, y))
    elif geom.geom_type == "MultiPolygon":
        coords = []
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            coords.append(list(zip(x, y)))
        return coords
    return []


def export_scene_for_blender(
    gdf,
    meta,
    start,
    goal,
    base_path,
    base_invalid_edges,
    base_result,
    rl_path,
    rl_invalid_edges,
    rl_result,
    outfile,
):
    """
    将场景数据导出为 JSON，供 Blender 导入使用。
    """
    # 转换建筑数据
    buildings = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row["height_m"])
        buildings.append({
            "coordinates": geometry_to_coords(geom),
            "height": h,
        })

    # 转换路径数据（转为列表格式）
    base_path_list = base_path.tolist() if hasattr(base_path, 'tolist') else list(base_path)
    rl_path_list = rl_path.tolist() if hasattr(rl_path, 'tolist') else list(rl_path)

    # 构建 JSON 数据结构
    scene_data = {
        "metadata": {
            "version": "1.0",
            "description": "UAV path planning scene for Blender import",
            "source": "plot_compare_real_3d.py",
        },
        "bounds": meta,
        "start": list(start),
        "goal": list(goal),
        "buildings": buildings,
        "paths": {
            "baseline": {
                "coordinates": base_path_list,
                "result": base_result,
                "invalid_edges": base_invalid_edges,
            },
            "rl": {
                "coordinates": rl_path_list,
                "result": rl_result,
                "invalid_edges": rl_invalid_edges,
            },
        },
    }

    # 写入 JSON 文件
    json_outfile = outfile.rsplit('.', 1)[0] + '_blender.json'
    Path(json_outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(json_outfile, 'w', encoding='utf-8') as f:
        json.dump(scene_data, f, indent=2, ensure_ascii=False)

    print(f"[Done] Blender scene exported: {json_outfile}")
    return json_outfile


def plot_compare_3d(
    gdf,
    meta,
    start,
    goal,
    base_path,
    base_invalid_edges,
    base_result,
    rl_path,
    rl_invalid_edges,
    rl_result,
    outfile,
):
    """绘制3D对比图"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制建筑（半透明）
    plot_buildings_3d(ax, gdf)

    # 绘制baseline无效扩展（蓝色细实线）
    plot_invalid_edges_3d(ax, base_invalid_edges, color="#4292c6", linewidth=1.2, alpha=0.7)

    # 绘制RL无效扩展（橙色细实线）
    plot_invalid_edges_3d(ax, rl_invalid_edges, color="#e6550d", linewidth=1.2, alpha=0.7)

    # 绘制两条路径（粗实线）
    plot_path_3d(ax, base_path, color="#08519c", label="Baseline RRT Path", linewidth=3.0, alpha=1)
    plot_path_3d(ax, rl_path, color="#d95f0e", label="RRT + RL Path", linewidth=3.0, alpha=1)

    # 添加起终点
    add_start_goal_3d(ax, start, goal)

    # 设置坐标轴
    ax.set_xlim(meta["x_min"], meta["x_max"])
    ax.set_ylim(meta["y_min"], meta["y_max"])
    ax.set_zlim(meta["z_min"], meta["z_max"])
    ax.set_xlabel("X (m)", labelpad=10)
    ax.set_ylabel("Y (m)", labelpad=10)
    ax.set_zlabel("Z (m)", labelpad=10)

    # 视角：斜俯视
    ax.view_init(elev=30, azim=-45)

    # 控制三轴比例
    x_range = meta["x_max"] - meta["x_min"]
    y_range = meta["y_max"] - meta["y_min"]
    z_range = meta["z_max"] - meta["z_min"]
    try:
        ax.set_box_aspect((x_range, y_range, z_range * 2))
    except Exception:
        pass

    # 面板简洁
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 图例（包含所有6项）
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    legend_elements = [
        Line2D([0], [0], color="#08519c", lw=3.0, linestyle="-", label="Baseline RRT Path"),
        Line2D([0], [0], color="#4292c6", lw=1.2, linestyle="-", label="Baseline Invalid Expansions"),
        Line2D([0], [0], color="#d95f0e", lw=3.0, linestyle="-", label="RRT + RL Path"),
        Line2D([0], [0], color="#e6550d", lw=1.2, linestyle="-", label="RRT + RL Invalid Expansions"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4A90E2", markersize=10, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#cb181d", markersize=15, label="Goal"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=15, framealpha=0.9)

    # 调整布局，压缩空白
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # 输出SVG和PDF
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    svg_outfile = outfile.rsplit('.', 1)[0] + '.svg'
    pdf_outfile = outfile.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(svg_outfile, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(pdf_outfile, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.show()
    print(f"[Done] saved: {svg_outfile}")
    print(f"[Done] saved: {pdf_outfile}")


# =========================================================
# 6. 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--outfile", type=str, default="results/vis/random_compare_3d.svg")
    parser.add_argument("--min-dist", type=float, default=1000.0)
    parser.add_argument("--max-dist", type=float, default=1600.0)
    parser.add_argument("--export-blender-json", action="store_true",
                        help="Export scene data to JSON for Blender import")
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

    # 使用不同的随机种子
    planner_kwargs = planner_kwargs_base.copy()
    planner_kwargs["random_seed"] = random.randint(0, 10_000_000)

    print("\nLooking for valid start/goal...")

    success = False
    for trial in range(40):
        start, goal = sample_start_goal(
            obstacles=obstacles,
            meta=meta,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
        )

        print(f"[Trial {trial + 1}] start={start}, goal={goal}")

        try:
            base_path, base_invalid_edges, base_result = run_baseline(
                planner_kwargs, start, goal
            )
            rl_path, rl_invalid_edges, rl_result = run_rl(
                args.model, planner_kwargs, start, goal
            )

            print("[Baseline]", base_result)
            print("[RL]", rl_result)
            print(f"[Debug] Baseline invalid edges: {len(base_invalid_edges)} (should match n_exp_invalid + n_inc_invalid + n_prog_invalid)")
            print(f"[Debug] RL invalid edges: {len(rl_invalid_edges)} (should match n_exp_invalid + n_inc_invalid + n_prog_invalid)")

            if len(base_invalid_edges) > 0:
                print(f"[Debug] First few baseline invalid edges: {base_invalid_edges[:3]}")
            if len(rl_invalid_edges) > 0:
                print(f"[Debug] First few RL invalid edges: {rl_invalid_edges[:3]}")

            # 绘制3D对比图
            plot_compare_3d(
                gdf=gdf,
                meta=meta,
                start=start,
                goal=goal,
                base_path=base_path,
                base_invalid_edges=base_invalid_edges,
                base_result=base_result,
                rl_path=rl_path,
                rl_invalid_edges=rl_invalid_edges,
                rl_result=rl_result,
                outfile=args.outfile,
            )

            # 可选：导出 Blender JSON
            if args.export_blender_json:
                export_scene_for_blender(
                    gdf=gdf,
                    meta=meta,
                    start=start,
                    goal=goal,
                    base_path=base_path,
                    base_invalid_edges=base_invalid_edges,
                    base_result=base_result,
                    rl_path=rl_path,
                    rl_invalid_edges=rl_invalid_edges,
                    rl_result=rl_result,
                    outfile=args.outfile,
                )

            success = True
            break

        except RuntimeError as e:
            print(f"[Trial {trial + 1}] failed: {e}")

    if not success:
        raise RuntimeError("Failed to find a successful run after 40 trials.")


if __name__ == "__main__":
    main()
