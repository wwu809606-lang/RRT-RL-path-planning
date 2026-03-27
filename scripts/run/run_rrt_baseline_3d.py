import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.planners.rrt_3d import RRTPlanner3D, set_seed
from src.utils.geometry import path_length_3d


# =========================================================
# 配置区
# =========================================================
CONFIG = {
    "infile": "data/processed/buildings_keep_15_recommended.geojson",
    "result_dir": "results/rrt_baseline_3d",
    "random_seed": 42,

    # 三维起终点
    "start": (60.0, 60.0, 40.0),
    "goal": (1000.0, 1300.0, 140.0),

    # 三维搜索空间 z 范围
    "z_min": 20.0,
    "z_max": 220.0,

    # RRT 参数
    "step_size": 45.0,
    "goal_sample_rate": 0.06,
    "max_iter": 12000,
    "goal_tolerance": 35.0,

    # 碰撞检测分辨率
    "collision_resolution": 5.0,

    # 建筑安全外扩（平面 buffer）
    "obstacle_buffer": 8.0,

    # 地图边界扩展
    "map_margin": 80.0,

    # 不做平滑，只保留原始路径
    "enable_path_smoothing": False,

    # 输出
    "save_tree_json": True,
    "save_path_json": True,
    "save_2d_plot": True,
    "save_3d_plot": True,
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
        raise ValueError("输入文件缺少 height_m 字段。")

    gdf["height_m"] = gdf["height_m"].astype(float)
    gdf = gdf[np.isfinite(gdf["height_m"])].copy()

    meta = {
        "xmin": 0.0,
        "ymin": 0.0,
        "xmax": xmax - xmin,
        "ymax": ymax - ymin,
        "x_range": xmax - xmin,
        "y_range": ymax - ymin,
        "z_max_data": float(gdf["height_m"].max()),
    }
    return gdf, meta


def prepare_obstacles_3d(gdf, obstacle_buffer):
    """
    将建筑准备成 3D 障碍体的轻量表示：
    - 底面 polygon（已做平面 buffer）
    - 建筑高度
    """
    obstacles = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row["height_m"])
        buffered = geom.buffer(obstacle_buffer)

        if buffered.geom_type == "Polygon":
            obstacles.append({
                "polygon": buffered,
                "height": h
            })
        elif buffered.geom_type == "MultiPolygon":
            for poly in buffered.geoms:
                obstacles.append({
                    "polygon": poly,
                    "height": h
                })

    return obstacles


# =========================================================
# 输出
# =========================================================
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# 2D 绘图（只画原始路径的 XY 投影）
# =========================================================
def plot_2d_result(gdf, meta, start_xyz, goal_xyz, tree_edges, path_xyz, outfile):
    fig, ax = plt.subplots(figsize=(9, 7))

    gdf.plot(ax=ax, color="#cfcfcf", edgecolor="#7f7f7f", linewidth=0.5)

    for e in tree_edges:
        p1 = e["from"]
        p2 = e["to"]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color="#6b7280",
            linewidth=0.45,
            alpha=0.35
        )

    if path_xyz is not None and len(path_xyz) > 0:
        arr = np.asarray(path_xyz)
        ax.plot(
            arr[:, 0], arr[:, 1],
            color="#4f6d8a",
            linewidth=2.2,
            label="Raw 3D RRT path"
        )

    ax.scatter(start_xyz[0], start_xyz[1], s=60, marker="o", color="#355c7d")
    ax.scatter(goal_xyz[0], goal_xyz[1], s=70, marker="^", color="#355c7d")

    ax.text(start_xyz[0] + 8, start_xyz[1] + 8, f"Start z={start_xyz[2]:.0f}", fontsize=9)
    ax.text(goal_xyz[0] + 8, goal_xyz[1] + 8, f"Goal z={goal_xyz[2]:.0f}", fontsize=9)

    ax.set_xlim(0, meta["x_range"])
    ax.set_ylim(0, meta["y_range"])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.set_title("3D RRT Baseline")

    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 3D 绘图（只画原始路径）
# =========================================================
def add_extruded_polygon(ax, polygon, height,
                         facecolor="#d3d3d3",
                         edgecolor="#9a9a9a",
                         lw=0.10,
                         alpha=0.22):
    if polygon.is_empty:
        return

    x, y = polygon.exterior.xy
    coords = list(zip(x, y))
    if len(coords) < 3:
        return

    top_face = [(px, py, height) for px, py in coords]
    ax.add_collection3d(
        Poly3DCollection(
            [top_face],
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=lw,
            alpha=alpha
        )
    )

    side_faces = []
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        side_faces.append([
            (x1, y1, 0),
            (x2, y2, 0),
            (x2, y2, height),
            (x1, y1, height)
        ])

    ax.add_collection3d(
        Poly3DCollection(
            side_faces,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=lw,
            alpha=alpha
        )
    )


def plot_3d_result(gdf, meta, start_xyz, goal_xyz, path_xyz, outfile, z_max_cfg):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row["height_m"])

        if geom.geom_type == "Polygon":
            add_extruded_polygon(ax, geom, h)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                add_extruded_polygon(ax, poly, h)

    if path_xyz is not None and len(path_xyz) > 0:
        arr = np.asarray(path_xyz, dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], linewidth=2.6, label="Raw 3D RRT path")

    ax.scatter(start_xyz[0], start_xyz[1], start_xyz[2], s=60, marker="o", depthshade=False)
    ax.scatter(goal_xyz[0], goal_xyz[1], goal_xyz[2], s=70, marker="^", depthshade=False)

    ax.text(start_xyz[0], start_xyz[1], start_xyz[2] + 6, "Start", fontsize=10)
    ax.text(goal_xyz[0], goal_xyz[1], goal_xyz[2] + 6, "Goal", fontsize=10)

    x_range = meta["x_range"]
    y_range = meta["y_range"]
    z_max = max(float(z_max_cfg), meta["z_max_data"])

    ax.set_xlim(0, x_range)
    ax.set_ylim(0, y_range)
    ax.set_zlim(0, z_max)

    ax.set_xlabel("X (m)", labelpad=10)
    ax.set_ylabel("Y (m)", labelpad=10)
    ax.set_zlabel("Z (m)", labelpad=8)
    ax.view_init(elev=34, azim=-64)

    try:
        ax.set_box_aspect((x_range, y_range, z_max * 2.2))
    except Exception:
        pass

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.legend(loc="upper right")

    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 主流程
# =========================================================
def main():
    cfg = CONFIG.copy()
    set_seed(cfg["random_seed"])

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    gdf, meta = load_buildings_local(cfg["infile"])
    obstacles = prepare_obstacles_3d(gdf, obstacle_buffer=cfg["obstacle_buffer"])

    x_min = -cfg["map_margin"]
    x_max = meta["x_range"] + cfg["map_margin"]
    y_min = -cfg["map_margin"]
    y_max = meta["y_range"] + cfg["map_margin"]
    z_min = cfg["z_min"]
    z_max = cfg["z_max"]

    start_xyz = tuple(cfg["start"])
    goal_xyz = tuple(cfg["goal"])

    planner = RRTPlanner3D(
        obstacles=obstacles,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        step_size=cfg["step_size"],
        goal_sample_rate=cfg["goal_sample_rate"],
        max_iter=cfg["max_iter"],
        goal_tolerance=cfg["goal_tolerance"],
        resolution=cfg["collision_resolution"],
        random_seed=cfg["random_seed"],
        duplicate_threshold=22.5,
        min_progress=6.75,
    )

    result = planner.plan(start_xyz=start_xyz, goal_xyz=goal_xyz)

    path_xyz = result["path_xyz"] or []
    tree_edges = result["tree_edges"]

    raw_path_length_m = None
    if result["success"] and len(path_xyz) > 0:
        raw_path_length_m = path_length_3d(path_xyz)

    eval_summary = {
        "random_seed": cfg["random_seed"],
        "config": {
            "start_xyz": list(start_xyz),
            "goal_xyz": list(goal_xyz),
            "z_min": cfg["z_min"],
            "z_max": cfg["z_max"],
            "step_size": cfg["step_size"],
            "goal_sample_rate": cfg["goal_sample_rate"],
            "max_iter": cfg["max_iter"],
            "goal_tolerance": cfg["goal_tolerance"],
            "collision_resolution": cfg["collision_resolution"],
            "obstacle_buffer": cfg["obstacle_buffer"],
            "map_margin": cfg["map_margin"],
            "duplicate_threshold": 22.5,
            "min_progress": 6.75,
        },
        "success": result["success"],
        "iterations": result["iterations"],
        "first_path_iter": result.get("first_path_iter", None),
        "path_xyz": [list(p) for p in path_xyz],
        "tree_edges": tree_edges,
        "n_exp_invalid": result["n_exp_invalid"],
        "n_inc_invalid": result["n_inc_invalid"],
        "n_prog_invalid": result["n_prog_invalid"],
        "invalid_ratio": result["invalid_ratio"],
        "d_best_trace": result["d_best_trace"],
        "raw_path_length_m": raw_path_length_m,
    }

    save_json(eval_summary, result_dir / "rrt_baseline_eval_3d.json")

    if cfg["save_tree_json"]:
        save_json(
            {
                "random_seed": cfg["random_seed"],
                "success": result["success"],
                "iterations": result["iterations"],
                "tree_edges": tree_edges,
                "n_exp_invalid": result["n_exp_invalid"],
                "n_inc_invalid": result["n_inc_invalid"],
                "n_prog_invalid": result["n_prog_invalid"],
                "invalid_ratio": result["invalid_ratio"],
            },
            result_dir / "rrt_tree_3d.json",
        )

    if cfg["save_path_json"]:
        save_json(
            {
                "random_seed": cfg["random_seed"],
                "success": result["success"],
                "start_xyz": list(start_xyz),
                "goal_xyz": list(goal_xyz),
                "raw_path_xyz": [list(p) for p in path_xyz],
                "raw_path_length_m": raw_path_length_m,
                "iterations": result["iterations"],
                "first_path_iter": result.get("first_path_iter", None),
                "n_exp_invalid": result["n_exp_invalid"],
                "n_inc_invalid": result["n_inc_invalid"],
                "n_prog_invalid": result["n_prog_invalid"],
                "invalid_ratio": result["invalid_ratio"],
                "d_best_trace": result["d_best_trace"],
            },
            result_dir / "rrt_path_3d.json",
        )

    if result["success"] and len(path_xyz) > 0:
        if cfg["save_2d_plot"]:
            plot_2d_result(
                gdf=gdf,
                meta=meta,
                start_xyz=start_xyz,
                goal_xyz=goal_xyz,
                tree_edges=tree_edges,
                path_xyz=path_xyz,
                outfile=result_dir / "rrt_baseline_3d_topview.png",
            )

        if cfg["save_3d_plot"]:
            plot_3d_result(
                gdf=gdf,
                meta=meta,
                start_xyz=start_xyz,
                goal_xyz=goal_xyz,
                path_xyz=path_xyz,
                outfile=result_dir / "rrt_baseline_3d_raw.png",
                z_max_cfg=cfg["z_max"],
            )

    print("3D RRT baseline 运行完成")
    print(f"success = {result['success']}")
    print(f"iterations = {result['iterations']}")
    print(f"first_path_iter = {result.get('first_path_iter', None)}")
    print(f"n_exp_invalid = {result['n_exp_invalid']}")
    print(f"n_inc_invalid = {result['n_inc_invalid']}")
    print(f"n_prog_invalid = {result['n_prog_invalid']}")
    print(f"invalid_ratio = {result['invalid_ratio']:.4f}")

    if raw_path_length_m is not None:
        print(f"raw_path_length = {raw_path_length_m:.2f} m")
    else:
        print("raw_path_length = None")

    print(f"d_best_trace_len = {len(result['d_best_trace'])}")
    print(f"结果目录: {result_dir}")

    if not result["success"]:
        print("警告：本次 baseline 未找到可行路径，但评估 JSON 已保存。")


if __name__ == "__main__":
    main()