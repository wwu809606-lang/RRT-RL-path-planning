import json
import math
import random
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Point


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
# 基础工具
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def dist_3d(p1, p2):
    return float(math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    ))


def interpolate_segment_3d(p1, p2, resolution=5.0):
    length = dist_3d(p1, p2)
    n = max(2, int(length / resolution) + 1)
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    zs = np.linspace(p1[2], p2[2], n)
    return list(zip(xs, ys, zs))


def path_length_3d(path_xyz):
    if path_xyz is None or len(path_xyz) < 2:
        return 0.0
    return float(sum(dist_3d(path_xyz[i], path_xyz[i + 1]) for i in range(len(path_xyz) - 1)))


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
# 三维碰撞检测
# =========================================================
def point_in_collision_3d(p, obstacles):
    """
    p = (x, y, z)
    若点的平面投影落在某建筑底面内，且 z <= 建筑高度，则碰撞
    """
    pt2d = Point(p[0], p[1])
    z = p[2]

    for obs in obstacles:
        if z <= obs["height"] and obs["polygon"].intersects(pt2d):
            return True
    return False


def segment_collision_free_3d(p1, p2, obstacles, resolution=5.0):
    samples = interpolate_segment_3d(p1, p2, resolution=resolution)
    for p in samples:
        if point_in_collision_3d(p, obstacles):
            return False
    return True


# =========================================================
# RRT 结点
# =========================================================
class Node3D:
    def __init__(self, x, y, z, parent=None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.parent = parent

    @property
    def xyz(self):
        return (self.x, self.y, self.z)


# =========================================================
# 三维 RRT
# =========================================================
class RRTPlanner3D:
    def __init__(
        self,
        obstacles,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        step_size=45.0,
        goal_sample_rate=0.06,
        max_iter=12000,
        goal_tolerance=35.0,
        resolution=5.0,
        random_seed=42,
    ):
        self.obstacles = obstacles

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

        self.step_size = float(step_size)
        self.goal_sample_rate = float(goal_sample_rate)
        self.max_iter = int(max_iter)
        self.goal_tolerance = float(goal_tolerance)
        self.resolution = float(resolution)
        self.random_seed = int(random_seed)

        set_seed(self.random_seed)
        self.nodes = []

    def sample_free(self, goal_xyz):
        if random.random() < self.goal_sample_rate:
            return goal_xyz

        while True:
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            z = random.uniform(self.z_min, self.z_max)
            p = (x, y, z)
            if not point_in_collision_3d(p, self.obstacles):
                return p

    def nearest_index(self, xyz):
        dists = [dist_3d(node.xyz, xyz) for node in self.nodes]
        return int(np.argmin(dists))

    def steer(self, from_xyz, to_xyz):
        dx = to_xyz[0] - from_xyz[0]
        dy = to_xyz[1] - from_xyz[1]
        dz = to_xyz[2] - from_xyz[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)

        if d <= self.step_size:
            return (to_xyz[0], to_xyz[1], to_xyz[2])

        ux = dx / d
        uy = dy / d
        uz = dz / d

        return (
            from_xyz[0] + self.step_size * ux,
            from_xyz[1] + self.step_size * uy,
            from_xyz[2] + self.step_size * uz,
        )

    def in_search_bounds(self, p):
        return (
            self.x_min <= p[0] <= self.x_max and
            self.y_min <= p[1] <= self.y_max and
            self.z_min <= p[2] <= self.z_max
        )

    def plan(self, start_xyz, goal_xyz):
        if point_in_collision_3d(start_xyz, self.obstacles):
            raise ValueError(f"起点位于障碍物内: {start_xyz}")
        if point_in_collision_3d(goal_xyz, self.obstacles):
            raise ValueError(f"终点位于障碍物内: {goal_xyz}")

        self.nodes = [Node3D(*start_xyz, parent=None)]

        for k in range(self.max_iter):
            rnd = self.sample_free(goal_xyz)

            nearest_idx = self.nearest_index(rnd)
            nearest_node = self.nodes[nearest_idx]

            new_xyz = self.steer(nearest_node.xyz, rnd)

            if not self.in_search_bounds(new_xyz):
                continue

            if point_in_collision_3d(new_xyz, self.obstacles):
                continue

            if not segment_collision_free_3d(
                nearest_node.xyz, new_xyz, self.obstacles, resolution=self.resolution
            ):
                continue

            new_node = Node3D(*new_xyz, parent=nearest_idx)
            self.nodes.append(new_node)

            if dist_3d(new_node.xyz, goal_xyz) <= self.goal_tolerance:
                if segment_collision_free_3d(
                    new_node.xyz, goal_xyz, self.obstacles, resolution=self.resolution
                ):
                    goal_node = Node3D(*goal_xyz, parent=len(self.nodes) - 1)
                    self.nodes.append(goal_node)

                    path_xyz = self.extract_path(len(self.nodes) - 1)
                    return {
                        "success": True,
                        "iterations": k + 1,
                        "path_xyz": path_xyz,
                        "tree_edges": self.export_tree_edges(),
                    }

        return {
            "success": False,
            "iterations": self.max_iter,
            "path_xyz": None,
            "tree_edges": self.export_tree_edges(),
        }

    def extract_path(self, goal_idx):
        path = []
        idx = goal_idx
        while idx is not None:
            node = self.nodes[idx]
            path.append(node.xyz)
            idx = node.parent
        path.reverse()
        return path

    def export_tree_edges(self):
        edges = []
        for node in self.nodes:
            if node.parent is None:
                continue
            parent = self.nodes[node.parent]
            edges.append({
                "from": [parent.x, parent.y, parent.z],
                "to": [node.x, node.y, node.z],
            })
        return edges


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

    # 建筑物：与 3D 图一致的浅蓝灰
    gdf.plot(ax=ax, color="#cfcfcf", edgecolor="#7f7f7f", linewidth=0.5)

    # RRT 树：改成与 3D 图一致的蓝灰色
    for e in tree_edges:
        p1 = e["from"]
        p2 = e["to"]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color="#6b7280",
            linewidth=0.45,
            alpha=0.35
        )

    # 原始路径：改成更明显的蓝色
    if path_xyz is not None:
        arr = np.asarray(path_xyz)
        ax.plot(
            arr[:, 0], arr[:, 1],
            color="#4f6d8a",
            linewidth=2.2,
            label="Raw 3D RRT path"
        )

    # 起点终点也顺手统一一下颜色
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
    ax.set_title("3D RRT Baseline ")

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


def plot_3d_result(gdf, meta, start_xyz, goal_xyz, path_xyz, outfile):
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

    if path_xyz is not None:
        arr = np.asarray(path_xyz, dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], linewidth=2.6, label="Raw 3D RRT path")

    ax.scatter(start_xyz[0], start_xyz[1], start_xyz[2], s=60, marker="o", depthshade=False)
    ax.scatter(goal_xyz[0], goal_xyz[1], goal_xyz[2], s=70, marker="^", depthshade=False)

    ax.text(start_xyz[0], start_xyz[1], start_xyz[2] + 6, "Start", fontsize=10)
    ax.text(goal_xyz[0], goal_xyz[1], goal_xyz[2] + 6, "Goal", fontsize=10)

    x_range = meta["x_range"]
    y_range = meta["y_range"]
    z_max = max(float(CONFIG["z_max"]), meta["z_max_data"])

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
    )

    result = planner.plan(start_xyz=start_xyz, goal_xyz=goal_xyz)

    if not result["success"]:
        raise RuntimeError(
            f"3D RRT 在 {result['iterations']} 次迭代内未找到可行路径。"
            f"建议调大 max_iter、z_max，或微调 start/goal。"
        )

    path_xyz = result["path_xyz"]
    tree_edges = result["tree_edges"]

    if cfg["save_tree_json"]:
        save_json(
            {
                "random_seed": cfg["random_seed"],
                "iterations": result["iterations"],
                "tree_edges": tree_edges,
            },
            result_dir / "rrt_tree_3d.json",
        )

    if cfg["save_path_json"]:
        save_json(
            {
                "random_seed": cfg["random_seed"],
                "start_xyz": list(start_xyz),
                "goal_xyz": list(goal_xyz),
                "raw_path_xyz": [list(p) for p in path_xyz],
                "raw_path_length_m": path_length_3d(path_xyz),
                "iterations": result["iterations"],
            },
            result_dir / "rrt_path_3d.json",
        )

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
        )

    print("3D RRT baseline 运行完成")
    print(f"iterations = {result['iterations']}")
    print(f"raw_path_length = {path_length_3d(path_xyz):.2f} m")
    print(f"结果目录: {result_dir}")


if __name__ == "__main__":
    main()