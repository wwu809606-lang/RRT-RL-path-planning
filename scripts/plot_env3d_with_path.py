import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path


# =========================================================
# 基础配置
# =========================================================
DEFAULT_BUILDING_FILE = "data/processed/buildings_keep_15_recommended.geojson"
DEFAULT_OUTFILE = "results/env3d_with_path.png"


# =========================================================
# 数据读取与预处理
# =========================================================
def load_buildings(infile):
    gdf = gpd.read_file(infile)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    # 转到米制投影
    gdf = gdf.to_crs(epsg=3857)

    # 原始边界
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # 平移到局部坐标系
    gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)

    # 仅保留有效几何
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # 高度字段
    gdf["height_m"] = gdf["height_m"].astype(float)
    gdf = gdf[np.isfinite(gdf["height_m"])].copy()

    if len(gdf) == 0:
        raise ValueError("没有可绘制的建筑数据，请检查输入文件。")

    x_range = xmax - xmin
    y_range = ymax - ymin
    z_max = float(gdf["height_m"].max())

    meta = {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "x_range": x_range,
        "y_range": y_range,
        "z_max": z_max,
    }
    return gdf, meta


# =========================================================
# 建筑绘制
# =========================================================
def add_extruded_polygon(
    ax,
    polygon,
    height,
    facecolor="#d3d3d3",
    edgecolor="#9a9a9a",
    lw=0.10,
    alpha=0.22
):
    if polygon.is_empty:
        return

    x, y = polygon.exterior.xy
    coords = list(zip(x, y))
    if len(coords) < 3:
        return

    # 顶面
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

    # 侧面
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


def plot_buildings_3d(ax, gdf):
    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row["height_m"])

        if geom.geom_type == "Polygon":
            add_extruded_polygon(ax, geom, h)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                add_extruded_polygon(ax, poly, h)


# =========================================================
# 起点终点与路径绘制
# =========================================================
def plot_start_goal(ax, start=None, goal=None, zorder_offset=5):
    """
    start, goal: (x, y, z)
    """
    if start is not None:
        ax.scatter(
            start[0], start[1], start[2],
            s=60,
            marker="o",
            depthshade=False
        )
        ax.text(
            start[0], start[1], start[2] + 6,
            "Start",
            fontsize=10
        )

    if goal is not None:
        ax.scatter(
            goal[0], goal[1], goal[2],
            s=70,
            marker="^",
            depthshade=False
        )
        ax.text(
            goal[0], goal[1], goal[2] + 6,
            "Goal",
            fontsize=10
        )


def plot_single_path(ax, path_xyz, linewidth=2.2, label=None):
    """
    path_xyz: ndarray, shape (N, 3)
    """
    path_xyz = np.asarray(path_xyz, dtype=float)
    if path_xyz.ndim != 2 or path_xyz.shape[1] != 3:
        raise ValueError("path_xyz 必须是形状为 (N, 3) 的数组。")

    ax.plot(
        path_xyz[:, 0],
        path_xyz[:, 1],
        path_xyz[:, 2],
        linewidth=linewidth,
        label=label
    )


def plot_multiple_paths(ax, paths):
    """
    paths: list of dict
    每个元素示例：
    {
        "xyz": np.array([[x,y,z], ...]),
        "label": "RRT",
        "linewidth": 2.2
    }
    """
    if paths is None:
        return

    for item in paths:
        xyz = item["xyz"]
        label = item.get("label", None)
        linewidth = item.get("linewidth", 2.2)
        plot_single_path(ax, xyz, linewidth=linewidth, label=label)


# =========================================================
# 坐标轴与画布样式
# =========================================================
def setup_axes(ax, meta, z_scale=2.2):
    x_range = meta["x_range"]
    y_range = meta["y_range"]
    z_max = max(140.0, meta["z_max"])

    ax.set_xlim(0, x_range)
    ax.set_ylim(0, y_range)
    ax.set_zlim(0, z_max)

    ax.set_xlabel("X (m)", labelpad=10)
    ax.set_ylabel("Y (m)", labelpad=10)
    ax.set_zlabel("Z (m)", labelpad=8)

    # 更适合路径展示
    ax.view_init(elev=34, azim=-64)

    try:
        ax.set_box_aspect((x_range, y_range, z_max * z_scale))
    except Exception:
        pass

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    try:
        ax.xaxis.pane.set_edgecolor("white")
        ax.yaxis.pane.set_edgecolor("white")
        ax.zaxis.pane.set_edgecolor("white")
    except Exception:
        pass


# =========================================================
# 主函数
# =========================================================
def plot_environment_with_paths(
    infile=DEFAULT_BUILDING_FILE,
    outfile=DEFAULT_OUTFILE,
    start=None,
    goal=None,
    paths=None,
    figsize=(10, 8)
):
    gdf, meta = load_buildings(infile)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # 先画建筑
    plot_buildings_3d(ax, gdf)

    # 再画起终点和路径
    plot_start_goal(ax, start=start, goal=goal)
    plot_multiple_paths(ax, paths=paths)

    setup_axes(ax, meta, z_scale=2.2)

    # 如果路径有标签，就显示图例
    has_label = paths is not None and any(item.get("label") for item in paths)
    if has_label:
        ax.legend(loc="upper right")

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存 {outfile}")


# =========================================================
# 示例
# =========================================================
if __name__ == "__main__":
    # 下面这组只是演示用，后面替换成你自己的真实起终点和路径
    start = (80, 80, 20)
    goal = (900, 650, 80)

    # 示例路径1：baseline
    path_rrt = np.array([
        [80, 80, 20],
        [140, 120, 35],
        [240, 180, 50],
        [360, 260, 60],
        [520, 380, 60],
        [700, 520, 70],
        [900, 650, 80]
    ])

    # 示例路径2：优化后
    path_opt = np.array([
        [80, 80, 20],
        [170, 130, 30],
        [300, 220, 42],
        [470, 340, 52],
        [650, 470, 63],
        [900, 650, 80]
    ])

    paths = [
        {"xyz": path_rrt, "label": "RRT Baseline", "linewidth": 2.0},
        {"xyz": path_opt, "label": "RRT + RL", "linewidth": 2.6},
    ]

    plot_environment_with_paths(
        infile=DEFAULT_BUILDING_FILE,
        outfile="results/env3d_demo_with_paths.png",
        start=start,
        goal=goal,
        paths=paths
    )