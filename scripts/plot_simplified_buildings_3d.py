import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path

infile = "data/processed/buildings_keep_15_recommended.geojson"
outfile = "results/buildings_keep_15_3d_extrusion.png"

# -------------------------
# 读取数据
# -------------------------
gdf = gpd.read_file(infile)

if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

# 转到米制投影
gdf = gdf.to_crs(epsg=3857)

# 平移到局部坐标系，避免坐标太大
xmin, ymin, xmax, ymax = gdf.total_bounds
gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)

# 只保留有效几何
gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

# 去掉没有高度的记录
gdf["height_m"] = gdf["height_m"].astype(float)
gdf = gdf[np.isfinite(gdf["height_m"])].copy()

if len(gdf) == 0:
    raise ValueError("没有可绘制的建筑数据，请检查输入文件。")

# -------------------------
# 颜色映射：按高度着色
# -------------------------
heights = gdf["height_m"].values
vmin = float(np.nanmin(heights))
vmax = float(np.nanmax(heights))

cmap = plt.cm.YlGnBu
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# -------------------------
# 创建三维画布
# -------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# -------------------------
# 构造建筑立面与顶面
# -------------------------
def add_extruded_polygon(ax, polygon, height, facecolor, edgecolor="#444444", lw=0.15):
    """
    将 shapely Polygon 挤出为三维建筑
    """
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
            linewidths=lw
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
            (x2, y2, height),
            (x1, y1, height)
        ])

    ax.add_collection3d(
        Poly3DCollection(
            side_faces,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=lw
        )
    )

# -------------------------
# 绘制所有建筑
# -------------------------
for _, row in gdf.iterrows():
    geom = row.geometry
    h = float(row["height_m"])
    color = cmap(norm(h))

    if geom.geom_type == "Polygon":
        add_extruded_polygon(ax, geom, h, color)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            add_extruded_polygon(ax, poly, h, color)

# -------------------------
# 坐标与视角设置
# -------------------------
x_range = xmax - xmin
y_range = ymax - ymin
z_range = max(120, vmax)

ax.set_xlim(0, x_range)
ax.set_ylim(0, y_range)
ax.set_zlim(0, z_range)

ax.set_xlabel("X (m)", labelpad=10)
ax.set_ylabel("Y (m)", labelpad=10)
ax.set_zlabel("Z (m)", labelpad=8)

# 视角：斜俯视
ax.view_init(elev=32, azim=-58)

# 控制三轴比例，避免建筑被压扁
try:
    ax.set_box_aspect((x_range, y_range, z_range * 3.2))
except Exception:
    pass

# 面板尽量简洁
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# -------------------------
# 色带
# -------------------------
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.08)
cbar.set_label("Height (m)")

# -------------------------
# 输出
# -------------------------
Path(outfile).parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"已保存 {outfile}")