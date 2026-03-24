import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

infile = "data/processed/buildings_with_height.geojson"
outfile = "results/pathplanning_3d_basemap.png"

gdf = gpd.read_file(infile)

if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

gdf = gdf.to_crs(epsg=3857)

xmin, ymin, xmax, ymax = gdf.total_bounds
gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

def add_extruded_polygon(ax, polygon, height,
                         facecolor="#d0d0d0",
                         edgecolor="#888888",
                         lw=0.12,
                         alpha=0.3):
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

for _, row in gdf.iterrows():
    geom = row.geometry
    h = float(row["height_m"])

    if geom.geom_type == "Polygon":
        add_extruded_polygon(ax, geom, h)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            add_extruded_polygon(ax, poly, h)

x_range = xmax - xmin
y_range = ymax - ymin
z_max = max(140, float(gdf["height_m"].max()))

ax.set_xlim(0, x_range)
ax.set_ylim(0, y_range)
ax.set_zlim(0, z_max)

ax.set_xlabel("X (m)", labelpad=10)
ax.set_ylabel("Y (m)", labelpad=10)
ax.set_zlabel("Z (m)", labelpad=8)

# 更适合路径展示的视角
ax.view_init(elev=34, azim=-64)

try:
    ax.set_box_aspect((x_range, y_range, z_max * 3.0))
except Exception:
    pass

ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"已保存 {outfile}")
