import geopandas as gpd
import numpy as np
import json
from shapely.geometry import Point

infile = "data/processed/buildings_with_height.geojson"
outfile_grid = "data/processed/occupancy_grid.npy"
outfile_meta = "data/processed/grid_meta.json"

# -------------------------
# 参数
# -------------------------
RES_XY = 5.0   # 水平分辨率，单位 m
RES_Z = 5.0    # 垂直分辨率，单位 m
Z_MIN = 0.0
Z_MAX = 120.0  # 可按需要改成 150.0

# -------------------------
# 读取建筑数据
# -------------------------
gdf = gpd.read_file(infile)

if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

# 转成米制投影，便于栅格化
gdf = gdf.to_crs(epsg=3857)

xmin, ymin, xmax, ymax = gdf.total_bounds

# 网格坐标
xs = np.arange(xmin, xmax + RES_XY, RES_XY)
ys = np.arange(ymin, ymax + RES_XY, RES_XY)
zs = np.arange(Z_MIN, Z_MAX + RES_Z, RES_Z)

nx, ny, nz = len(xs), len(ys), len(zs)
grid = np.zeros((nx, ny, nz), dtype=np.uint8)

print(f"网格尺寸: nx={nx}, ny={ny}, nz={nz}")
print(f"总单元数: {nx * ny * nz}")

# -------------------------
# 建筑填充为障碍体素
# -------------------------
for idx, row in gdf.iterrows():
    geom = row.geometry
    h = float(row["height_m"])

    bxmin, bymin, bxmax, bymax = geom.bounds

    ix = np.where((xs >= bxmin) & (xs <= bxmax))[0]
    iy = np.where((ys >= bymin) & (ys <= bymax))[0]
    iz = np.where(zs <= h)[0]

    if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
        continue

    for i in ix:
        for j in iy:
            p = Point(xs[i], ys[j])
            if geom.contains(p) or geom.touches(p):
                grid[i, j, iz] = 1

    if (idx + 1) % 20 == 0:
        print(f"已处理 {idx + 1}/{len(gdf)} 栋建筑")

# -------------------------
# 保存结果
# -------------------------
np.save(outfile_grid, grid)

meta = {
    "crs": "EPSG:3857",
    "resolution_xy_m": RES_XY,
    "resolution_z_m": RES_Z,
    "z_min_m": Z_MIN,
    "z_max_m": Z_MAX,
    "xmin": float(xmin),
    "ymin": float(ymin),
    "xmax": float(xmax),
    "ymax": float(ymax),
    "nx": int(nx),
    "ny": int(ny),
    "nz": int(nz)
}

with open(outfile_meta, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n已保存: {outfile_grid}")
print(f"已保存: {outfile_meta}")
print(f"障碍体素数: {int(grid.sum())}")
print(f"障碍占比: {grid.mean():.6f}")
