import numpy as np
import json
import matplotlib.pyplot as plt

grid = np.load("data/processed/occupancy_grid.npy")

with open("data/processed/grid_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# 1. 二维占据图：只要某个(x,y)任意高度有障碍，就标1
occ2d = (grid.sum(axis=2) > 0).astype(int)

# 2. 高度投影图：每个(x,y)最高占据层
height_idx = np.full(occ2d.shape, -1, dtype=int)
for k in range(grid.shape[2] - 1, -1, -1):
    mask = (grid[:, :, k] == 1) & (height_idx == -1)
    height_idx[mask] = k

height_m = np.where(
    height_idx >= 0,
    height_idx * meta["resolution_z_m"],
    np.nan
)

# -------------------------
# 图1：二维占据俯视图
# -------------------------
plt.figure(figsize=(8, 8))
plt.imshow(occ2d.T, origin="lower", cmap="gray")
plt.title("2D Occupancy Map (Top View)")
plt.xlabel("Grid X")
plt.ylabel("Grid Y")
plt.tight_layout()
plt.savefig("results/occupancy_topview.png", dpi=200)
plt.close()

# -------------------------
# 图2：建筑高度投影图
# -------------------------
plt.figure(figsize=(8, 8))
plt.imshow(height_m.T, origin="lower")
plt.colorbar(label="Height (m)")
plt.title("Building Height Projection")
plt.xlabel("Grid X")
plt.ylabel("Grid Y")
plt.tight_layout()
plt.savefig("results/occupancy_height_projection.png", dpi=200)
plt.close()

print("已保存 results/occupancy_topview.png")
print("已保存 results/occupancy_height_projection.png")
