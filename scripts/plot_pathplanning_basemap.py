import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

infile = "data/processed/buildings_with_height.geojson"
outfile = "results/pathplanning_basemap.png"

gdf = gpd.read_file(infile)

if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

gdf = gdf.to_crs(epsg=3857)

xmin, ymin, xmax, ymax = gdf.total_bounds
gdf["geometry"] = gdf.translate(xoff=-xmin, yoff=-ymin)

fig, ax = plt.subplots(figsize=(7.2, 7.2))

gdf.plot(
    ax=ax,
    facecolor="#cfcfcf",
    edgecolor="#555555",
    linewidth=0.25
)

ax.set_xlim(0, xmax - xmin)
ax.set_ylim(0, ymax - ymin)
ax.set_aspect("equal")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

ax.ticklabel_format(style="plain", axis="both", useOffset=False)
ax.xaxis.set_major_locator(MultipleLocator(250))
ax.yaxis.set_major_locator(MultipleLocator(250))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"已保存 {outfile}")
