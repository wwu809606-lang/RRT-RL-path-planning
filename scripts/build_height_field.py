import geopandas as gpd
import pandas as pd
import numpy as np
import re

infile = "data/raw/qjc_buildings.geojson"
outfile = "data/processed/buildings_with_height.geojson"

gdf = gpd.read_file(infile)

def parse_num(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower().replace("m", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else np.nan

def default_height(building_type):
    if pd.isna(building_type):
        return 18.0
    bt = str(building_type).lower()
    if bt in ["residential", "apartments", "house"]:
        return 24.0
    if bt in ["office", "commercial", "hotel"]:
        return 36.0
    if bt in ["retail", "supermarket", "mall"]:
        return 15.0
    if bt in ["school", "kindergarten"]:
        return 18.0
    return 18.0

height_vals = []
src_vals = []

for _, row in gdf.iterrows():
    h = parse_num(row["height"]) if "height" in gdf.columns else np.nan
    if not np.isnan(h):
        height_vals.append(h)
        src_vals.append("height")
        continue

    lv = parse_num(row["building:levels"]) if "building:levels" in gdf.columns else np.nan
    if not np.isnan(lv):
        height_vals.append(lv * 3.2)
        src_vals.append("building:levels*3.2")
        continue

    bt = row["building"] if "building" in gdf.columns else None
    height_vals.append(default_height(bt))
    src_vals.append("default")

gdf["height_m"] = height_vals
gdf["height_src"] = src_vals

gdf.to_file(outfile, driver="GeoJSON")

print("输出文件:", outfile)
print(gdf["height_src"].value_counts())
print("\n高度统计：")
print(gdf["height_m"].describe())