# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
提取并合并 New York + New Jersey 的 OSM roads / waterways
"""

import os
import sys
from pathlib import Path

# =========================================================
# 强制指定 PROJ / GDAL 数据目录
# 必须放在 import geopandas / fiona / pyogrio 之前
# =========================================================

CONDA_ENV_DIR = Path(os.environ.get("CONDA_PREFIX", Path(sys.executable).resolve().parent))

PROJ_DIR = CONDA_ENV_DIR / "Library" / "share" / "proj"
GDAL_DIR = CONDA_ENV_DIR / "Library" / "share" / "gdal"

os.environ["PROJ_DATA"] = str(PROJ_DIR)
os.environ["PROJ_LIB"] = str(PROJ_DIR)
os.environ["GDAL_DATA"] = str(GDAL_DIR)

if not (PROJ_DIR / "proj.db").exists():
    raise FileNotFoundError(f"没有找到 proj.db：{PROJ_DIR / 'proj.db'}")

if not GDAL_DIR.exists():
    raise FileNotFoundError(f"没有找到 GDAL_DATA 目录：{GDAL_DIR}")

print("[环境检查]")
print(f"Python: {sys.executable}")
print(f"CONDA_ENV_DIR: {CONDA_ENV_DIR}")
print(f"PROJ_DATA: {os.environ['PROJ_DATA']}")
print(f"GDAL_DATA: {os.environ['GDAL_DATA']}")

# =========================================================
# 下面再导入 GIS 库
# =========================================================

import pandas as pd
import geopandas as gpd
import fiona


BASE_DIR = Path(r"D:\python\DEM_work\data\Co_work")

NY_GPKG = BASE_DIR / "osm" / "new-york-260503-free.gpkg" / "new-york.gpkg"
NJ_GPKG = BASE_DIR / "osm" / "new-jersey-latest-free.gpkg_2" / "new-jersey.gpkg"

OUT_DIR = BASE_DIR / "osm" / "ny_nj_extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ROADS = OUT_DIR / "NY_NJ_roads.gpkg"
OUT_WATERWAYS = OUT_DIR / "NY_NJ_waterways.gpkg"


def find_layer(gpkg_path, keywords):
    """
    根据关键词自动查找图层名。
    """
    layers = fiona.listlayers(gpkg_path)

    for layer in layers:
        lower = layer.lower()
        if any(k.lower() in lower for k in keywords):
            return layer

    raise ValueError(
        f"在 {gpkg_path} 中没有找到包含 {keywords} 的图层。\n"
        f"已有图层：{layers}"
    )


def read_layer(gpkg_path, layer):
    print(f"[读取] {gpkg_path}")
    print(f"[图层] {layer}")
    gdf = gpd.read_file(gpkg_path, layer=layer)

    if gdf.empty:
        raise ValueError(f"图层为空：{gpkg_path}, layer={layer}")

    if gdf.crs is None:
        raise ValueError(f"图层没有 CRS：{gpkg_path}, layer={layer}")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf


def merge_two_states(layer_keywords, out_path, out_layer):
    ny_layer = find_layer(NY_GPKG, layer_keywords)
    nj_layer = find_layer(NJ_GPKG, layer_keywords)

    ny = read_layer(NY_GPKG, ny_layer)
    nj = read_layer(NJ_GPKG, nj_layer)

    # 坐标系统一
    nj = nj.to_crs(ny.crs)

    merged = gpd.GeoDataFrame(
        pd.concat([ny, nj], ignore_index=True),
        crs=ny.crs
    )

    # 简单清理
    merged = merged[merged.geometry.notna()].copy()
    merged = merged[~merged.geometry.is_empty].copy()

    print(f"[合并] 要素数量：{len(merged)}")
    print(f"[输出] {out_path}")

    if out_path.exists():
        out_path.unlink()

    merged.to_file(out_path, layer=out_layer, driver="GPKG")

    return out_path


def main():
    print("=" * 80)
    print("提取并合并 NY + NJ 道路和水系")
    print("=" * 80)

    roads_path = merge_two_states(
        layer_keywords=["roads", "road"],
        out_path=OUT_ROADS,
        out_layer="roads"
    )

    waterways_path = merge_two_states(
        layer_keywords=["waterways", "waterway"],
        out_path=OUT_WATERWAYS,
        out_layer="waterways"
    )

    print("\n完成。")
    print(f"道路文件: {roads_path}")
    print(f"水系文件: {waterways_path}")


if __name__ == "__main__":
    main()