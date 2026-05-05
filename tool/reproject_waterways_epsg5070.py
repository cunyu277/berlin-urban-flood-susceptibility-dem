# -*- coding: utf-8 -*-

from pathlib import Path
import geopandas as gpd
import os
import sys
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
IN_WATERWAYS = Path(
    r"D:\python\DEM_work\data\Co_work\osm\ny_nj_extracted\NY_NJ_waterways.gpkg"
)

OUT_WATERWAYS = Path(
    r"D:\python\DEM_work\data\Co_work\projected_vectors\NY_NJ_waterways_EPSG5070.gpkg"
)

gdf = gpd.read_file(IN_WATERWAYS)

print("输入要素数:", len(gdf))
print("输入 CRS:", gdf.crs)
print("输入 Bounds:", gdf.total_bounds)

if gdf.crs is None:
    raise ValueError("输入水系没有 CRS。")

gdf = gdf[gdf.geometry.notna()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

# 只保留线
gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

# 直接转 EPSG:5070
gdf = gdf.to_crs("EPSG:5070")

print("\n输出要素数:", len(gdf))
print("输出 CRS:", gdf.crs)
print("输出 Bounds:", gdf.total_bounds)

OUT_WATERWAYS.parent.mkdir(parents=True, exist_ok=True)

if OUT_WATERWAYS.exists():
    OUT_WATERWAYS.unlink()

gdf.to_file(OUT_WATERWAYS, layer="waterways", driver="GPKG")

print("\n完成:")
print(OUT_WATERWAYS)