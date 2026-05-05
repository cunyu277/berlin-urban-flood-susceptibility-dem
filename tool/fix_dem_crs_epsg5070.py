# -*- coding: utf-8 -*-

from pathlib import Path
import rasterio
from rasterio.crs import CRS
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

DEM_PATH = Path(r"D:\python\DEM_work\data\Features_resolution_30_m\DEM.tif")

with rasterio.open(DEM_PATH, "r+") as ds:
    print("修复前 CRS:")
    print(ds.crs)

    ds.crs = CRS.from_epsg(5070)

    print("\n修复后 CRS:")
    print(ds.crs)

print("\n完成：已将 DEM CRS 重新定义为 EPSG:5070")