# -*- coding: utf-8 -*-

from pathlib import Path
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from tqdm import tqdm
import numpy as np
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

WATERWAYS_PATH = Path(
    r"D:\python\DEM_work\data\Co_work\projected_vectors\NY_NJ_waterways_EPSG5070.gpkg"
)

OUT_MASK = Path(
    r"D:\python\DEM_work\data\Features_resolution_30_m\DTRiver_mask.tif"
)

with rasterio.open(DEM_PATH) as dem:
    print("=" * 80)
    print("[DEM]")
    print("CRS:", dem.crs)
    print("Size:", dem.width, dem.height)
    print("Bounds:", dem.bounds)
    print("=" * 80)

    if dem.crs.to_epsg() != 5070:
        print("[警告] DEM CRS 不是 EPSG:5070，请先修复 DEM CRS。")

    gdf = gpd.read_file(WATERWAYS_PATH)

    print("[水系]")
    print("要素数:", len(gdf))
    print("CRS:", gdf.crs)
    print("Bounds:", gdf.total_bounds)

    if gdf.crs != dem.crs:
        gdf = gdf.to_crs(dem.crs)

    # 只保留和 DEM 范围相交的水系，减少栅格化负担
    from shapely.geometry import box
    dem_box = box(*dem.bounds)
    gdf = gdf[gdf.intersects(dem_box)].copy()

    print("与 DEM 相交的水系要素数:", len(gdf))

    if gdf.empty:
        raise RuntimeError("没有任何水系要素与 DEM 相交，不能栅格化。")

    profile = dem.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="DEFLATE",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]

    print("\n开始栅格化水系 mask...")

    mask = rasterize(
        shapes,
        out_shape=(dem.height, dem.width),
        transform=dem.transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=True,
    )

    # 用 DEM mask 去掉研究区外
    dem_arr = dem.read(1, masked=True)
    valid = ~dem_arr.mask
    mask = np.where(valid, mask, 0).astype("uint8")

OUT_MASK.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(OUT_MASK, "w", **profile) as dst:
    dst.write(mask, 1)

print("\n完成 mask:")
print(OUT_MASK)
print("河流像元数:", int(np.count_nonzero(mask == 1)))
print("河流像元比例:", float(np.count_nonzero(mask == 1) / mask.size))