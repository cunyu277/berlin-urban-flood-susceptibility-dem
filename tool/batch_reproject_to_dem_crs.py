# -*- coding: utf-8 -*-
"""
批量统一坐标系脚本

功能：
1. 将多个栅格批量重投影 / 重采样 / 对齐到 DEM 网格；
2. 将多个矢量批量重投影到 DEM CRS；
3. 输出结果可直接用于洪水易发性制图因子栅格叠加。

推荐用途：
- PRISM AP 栅格 → 对齐到 DEM 网格
- GCN250 CN 栅格 → 对齐到 DEM 网格
- OSM roads / waterways → 转到 DEM CRS
"""

import os
import sys
from pathlib import Path

# =========================================================
# 0. 可选：兜底设置 PROJ / GDAL
# 如果你已经 conda env config vars 配好了，这段不会干扰。
# =========================================================

CONDA_ENV_DIR = Path(os.environ.get("CONDA_PREFIX", Path(sys.executable).resolve().parent))
PROJ_DIR = CONDA_ENV_DIR / "Library" / "share" / "proj"
GDAL_DIR = CONDA_ENV_DIR / "Library" / "share" / "gdal"

if PROJ_DIR.exists():
    os.environ.setdefault("PROJ_DATA", str(PROJ_DIR))
    os.environ.setdefault("PROJ_LIB", str(PROJ_DIR))

if GDAL_DIR.exists():
    os.environ.setdefault("GDAL_DATA", str(GDAL_DIR))


import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject
from tqdm import tqdm


# =========================================================
# 1. 路径配置：主要改这里
# =========================================================

BASE_DIR = Path(r"D:\python\DEM_work\data")

# 目标 DEM：所有栅格最终都会对齐到这个 DEM 的 CRS、分辨率、范围、行列数
DEM_PATH = BASE_DIR / "Features_resolution_30_m" / "DEM.tif"

# 你刚才按 shp 裁剪后的 PRISM / GCN250 栅格目录
RASTER_IN_DIR = BASE_DIR / "Co_work" / "clip"

# 栅格输出目录
RASTER_OUT_DIR = BASE_DIR / "Co_work" / "projected_to_dem_grid"

# 矢量输入文件：可以根据你的实际情况修改
VECTOR_FILES = [
    BASE_DIR / "Co_work" / "osm" / "ny_nj_extracted" / "NY_NJ_roads.gpkg",
    BASE_DIR / "Co_work" / "osm" / "ny_nj_extracted" / "NY_NJ_waterways.gpkg",
]

# 矢量输出目录
VECTOR_OUT_DIR = BASE_DIR / "Co_work" / "projected_vectors"

# 输出 NoData
NODATA = -9999.0

# 分块大小：越大越快，但越占内存
BLOCK_SIZE = 1024

# 是否建立金字塔
BUILD_OVERVIEWS = True

# GDAL 参数
NUM_THREADS = max(1, os.cpu_count() or 1)
GDAL_CACHEMAX_MB = 2048


# =========================================================
# 2. 工具函数
# =========================================================

def iter_windows(width, height, block_size):
    for row_off in range(0, height, block_size):
        win_h = min(block_size, height - row_off)

        for col_off in range(0, width, block_size):
            win_w = min(block_size, width - col_off)
            yield Window(col_off, row_off, win_w, win_h)


def count_windows(width, height, block_size):
    return int(np.ceil(width / block_size) * np.ceil(height / block_size))


def build_overviews(raster_path):
    if not BUILD_OVERVIEWS:
        return

    print(f"[金字塔] {raster_path}")

    with rasterio.open(raster_path, "r+") as dst:
        factors = [2, 4, 8, 16, 32, 64]
        dst.build_overviews(factors, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")


def get_resampling_method(raster_path):
    """
    根据文件名自动判断重采样方法。

    连续变量：
    - AP / PRISM / ppt / precipitation 用 bilinear

    离散或半离散变量：
    - CN / GCN 用 nearest
    """
    name = raster_path.name.lower()

    if any(k in name for k in ["cn", "gcn", "curve"]):
        return Resampling.nearest

    if any(k in name for k in ["ap", "prism", "ppt", "precip", "rain"]):
        return Resampling.bilinear

    # 默认用 bilinear，适合连续变量
    return Resampling.bilinear


def make_output_profile_like_dem(dem, dtype="float32", nodata=NODATA):
    profile = dem.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    return profile


def validate_inputs():
    print("\n" + "=" * 80)
    print("[输入检查]")
    print("=" * 80)

    if not DEM_PATH.exists():
        raise FileNotFoundError(f"DEM 不存在：{DEM_PATH}")

    if not RASTER_IN_DIR.exists():
        raise FileNotFoundError(f"栅格输入目录不存在：{RASTER_IN_DIR}")

    raster_files = sorted(RASTER_IN_DIR.glob("*.tif"))

    if not raster_files:
        raise FileNotFoundError(f"栅格输入目录中没有 tif 文件：{RASTER_IN_DIR}")

    print(f"DEM: {DEM_PATH}")
    print(f"栅格输入目录: {RASTER_IN_DIR}")
    print(f"发现栅格数量: {len(raster_files)}")

    for p in raster_files:
        print(f"  - {p.name}")

    print("\n矢量文件：")
    for p in VECTOR_FILES:
        print(f"  - {p} | exists={p.exists()}")

    return raster_files


# =========================================================
# 3. 栅格：批量对齐到 DEM 网格
# =========================================================

def reproject_raster_to_dem_grid(src_raster, dem_path, out_raster, resampling):
    """
    将单个栅格重投影、重采样、裁剪到 DEM 网格。

    输出结果：
    - CRS = DEM CRS
    - transform = DEM transform
    - width/height = DEM width/height
    - resolution = DEM resolution
    - extent = DEM extent
    """
    src_raster = Path(src_raster)
    out_raster = Path(out_raster)
    out_raster.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise ValueError(f"DEM 没有 CRS：{dem_path}")

        if dem.crs.is_geographic:
            raise ValueError(
                "DEM 是经纬度坐标系。建议先把 DEM 投影到米制坐标系，例如 EPSG:5070。"
            )

        dem_nodata = dem.nodata if dem.nodata is not None else NODATA

        profile = make_output_profile_like_dem(
            dem,
            dtype="float32",
            nodata=NODATA,
        )

        total = count_windows(dem.width, dem.height, BLOCK_SIZE)

        with rasterio.open(src_raster) as src:
            if src.crs is None:
                raise ValueError(f"输入栅格没有 CRS：{src_raster}")

            print("\n" + "=" * 80)
            print(f"[栅格投影/对齐] {src_raster.name}")
            print(f"输入 CRS: {src.crs}")
            print(f"目标 CRS: {dem.crs}")
            print(f"目标尺寸: {dem.width} x {dem.height}")
            print(f"目标分辨率: {dem.res}")
            print(f"重采样方法: {resampling.name}")
            print(f"输出: {out_raster}")
            print("=" * 80)

            with rasterio.open(out_raster, "w", **profile) as dst:
                for win in tqdm(
                    iter_windows(dem.width, dem.height, BLOCK_SIZE),
                    total=total,
                    desc=f"Align {src_raster.stem}",
                    unit="block",
                ):
                    dest = np.full(
                        (int(win.height), int(win.width)),
                        NODATA,
                        dtype="float32",
                    )

                    win_transform = dem.window_transform(win)

                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src.nodata,
                        dst_transform=win_transform,
                        dst_crs=dem.crs,
                        dst_nodata=NODATA,
                        resampling=resampling,
                        num_threads=NUM_THREADS,
                        init_dest_nodata=True,
                    )

                    # 用 DEM mask 去掉研究区外区域
                    dem_block = dem.read(1, window=win, masked=True)
                    dem_valid = ~dem_block.mask

                    valid = dem_valid & np.isfinite(dest) & (dest != NODATA)
                    dest = np.where(valid, dest, NODATA).astype("float32")

                    dst.write(dest, 1, window=win)

    build_overviews(out_raster)

    return out_raster


def batch_reproject_rasters_to_dem_grid(raster_files):
    RASTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []

    print("\n" + "#" * 80)
    print("开始批量统一栅格到 DEM 网格")
    print("#" * 80)

    for src_raster in raster_files:
        resampling = get_resampling_method(src_raster)

        out_name = src_raster.stem + "_to_DEM_grid.tif"
        out_raster = RASTER_OUT_DIR / out_name

        result = reproject_raster_to_dem_grid(
            src_raster=src_raster,
            dem_path=DEM_PATH,
            out_raster=out_raster,
            resampling=resampling,
        )

        outputs.append(result)

    return outputs


# =========================================================
# 4. 矢量：批量转换到 DEM CRS
# =========================================================

def clean_vector(gdf):
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        # 线数据一般不需要 buffer(0)，这里只做兜底
        pass

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf


def reproject_vector_to_dem_crs(vector_path, dem_crs, out_dir):
    vector_path = Path(vector_path)

    if not vector_path.exists():
        print(f"[跳过] 矢量不存在：{vector_path}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{vector_path.stem}_to_DEM_CRS.gpkg"

    print("\n" + "=" * 80)
    print(f"[矢量投影] {vector_path}")
    print(f"输出: {out_path}")
    print("=" * 80)

    gdf = gpd.read_file(vector_path)

    if gdf.empty:
        print(f"[跳过] 矢量为空：{vector_path}")
        return None

    if gdf.crs is None:
        raise ValueError(f"矢量没有 CRS，请先定义坐标系：{vector_path}")

    print(f"输入 CRS: {gdf.crs}")
    print(f"目标 CRS: {dem_crs}")
    print(f"输入要素数: {len(gdf)}")

    gdf = clean_vector(gdf)
    gdf = gdf.to_crs(dem_crs)

    print(f"输出要素数: {len(gdf)}")

    if out_path.exists():
        out_path.unlink()

    gdf.to_file(out_path, layer=vector_path.stem, driver="GPKG")

    return out_path


def batch_reproject_vectors_to_dem_crs():
    outputs = []

    with rasterio.open(DEM_PATH) as dem:
        dem_crs = dem.crs

    print("\n" + "#" * 80)
    print("开始批量统一矢量到 DEM CRS")
    print("#" * 80)

    for vector_path in VECTOR_FILES:
        result = reproject_vector_to_dem_crs(
            vector_path=vector_path,
            dem_crs=dem_crs,
            out_dir=VECTOR_OUT_DIR,
        )

        if result is not None:
            outputs.append(result)

    return outputs


# =========================================================
# 5. 主函数
# =========================================================

def main():
    raster_files = validate_inputs()

    env_options = {
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "NUM_THREADS": "ALL_CPUS",
        "GDAL_CACHEMAX": GDAL_CACHEMAX_MB,
        "CHECK_WITH_INVERT_PROJ": "YES",
    }

    with rasterio.Env(**env_options):
        raster_outputs = batch_reproject_rasters_to_dem_grid(raster_files)
        vector_outputs = batch_reproject_vectors_to_dem_crs()

    print("\n" + "#" * 80)
    print("全部统一坐标系完成")
    print("#" * 80)

    print("\n栅格输出：")
    for p in raster_outputs:
        print(f"  - {p}")

    print("\n矢量输出：")
    for p in vector_outputs:
        print(f"  - {p}")

    print("\n建议在因子提取 Notebook / py 脚本中使用：")

    for p in raster_outputs:
        lower = p.name.lower()

        if "prism" in lower or "ap" in lower or "ppt" in lower:
            print(f'AP_RASTER_PATH = Path(r"{p}")')

        if "arcii" in lower:
            print(f'CN_RASTER_PATH = Path(r"{p}")')

    for p in vector_outputs:
        lower = p.name.lower()

        if "road" in lower:
            print(f'ROAD_VECTOR_PATH = Path(r"{p}")')

        if "water" in lower or "river" in lower:
            print(f'RIVER_VECTOR_PATH = Path(r"{p}")')


if __name__ == "__main__":
    main()