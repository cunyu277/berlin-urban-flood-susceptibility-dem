# -*- coding: utf-8 -*-
"""
NY-NJ DEM 裁剪、统一投影、拼接脚本

适用场景：
1. 你有两个 DEM，例如 New York DEM 和 New Jersey DEM；
2. 两个 DEM 有重叠区域；
3. 你有一个矢量边界文件，这个矢量已经包含“纽约州 + 新泽西州”两个州；
4. 希望先用行政边界裁剪 DEM，再统一投影坐标系，最后拼接成一个 DEM。

主要依赖：
rasterio, geopandas, shapely, pyproj, fiona, tqdm
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    aligned_target,
)

from shapely.geometry import mapping
from shapely.ops import unary_union
from tqdm import tqdm
from pathlib import Path


# =========================================================
# 强制指定当前 conda 环境中的 PROJ / GDAL 数据目录
# 必须放在 import rasterio / geopandas / pyproj 之前
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


# 当前脚本路径：D:\python\DEM_work\tool\clip_project_mosaic_dem.py
SCRIPT_DIR = Path(__file__).resolve().parent

# 项目根目录：D:\python\DEM_work
PROJECT_DIR = SCRIPT_DIR.parent

# 数据目录：D:\python\DEM_work\data
DATA_DIR = PROJECT_DIR / "data"

DEM_FILES = [
    DATA_DIR / "Features_resolution_30_m" / "DEM_xzx.tif",
    DATA_DIR / "Features_resolution_30_m" / "DEM_ny.tif",
]

BOUNDARY_FILE = DATA_DIR / "shp" / "nyxzx.shp"

OUT_DIR = Path(r"D:\data\output\ny_nj_dem")

BOUNDARY_QUERY = None

# 推荐用于美国本土大范围分析的投影坐标系，单位为米
# DEM 水文分析、坡度、TWI、距离等因子计算时，比经纬度坐标更合理
TARGET_CRS = "EPSG:5070"

# 输出分辨率，单位：米
# 30m DEM 写 30；10m DEM 写 10
TARGET_RESOLUTION = 30

# 输出 NoData
NODATA = -9999.0

# 输出数据类型
OUT_DTYPE = "float32"

# 裁剪时是否包含边界触碰像元
# True 会略微扩大裁剪范围；False 更严格
ALL_TOUCHED = False

# 分块大小
# 越大速度越快，但越占内存；一般 1024 或 2048 都可以
CLIP_BLOCK_SIZE = 2048

# 两个 DEM 是否并行处理
# 内存小就改成 1；内存够可以用 2
MAX_PARALLEL_DEMS = 2

# GDAL 多线程设置
NUM_THREADS = max(1, os.cpu_count() or 1)
GDAL_CACHEMAX_MB = 2048
WARP_MEM_LIMIT_MB = 2048

# 是否为最终 DEM 建立金字塔，方便 QGIS / ArcGIS 快速显示
BUILD_OVERVIEWS = True


# =========================================================
# 2. 几何和窗口工具函数
# =========================================================

def fix_geometry(geom):
    """
    修复无效几何。
    很多行政区划矢量可能存在自相交、空洞异常等问题，
    不修复的话可能导致裁剪失败。
    """
    if geom is None or geom.is_empty:
        return None

    if geom.is_valid:
        return geom

    try:
        from shapely.validation import make_valid
        geom = make_valid(geom)
    except Exception:
        geom = geom.buffer(0)

    if geom is None or geom.is_empty:
        return None

    return geom


def read_boundary_geometry(boundary_file, dst_crs, query=None):
    """
    读取“纽约州 + 新泽西州”的总矢量边界，
    并转换到目标 CRS。

    参数：
    boundary_file : str
        行政区划矢量文件路径。
    dst_crs : rasterio.crs.CRS
        目标坐标系，通常是 DEM 当前坐标系或最终投影坐标系。
    query : str or None
        可选筛选条件。如果矢量已经只包含两个州，保持 None。

    返回：
    geoms : list
        rasterio 可直接使用的 GeoJSON geometry 列表。
    """
    gdf = gpd.read_file(boundary_file)

    if gdf.empty:
        raise ValueError(f"边界文件为空：{boundary_file}")

    if gdf.crs is None:
        raise ValueError(
            f"边界文件没有 CRS，请先在 GIS 软件中定义坐标系：{boundary_file}"
        )

    # 如果边界文件还包含其他州，则使用 query 筛选
    if query is not None:
        gdf = gdf.query(query, engine="python")
        if gdf.empty:
            raise ValueError(f"筛选条件没有选中任何要素：{query}")

    # 修复几何
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(fix_geometry)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    if gdf.empty:
        raise ValueError("边界几何为空或无效。")

    # 关键步骤：统一坐标系
    # 裁剪 DEM 前，必须把矢量边界转到 DEM 的 CRS
    gdf = gdf.to_crs(dst_crs)

    # 把两个州融合成一个整体边界
    union_geom = unary_union(list(gdf.geometry))
    union_geom = fix_geometry(union_geom)

    if union_geom is None or union_geom.is_empty:
        raise ValueError("融合后的 NY-NJ 边界为空。")

    return [mapping(union_geom)]


def iter_windows(width, height, block_size):
    """
    生成分块窗口，避免一次性读取整个 DEM。
    """
    width = int(width)
    height = int(height)

    for row_off in range(0, height, block_size):
        win_h = min(block_size, height - row_off)

        for col_off in range(0, width, block_size):
            win_w = min(block_size, width - col_off)
            yield Window(col_off, row_off, win_w, win_h)


def count_windows(width, height, block_size):
    """
    计算分块数量，用于 tqdm 进度条。
    """
    n_cols = int(np.ceil(width / block_size))
    n_rows = int(np.ceil(height / block_size))
    return n_cols * n_rows


# =========================================================
# 3. DEM 分块裁剪函数
# =========================================================

def clip_raster_by_boundary_blockwise(
    src_raster,
    boundary_file,
    out_raster,
    query=None,
    desc="Clip"
):
    """
    使用一个总边界矢量分块裁剪 DEM。

    这个函数不会一次性把整幅 DEM 读入内存，
    而是按窗口逐块读取、逐块生成 mask、逐块写出。
    对较大 DEM 更稳定。
    """
    src_raster = Path(src_raster)
    out_raster = Path(out_raster)
    out_raster.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_raster, "r", sharing=False) as src:
        if src.crs is None:
            raise ValueError(f"DEM 没有 CRS：{src_raster}")

        # 把 NY-NJ 总边界转到当前 DEM 的 CRS
        geoms = read_boundary_geometry(
            boundary_file=boundary_file,
            dst_crs=src.crs,
            query=query
        )

        # 计算边界和 DEM 的最小相交外接窗口
        try:
            base_window = geometry_window(src, geoms, pad_x=0, pad_y=0)
        except WindowError as e:
            raise RuntimeError(
                f"边界与 DEM 没有空间重叠，请检查 DEM 范围和坐标系：{src_raster}"
            ) from e

        base_window = base_window.round_offsets().round_lengths()

        out_width = int(base_window.width)
        out_height = int(base_window.height)
        out_transform = src.window_transform(base_window)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=out_height,
            width=out_width,
            transform=out_transform,
            count=src.count,
            dtype=OUT_DTYPE,
            nodata=NODATA,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )

        print("\n" + "=" * 80)
        print(f"[裁剪] 输入 DEM: {src_raster}")
        print(f"[裁剪] DEM CRS: {src.crs}")
        print(f"[裁剪] 输出尺寸: {out_width} x {out_height}")
        print(f"[裁剪] 输出文件: {out_raster}")
        print("=" * 80)

        total = count_windows(out_width, out_height, CLIP_BLOCK_SIZE)

        with rasterio.open(out_raster, "w", **profile) as dst:
            with tqdm(total=total, desc=desc, unit="block") as pbar:
                for dst_win in iter_windows(
                    out_width,
                    out_height,
                    CLIP_BLOCK_SIZE
                ):
                    # 目标窗口对应到原始 DEM 中的位置
                    src_win = Window(
                        base_window.col_off + dst_win.col_off,
                        base_window.row_off + dst_win.row_off,
                        dst_win.width,
                        dst_win.height,
                    )

                    # 当前窗口的仿射变换
                    chunk_transform = src.window_transform(src_win)
                    shape = (int(dst_win.height), int(dst_win.width))

                    # 生成当前窗口的行政边界 mask
                    # inside=True 表示在 NY-NJ 边界内部
                    inside = geometry_mask(
                        geoms,
                        out_shape=shape,
                        transform=chunk_transform,
                        invert=True,
                        all_touched=ALL_TOUCHED,
                    )

                    # 如果当前块完全不在边界内，直接写 NoData
                    if not inside.any():
                        nodata_block = np.full(shape, NODATA, dtype=OUT_DTYPE)
                        for band_id in range(1, src.count + 1):
                            dst.write(nodata_block, band_id, window=dst_win)
                        pbar.update(1)
                        continue

                    # 逐波段读取、掩膜、写出
                    for band_id in range(1, src.count + 1):
                        arr = src.read(
                            band_id,
                            window=src_win,
                            masked=True,
                            out_dtype=OUT_DTYPE,
                        )

                        data = np.asarray(arr.filled(NODATA), dtype=OUT_DTYPE)
                        data[~inside] = NODATA

                        dst.write(data, band_id, window=dst_win)

                    pbar.update(1)

    return str(out_raster)


# =========================================================
# 4. DEM 投影转换函数
# =========================================================

def reproject_dem(
    src_raster,
    out_raster,
    target_crs,
    target_resolution,
    desc="Reproject"
):
    """
    将 DEM 重投影到统一投影坐标系。

    注意：
    DEM 是连续变量，所以重采样方法使用 bilinear。
    如果你处理的是分类栅格，应该改成 Resampling.nearest。
    """
    src_raster = Path(src_raster)
    out_raster = Path(out_raster)
    out_raster.parent.mkdir(parents=True, exist_ok=True)

    dst_crs = CRS.from_user_input(target_crs)

    with rasterio.open(src_raster, "r", sharing=False) as src:
        if src.crs is None:
            raise ValueError(f"DEM 没有 CRS：{src_raster}")

        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_resolution, target_resolution),
        )

        # 对齐到规则网格，方便后续 mosaic
        transform, width, height = aligned_target(
            transform,
            width,
            height,
            (target_resolution, target_resolution),
        )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            dtype=OUT_DTYPE,
            nodata=NODATA,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )

        print("\n" + "=" * 80)
        print(f"[投影] 输入 DEM: {src_raster}")
        print(f"[投影] 原始 CRS: {src.crs}")
        print(f"[投影] 目标 CRS: {dst_crs}")
        print(f"[投影] 目标分辨率: {target_resolution} m")
        print(f"[投影] 输出尺寸: {width} x {height}")
        print(f"[投影] 输出文件: {out_raster}")
        print("=" * 80)

        with rasterio.open(out_raster, "w", **profile) as dst:
            for band_id in tqdm(
                range(1, src.count + 1),
                desc=desc,
                unit="band"
            ):
                reproject(
                    source=rasterio.band(src, band_id),
                    destination=rasterio.band(dst, band_id),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=NODATA,
                    resampling=Resampling.bilinear,
                    num_threads=NUM_THREADS,
                    warp_mem_limit=WARP_MEM_LIMIT_MB,
                    init_dest_nodata=True,
                )

    return str(out_raster)


# =========================================================
# 5. 单个 DEM 的处理流程
# =========================================================

def process_one_dem(dem_path, index):
    """
    单个 DEM 的完整处理流程：
    1. 用 NY-NJ 总边界裁剪；
    2. 重投影到 TARGET_CRS；
    3. 返回投影后的 DEM 路径。
    """
    dem_path = Path(dem_path)
    stem = dem_path.stem

    clipped_path = (
        OUT_DIR
        / "01_clipped_src_crs"
        / f"{index:02d}_{stem}_clip_src_crs.tif"
    )

    projected_path = (
        OUT_DIR
        / "02_projected"
        / f"{index:02d}_{stem}_{TARGET_CRS.replace(':', '_')}_{TARGET_RESOLUTION}m.tif"
    )

    clipped = clip_raster_by_boundary_blockwise(
        src_raster=dem_path,
        boundary_file=BOUNDARY_FILE,
        out_raster=clipped_path,
        query=BOUNDARY_QUERY,
        desc=f"Clip DEM {index}"
    )

    projected = reproject_dem(
        src_raster=clipped,
        out_raster=projected_path,
        target_crs=TARGET_CRS,
        target_resolution=TARGET_RESOLUTION,
        desc=f"Reproject DEM {index}"
    )

    return projected


# =========================================================
# 6. 拼接 DEM
# =========================================================

def mosaic_projected_dems(src_paths, out_raster):
    """
    合并已经投影一致、分辨率一致的 DEM。

    method='first' 表示重叠区域优先使用前面的 DEM。
    如果两个 DEM 来自同一数据源，重叠区通常差异很小。
    """
    out_raster = Path(out_raster)
    out_raster.parent.mkdir(parents=True, exist_ok=True)

    datasets = [rasterio.open(p, "r", sharing=False) for p in src_paths]

    try:
        base = datasets[0]

        dst_kwds = base.profile.copy()
        dst_kwds.update(
            driver="GTiff",
            dtype=OUT_DTYPE,
            nodata=NODATA,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )

        print("\n" + "=" * 80)
        print("[拼接] 输入文件：")
        for p in src_paths:
            print(f"  - {p}")
        print(f"[拼接] 输出文件: {out_raster}")
        print("=" * 80)

        merge(
            datasets,
            dst_path=str(out_raster),
            dst_kwds=dst_kwds,
            method="first",
            nodata=NODATA,
            resampling=Resampling.nearest,
            target_aligned_pixels=True,
            mem_limit=1024,
        )

    finally:
        for ds in datasets:
            ds.close()

    return str(out_raster)


# =========================================================
# 7. 建金字塔
# =========================================================

def build_overviews(raster_path):
    """
    为最终 DEM 建立金字塔。
    这样在 QGIS / ArcGIS 中加载和缩放会快很多。
    """
    if not BUILD_OVERVIEWS:
        return

    raster_path = Path(raster_path)

    print("\n" + "=" * 80)
    print(f"[金字塔] 开始建立 overviews: {raster_path}")
    print("=" * 80)

    with rasterio.open(raster_path, "r+") as dst:
        factors = [2, 4, 8, 16, 32, 64]
        dst.build_overviews(factors, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    print("[金字塔] 完成。")


# =========================================================
# 8. 主函数
# =========================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_options = {
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "NUM_THREADS": "ALL_CPUS",
        "GDAL_CACHEMAX": GDAL_CACHEMAX_MB,
        "CHECK_WITH_INVERT_PROJ": "YES",
    }

    print("\n" + "#" * 80)
    print("NY-NJ DEM 裁剪、投影、拼接开始")
    print("#" * 80)
    print(f"DEM 数量: {len(DEM_FILES)}")
    print(f"边界文件: {BOUNDARY_FILE}")
    print(f"目标 CRS: {TARGET_CRS}")
    print(f"目标分辨率: {TARGET_RESOLUTION} m")
    print(f"输出目录: {OUT_DIR}")

    projected_paths = []

    with rasterio.Env(**env_options):
        # 并行处理两个 DEM
        # 注意：并行时两个 tqdm 进度条可能会交错显示；
        # 如果你想看得更清楚，可以把 MAX_PARALLEL_DEMS 改成 1。
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DEMS) as executor:
            future_map = {
                executor.submit(process_one_dem, dem_path, i + 1): dem_path
                for i, dem_path in enumerate(DEM_FILES)
            }

            for future in as_completed(future_map):
                dem_path = future_map[future]

                try:
                    projected_path = future.result()
                    projected_paths.append(projected_path)
                except Exception as e:
                    raise RuntimeError(f"DEM 处理失败：{dem_path}\n错误信息：{e}") from e

        # 为了保证拼接顺序稳定，按文件名排序
        projected_paths = sorted(projected_paths)

        mosaic_path = (
            OUT_DIR
            / "03_mosaic"
            / f"NY_NJ_DEM_mosaic_{TARGET_CRS.replace(':', '_')}_{TARGET_RESOLUTION}m.tif"
        )

        mosaic_result = mosaic_projected_dems(
            src_paths=projected_paths,
            out_raster=mosaic_path
        )

        # 最终再用同一个 NY-NJ 总边界精裁一次
        final_path = (
            OUT_DIR
            / "04_final"
            / f"NY_NJ_DEM_final_{TARGET_CRS.replace(':', '_')}_{TARGET_RESOLUTION}m.tif"
        )

        final_result = clip_raster_by_boundary_blockwise(
            src_raster=mosaic_result,
            boundary_file=BOUNDARY_FILE,
            out_raster=final_path,
            query=BOUNDARY_QUERY,
            desc="Final clip"
        )

        build_overviews(final_result)

    print("\n" + "#" * 80)
    print("全部完成")
    print("#" * 80)
    print(f"最终 DEM 输出位置：{final_result}")


if __name__ == "__main__":
    main()