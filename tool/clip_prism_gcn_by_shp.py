# -*- coding: utf-8 -*-
"""
直接使用研究区 shp 裁剪 PRISM 和 GCN250 栅格。

特点：
1. 不改变输出栅格坐标系；
2. 只裁剪空间范围；
3. 自动把 shp 临时转换到每个输入栅格的 CRS；
4. 分块处理，避免一次性读完整大栅格；
5. 输出压缩 GeoTIFF。
"""

import os
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, geometry_window
from rasterio.errors import WindowError
from rasterio.windows import Window
from rasterio.enums import Resampling
from shapely.ops import unary_union
from shapely.geometry import mapping
from tqdm import tqdm


# =========================================================
# 1. 路径配置：主要改这里
# =========================================================

BASE_DIR = Path(r"D:\python\DEM_work\data")

# 你的研究区 shp，例如纽约 + 新泽西边界
BOUNDARY_SHP = BASE_DIR / "shp" / "nyxzx.shp"

# 原始大范围栅格
RASTERS = [
    {
        "name": "AP_PRISM",
        "path": BASE_DIR / "Co_work" / "prism" / "prism_ppt_us_30s_2020_avg_30y.tif",
    },
    {
        "name": "CN_GCN250_ARCI",
        "path": BASE_DIR / "Co_work" / "GCN250" / "GCN250_ARCI.tif",
    },
    {
        "name": "CN_GCN250_ARCII",
        "path": BASE_DIR / "Co_work" / "GCN250" / "GCN250_ARCII.tif",
    },
    {
        "name": "CN_GCN250_ARCIII",
        "path": BASE_DIR / "Co_work" / "GCN250" / "GCN250_ARCIII.tif",
    },
]

OUT_DIR = BASE_DIR / "Co_work" / "clipped_by_shp_src_crs"

# 如果输入栅格没有 NoData，就使用这个
DEFAULT_NODATA = -9999.0

# 分块大小，越大越快但越占内存
BLOCK_SIZE = 2048

# 是否包含边界触碰像元
ALL_TOUCHED = False

# 是否建立金字塔，方便 GIS 软件显示
BUILD_OVERVIEWS = True


# =========================================================
# 2. 工具函数
# =========================================================

def validate_inputs():
    print("\n" + "=" * 80)
    print("[输入文件检查]")
    print("=" * 80)

    missing = []

    if not BOUNDARY_SHP.exists():
        missing.append(BOUNDARY_SHP)

    for item in RASTERS:
        if not item["path"].exists():
            missing.append(item["path"])

    if missing:
        print("[错误] 以下文件不存在：")
        for p in missing:
            print(f"  - {p}")
        raise FileNotFoundError("请检查路径配置。")

    print(f"研究区 shp: {BOUNDARY_SHP}")
    for item in RASTERS:
        print(f"{item['name']}: {item['path']}")

    print("[检查通过]")


def fix_geometry(geom):
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


def read_boundary_for_raster(boundary_shp, raster_crs):
    """
    读取研究区 shp，并临时转换到当前栅格 CRS。
    注意：这里只是用于裁剪，不改变输出栅格 CRS。
    """
    gdf = gpd.read_file(boundary_shp)

    if gdf.empty:
        raise ValueError(f"研究区 shp 为空：{boundary_shp}")

    if gdf.crs is None:
        raise ValueError(f"研究区 shp 没有 CRS，请先定义坐标系：{boundary_shp}")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(fix_geometry)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    if gdf.empty:
        raise ValueError("研究区 shp 几何为空或无效。")

    # 关键：临时转到栅格 CRS，用于正确裁剪
    gdf = gdf.to_crs(raster_crs)

    union_geom = unary_union(list(gdf.geometry))
    union_geom = fix_geometry(union_geom)

    if union_geom is None or union_geom.is_empty:
        raise ValueError("研究区融合后为空。")

    return [mapping(union_geom)]


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


# =========================================================
# 3. 分块裁剪函数
# =========================================================

def clip_raster_by_shp_keep_src_crs(src_raster, boundary_shp, out_raster, name):
    """
    用 shp 裁剪栅格。

    输出结果：
    - CRS 与输入栅格一致；
    - 分辨率与输入栅格一致；
    - 只缩小空间范围；
    - 研究区外写为 NoData。
    """
    src_raster = Path(src_raster)
    out_raster = Path(out_raster)
    out_raster.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_raster) as src:
        if src.crs is None:
            raise ValueError(f"输入栅格没有 CRS：{src_raster}")

        src_nodata = src.nodata
        if src_nodata is None:
            src_nodata = DEFAULT_NODATA

        geoms = read_boundary_for_raster(boundary_shp, src.crs)

        try:
            base_window = geometry_window(src, geoms, pad_x=0, pad_y=0)
        except WindowError as e:
            raise RuntimeError(
                f"{name}: 研究区 shp 与栅格没有空间重叠，请检查坐标系和范围。"
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
            count=1,
            nodata=src_nodata,
            compress="DEFLATE",
            predictor=3 if np.issubdtype(np.dtype(src.dtypes[0]), np.floating) else 2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )

        print("\n" + "=" * 80)
        print(f"[裁剪] {name}")
        print(f"输入: {src_raster}")
        print(f"输出: {out_raster}")
        print(f"输入 CRS: {src.crs}")
        print(f"输出尺寸: {out_width} x {out_height}")
        print(f"NoData: {src_nodata}")
        print("=" * 80)

        total = count_windows(out_width, out_height, BLOCK_SIZE)

        with rasterio.open(out_raster, "w", **profile) as dst:
            for dst_win in tqdm(
                iter_windows(out_width, out_height, BLOCK_SIZE),
                total=total,
                desc=f"Clip {name}",
                unit="block",
            ):
                src_win = Window(
                    base_window.col_off + dst_win.col_off,
                    base_window.row_off + dst_win.row_off,
                    dst_win.width,
                    dst_win.height,
                )

                chunk_transform = src.window_transform(src_win)
                shape = (int(dst_win.height), int(dst_win.width))

                inside = geometry_mask(
                    geoms,
                    out_shape=shape,
                    transform=chunk_transform,
                    invert=True,
                    all_touched=ALL_TOUCHED,
                )

                data = src.read(
                    1,
                    window=src_win,
                    masked=True,
                )

                arr = np.asarray(data.filled(src_nodata))

                arr[~inside] = src_nodata

                dst.write(arr, 1, window=dst_win)

    build_overviews(out_raster)

    print(f"[完成] {out_raster}")

    return out_raster


# =========================================================
# 4. 主函数
# =========================================================

def main():
    validate_inputs()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print("PRISM / GCN250 直接按 shp 裁剪开始")
    print("#" * 80)

    outputs = []

    for item in RASTERS:
        name = item["name"]
        src_path = item["path"]

        out_path = OUT_DIR / f"{name}_clip_by_shp.tif"

        result = clip_raster_by_shp_keep_src_crs(
            src_raster=src_path,
            boundary_shp=BOUNDARY_SHP,
            out_raster=out_path,
            name=name,
        )

        outputs.append(result)

    print("\n" + "#" * 80)
    print("全部裁剪完成")
    print("#" * 80)

    for p in outputs:
        print(f"  - {p}")

    print("\n后续你再把这些裁剪后的栅格统一投影 / 重采样到 DEM 网格即可。")


if __name__ == "__main__":
    main()