# -*- coding: utf-8 -*-
"""
修复 DTRiver / DTRoad 为空的问题。

流程：
1. 从原始 NY_NJ_roads.gpkg / NY_NJ_waterways.gpkg 读取矢量；
2. 强制转到 DEM 的 CRS，推荐 EPSG:5070；
3. 检查矢量与 DEM 是否重叠；
4. 输出修复后的矢量；
5. 栅格化为 0/1 mask；
6. 检查 mask 中目标像元数；
7. 计算距离栅格。
"""

import os
import sys
from pathlib import Path

# 如果你已经配置好 conda env vars，这段也不会有副作用
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
import fiona
from shapely.geometry import box
from osgeo import gdal, ogr
from tqdm import tqdm


# =========================================================
# 1. 路径配置
# =========================================================

DEM_PATH = Path(r"D:\python\DEM_work\data\Features_resolution_30_m\DEM.tif")

# 优先使用原始合并后的 OSM 文件，不要用之前可能投影错的 projected_vectors
RAW_ROADS = Path(
    r"D:\python\DEM_work\data\Co_work\osm\ny_nj_extracted\NY_NJ_roads.gpkg"
)

RAW_WATERWAYS = Path(
    r"D:\python\DEM_work\data\Co_work\osm\ny_nj_extracted\NY_NJ_waterways.gpkg"
)

OUT_VECTOR_DIR = Path(r"D:\python\DEM_work\data\Co_work\projected_vectors_fixed")
OUT_RASTER_DIR = Path(r"D:\python\DEM_work\data\Features_resolution_30_m")

OUT_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
OUT_RASTER_DIR.mkdir(parents=True, exist_ok=True)

NODATA = -9999.0

# 距离最大值，避免全域无限距离计算太慢
DTRIVER_MAX_DISTANCE = 15000
DTROAD_MAX_DISTANCE = 20000

MIN_FEATURE_PIXELS = 10


# =========================================================
# 2. 基础工具
# =========================================================

def print_dem_info():
    with rasterio.open(DEM_PATH) as dem:
        print("=" * 80)
        print("[DEM]")
        print("path:", DEM_PATH)
        print("crs:", dem.crs)
        print("epsg:", dem.crs.to_epsg() if dem.crs else None)
        print("size:", dem.width, dem.height)
        print("res:", dem.res)
        print("bounds:", dem.bounds)
        print("nodata:", dem.nodata)
        print("=" * 80)

        if dem.crs is None:
            raise ValueError("DEM 没有 CRS。")

        if dem.crs.to_epsg() != 5070:
            print("[警告] DEM EPSG 不是 5070，请确认是否正确。")

        return dem.crs, dem.bounds, dem.transform, dem.width, dem.height


def list_gpkg_layers(path: Path):
    print("\n[GPKG 图层]", path)
    layers = fiona.listlayers(path)
    for layer in layers:
        print("  -", layer)
    return layers


def read_first_layer(path: Path):
    layers = list_gpkg_layers(path)
    layer = layers[0]
    print(f"[读取图层] {layer}")
    return gpd.read_file(path, layer=layer)


def clean_line_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("清理前要素数:", len(gdf))

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    # 只保留线要素
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

    print("清理后线要素数:", len(gdf))

    if gdf.empty:
        raise ValueError("清理后没有有效线要素。")

    return gdf


def count_feature_pixels(mask_path: Path, value: int = 1) -> int:
    total = 0

    with rasterio.open(mask_path) as src:
        for _, win in src.block_windows(1):
            arr = src.read(1, window=win)
            total += int(np.count_nonzero(arr == value))

    return total


# =========================================================
# 3. 修复矢量 CRS 并检查重叠
# =========================================================

def prepare_vector_to_dem_crs(raw_vector: Path, out_vector: Path, name: str):
    if not raw_vector.exists():
        raise FileNotFoundError(f"{name} 原始矢量不存在: {raw_vector}")

    dem_crs, dem_bounds, _, _, _ = print_dem_info()
    dem_box = box(*dem_bounds)

    print("\n" + "=" * 80)
    print(f"[{name}] 读取原始矢量")
    print(raw_vector)
    print("=" * 80)

    gdf = read_first_layer(raw_vector)

    print(f"[{name}] 原始 CRS:", gdf.crs)
    print(f"[{name}] 原始 bounds:", gdf.total_bounds)

    if gdf.crs is None:
        raise ValueError(f"{name} 没有 CRS。OSM 通常应为 EPSG:4326。")

    gdf = clean_line_gdf(gdf)

    # 关键：直接转到 DEM CRS。不要使用之前坏掉的 projected_vectors。
    gdf = gdf.to_crs(dem_crs)

    print(f"[{name}] 转到 DEM CRS 后 CRS:", gdf.crs)
    print(f"[{name}] 转到 DEM CRS 后 bounds:", gdf.total_bounds)

    # 先用 cx 做 bbox 过滤，再精确 intersects
    minx, miny, maxx, maxy = dem_bounds
    try:
        gdf_clip = gdf.cx[minx:maxx, miny:maxy].copy()
    except Exception:
        gdf_clip = gdf.copy()

    gdf_clip = gdf_clip[gdf_clip.intersects(dem_box)].copy()

    print(f"[{name}] 与 DEM 范围相交的要素数:", len(gdf_clip))

    if gdf_clip.empty:
        raise RuntimeError(
            f"{name} 与 DEM 没有空间重叠。\n"
            f"请检查：\n"
            f"1. 原始 OSM 文件是否是纽约/新泽西；\n"
            f"2. DEM 是否真的是 EPSG:5070；\n"
            f"3. DEM bounds 是否合理。"
        )

    out_vector.parent.mkdir(parents=True, exist_ok=True)

    if out_vector.exists():
        out_vector.unlink()

    gdf_clip.to_file(out_vector, layer=name, driver="GPKG")

    print(f"[{name}] 修复后矢量已输出:", out_vector)

    return out_vector


# =========================================================
# 4. 栅格化
# =========================================================

def rasterize_vector_to_mask(vector_path: Path, out_mask: Path, name: str):
    dem_crs, dem_bounds, dem_transform, width, height = print_dem_info()

    if out_mask.exists():
        out_mask.unlink()

    print("\n" + "=" * 80)
    print(f"[{name}] 栅格化")
    print("vector:", vector_path)
    print("mask:", out_mask)
    print("=" * 80)

    vec_ds = ogr.Open(str(vector_path))
    if vec_ds is None:
        raise RuntimeError(f"无法打开矢量: {vector_path}")

    layer = vec_ds.GetLayer(0)

    feature_count = layer.GetFeatureCount()
    extent = layer.GetExtent()

    print(f"[{name}] 矢量要素数:", feature_count)
    print(f"[{name}] 矢量范围:", extent)

    if feature_count <= 0:
        raise RuntimeError(f"{name} 矢量要素数为 0。")

    driver = gdal.GetDriverByName("GTiff")

    dst_ds = driver.Create(
        str(out_mask),
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=[
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "COMPRESS=DEFLATE",
            "BIGTIFF=IF_SAFER",
        ],
    )

    if dst_ds is None:
        raise RuntimeError(f"无法创建 mask: {out_mask}")

    dst_ds.SetGeoTransform(dem_transform.to_gdal())
    dst_ds.SetProjection(dem_crs.to_wkt())

    band = dst_ds.GetRasterBand(1)

    # 注意：不要把 0 设置成 NoData。
    # 0 是背景，1 是道路/河流。
    band.Fill(0)

    print(f"[{name}] GDAL RasterizeLayer 开始...")

    err = gdal.RasterizeLayer(
        dst_ds,
        [1],
        layer,
        burn_values=[1],
        options=["ALL_TOUCHED=TRUE"],
    )

    band.FlushCache()
    band = None
    dst_ds = None
    vec_ds = None

    if err != 0:
        raise RuntimeError(f"{name} RasterizeLayer 失败，错误码: {err}")

    feature_pixels = count_feature_pixels(out_mask, value=1)

    print(f"[{name}] mask 目标像元数:", feature_pixels)

    if feature_pixels < MIN_FEATURE_PIXELS:
        raise RuntimeError(
            f"{name} mask 仍然为空或目标像元过少: {feature_pixels}\n"
            f"说明矢量没有被烧入 DEM 网格。"
        )

    return out_mask


# =========================================================
# 5. 距离计算
# =========================================================

def apply_dem_mask_to_distance(distance_path: Path):
    with rasterio.open(DEM_PATH) as dem, rasterio.open(distance_path, "r+") as dst:
        total_blocks = sum(1 for _ in dst.block_windows(1))

        for _, win in tqdm(
            dst.block_windows(1),
            total=total_blocks,
            desc="Apply DEM mask",
            unit="block"
        ):
            dist = dst.read(1, window=win).astype("float32")
            dem_block = dem.read(1, window=win, masked=True)

            valid = ~dem_block.mask
            dist = np.where(valid, dist, NODATA).astype("float32")

            dst.write(dist, 1, window=win)

        dst.nodata = NODATA


def compute_distance(mask_path: Path, out_distance: Path, max_distance: float, name: str):
    if out_distance.exists():
        out_distance.unlink()

    feature_pixels = count_feature_pixels(mask_path, value=1)
    print(f"[{name}] 距离计算前 mask 像元数:", feature_pixels)

    if feature_pixels < MIN_FEATURE_PIXELS:
        raise RuntimeError(f"{name} mask 为空，不能计算距离。")

    print("\n" + "=" * 80)
    print(f"[{name}] 距离计算")
    print("mask:", mask_path)
    print("out:", out_distance)
    print("max_distance:", max_distance)
    print("=" * 80)

    src_ds = gdal.Open(str(mask_path), gdal.GA_ReadOnly)
    src_band = src_ds.GetRasterBand(1)

    driver = gdal.GetDriverByName("GTiff")

    dst_ds = driver.Create(
        str(out_distance),
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=[
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BIGTIFF=IF_SAFER",
        ],
    )

    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())

    dst_band = dst_ds.GetRasterBand(1)
    dst_band.Fill(float(max_distance))

    options = [
        "VALUES=1",
        "DISTUNITS=GEO",
        f"MAXDIST={max_distance}",
    ]

    pbar = tqdm(total=100, desc=f"{name} proximity", unit="%")
    state = {"last": 0}

    def progress_callback(complete, message, data):
        pct = int(complete * 100)
        delta = pct - state["last"]
        if delta > 0:
            pbar.update(delta)
            state["last"] = pct
        return 1

    gdal.ComputeProximity(
        src_band,
        dst_band,
        options,
        callback=progress_callback,
    )

    if state["last"] < 100:
        pbar.update(100 - state["last"])
    pbar.close()

    dst_band.FlushCache()

    src_band = None
    dst_band = None
    src_ds = None
    dst_ds = None

    apply_dem_mask_to_distance(out_distance)

    print(f"[{name}] 距离完成:", out_distance)

    return out_distance


# =========================================================
# 6. 主流程
# =========================================================

def main():
    print_dem_info()

    tasks = [
        {
            "name": "DTRiver",
            "raw_vector": RAW_WATERWAYS,
            "fixed_vector": OUT_VECTOR_DIR / "NY_NJ_waterways_EPSG5070_fixed.gpkg",
            "mask": OUT_RASTER_DIR / "DTRiver_mask.tif",
            "distance": OUT_RASTER_DIR / "DTRiver.tif",
            "max_distance": DTRIVER_MAX_DISTANCE,
        },
        {
            "name": "DTRoad",
            "raw_vector": RAW_ROADS,
            "fixed_vector": OUT_VECTOR_DIR / "NY_NJ_roads_EPSG5070_fixed.gpkg",
            "mask": OUT_RASTER_DIR / "DTRoad_mask.tif",
            "distance": OUT_RASTER_DIR / "DTRoad.tif",
            "max_distance": DTROAD_MAX_DISTANCE,
        },
    ]

    for task in tasks:
        name = task["name"]

        fixed_vector = prepare_vector_to_dem_crs(
            raw_vector=task["raw_vector"],
            out_vector=task["fixed_vector"],
            name=name,
        )

        mask = rasterize_vector_to_mask(
            vector_path=fixed_vector,
            out_mask=task["mask"],
            name=name,
        )

        compute_distance(
            mask_path=mask,
            out_distance=task["distance"],
            max_distance=task["max_distance"],
            name=name,
        )

    print("\n全部完成。")


if __name__ == "__main__":
    main()