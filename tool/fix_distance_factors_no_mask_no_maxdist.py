# -*- coding: utf-8 -*-
"""
生成 DTRiver / DTRoad 距离因子：不保留 mask，不设置最大距离限制。

流程：
1. 读取原始 NY_NJ_roads.gpkg / NY_NJ_waterways.gpkg；
2. 自动选择 roads / waterways 图层；
3. 只保留线要素，强制转换到 DEM CRS；
4. 用 DEM 范围裁剪矢量；
5. 生成临时 0/1 mask；
6. 用 GDAL ComputeProximity 计算全域真实距离，不使用 MAXDIST；
7. 用 DEM 有效区域清理 NoData；
8. 删除临时 mask / 临时矢量，只保留 DTRiver.tif 和 DTRoad.tif。

注意：
- 这个脚本不会输出 DTRiver_mask.tif / DTRoad_mask.tif。
- 由于不设置 MAXDIST，距离计算会明显慢于截断版本，但不会出现大片最大值平台。
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple

# =========================================================
# 0. PROJ / GDAL 环境兜底
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
import fiona
from shapely.geometry import box
from osgeo import gdal, ogr
from tqdm import tqdm

try:
    gdal.UseExceptions()
except Exception:
    pass


# =========================================================
# 1. 路径配置
# =========================================================

DEM_PATH = Path(r"D:\python\DEM_work\data\Features_resolution_30_m\DEM.tif")

# 使用原始合并后的 OSM 文件，不使用旧的 projected_vectors
RAW_ROADS = Path(
    r"D:\python\DEM_work\data\Co_work\osm\ny_nj_extracted\NY_NJ_roads.gpkg"
)

RAW_WATERWAYS = Path(
    r"D:\python\DEM_work\data\Co_work\osm\ny_nj_extracted\NY_NJ_waterways.gpkg"
)

OUT_RASTER_DIR = Path(r"D:\python\DEM_work\data\Features_resolution_30_m")
OUT_RASTER_DIR.mkdir(parents=True, exist_ok=True)

# 临时目录：脚本运行结束后默认删除
TMP_DIR = OUT_RASTER_DIR / "_tmp_distance_no_mask"

NODATA = -9999.0
MIN_FEATURE_PIXELS = 10

# mask 中目标像元比例阈值：防止把整个区域都烧成河流/道路
MAX_RIVER_RATIO = 0.20
MAX_ROAD_RATIO = 0.60

# 是否删除历史遗留的 DTRiver_mask.tif / DTRoad_mask.tif
DELETE_OLD_MASK_OUTPUTS = True

# 是否保留临时文件。调试时可改 True。
KEEP_TEMP_FILES = False

# 是否建立金字塔
BUILD_OVERVIEWS = True

# GDAL 缓存，单位 MB
GDAL_CACHEMAX_MB = 4096


# =========================================================
# 2. 基础工具
# =========================================================

def print_dem_info() -> Tuple[object, object, object, int, int]:
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
        print("dtype:", dem.dtypes[0])
        print("=" * 80)

        if dem.crs is None:
            raise ValueError("DEM 没有 CRS。")

        if dem.crs.is_geographic:
            raise ValueError("DEM 是经纬度坐标系，距离计算前应先投影到米制坐标系，例如 EPSG:5070。")

        if dem.crs.to_epsg() != 5070:
            print("[警告] DEM EPSG 不是 5070，请确认是否符合你的研究设计。")

        return dem.crs, dem.bounds, dem.transform, dem.width, dem.height


def list_gpkg_layers(path: Path):
    layers = fiona.listlayers(path)
    print("\n[GPKG 图层]", path)
    for layer in layers:
        print("  -", layer)
    return layers


def choose_layer(path: Path, task_name: str) -> str:
    """
    自动选择 GPKG 图层：
    - DTRoad 优先匹配 road / roads；
    - DTRiver 优先匹配 waterway / waterways，避免误选 water polygon。
    """
    layers = list_gpkg_layers(path)

    if len(layers) == 1:
        print(f"[图层选择] 只有一个图层，使用：{layers[0]}")
        return layers[0]

    if task_name.lower() == "dtroad":
        prefer = ["roads", "road"]
    else:
        prefer = ["waterways", "waterway"]

    # 严格优先：图层名包含 prefer
    for key in prefer:
        for layer in layers:
            if key in layer.lower():
                print(f"[图层选择] {task_name} 使用图层：{layer}")
                return layer

    raise RuntimeError(
        f"无法为 {task_name} 自动选择图层。请检查 {path} 的图层列表，"
        f"并在 choose_layer() 中手动指定。"
    )


def read_vector_with_bbox(raw_vector: Path, layer: str, dem_crs, dem_bounds) -> gpd.GeoDataFrame:
    """
    读取矢量。若能获取源 CRS，则先把 DEM bounds 转到源 CRS 做 bbox 过滤，减少内存。
    """
    print("\n" + "=" * 80)
    print("[读取矢量]")
    print("file:", raw_vector)
    print("layer:", layer)
    print("=" * 80)

    # 先读取 1 行拿 CRS
    try:
        sample = gpd.read_file(raw_vector, layer=layer, rows=1)
        src_crs = sample.crs
    except Exception:
        src_crs = None

    if src_crs is None:
        print("[提示] 无法从 sample 获取 CRS，将直接完整读取。")
        gdf = gpd.read_file(raw_vector, layer=layer)
        return gdf

    # DEM bounds 转到矢量源 CRS
    try:
        dem_bbox_src = gpd.GeoSeries([box(*dem_bounds)], crs=dem_crs).to_crs(src_crs).total_bounds
        bbox_tuple = tuple(float(x) for x in dem_bbox_src)
        print("[BBox 过滤] 源 CRS:", src_crs)
        print("[BBox 过滤] bbox:", bbox_tuple)
        gdf = gpd.read_file(raw_vector, layer=layer, bbox=bbox_tuple)
    except Exception as e:
        print("[BBox 过滤失败] 将完整读取。原因:", repr(e))
        gdf = gpd.read_file(raw_vector, layer=layer)

    return gdf


def clean_line_gdf(gdf: gpd.GeoDataFrame, task_name: str) -> gpd.GeoDataFrame:
    print(f"[{task_name}] 清理前要素数:", len(gdf))
    print(f"[{task_name}] 原始 CRS:", gdf.crs)
    if not gdf.empty:
        print(f"[{task_name}] 原始 bounds:", gdf.total_bounds)
        print(f"[{task_name}] 几何类型统计:")
        print(gdf.geometry.geom_type.value_counts(dropna=False))

    if gdf.crs is None:
        raise ValueError(f"{task_name} 矢量没有 CRS。OSM 通常应为 EPSG:4326。")

    # 有效几何
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    # 只保留线要素。这样可以避免把水体 polygon 烧成大片 1。
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

    if gdf.empty:
        raise ValueError(f"{task_name} 清理后没有有效线要素。请确认图层是不是线状 roads / waterways。")

    # DTRiver 如果有 fclass / waterway 字段，进一步过滤常见线状水系类型
    if task_name.lower() == "dtriver":
        keep_classes = {"river", "stream", "canal", "drain", "ditch", "wadi"}
        class_col = None
        for col in ["fclass", "waterway", "type", "class"]:
            if col in gdf.columns:
                class_col = col
                break

        if class_col is not None:
            print(f"[DTRiver] 使用字段 {class_col} 过滤线状水系类型。")
            print(gdf[class_col].value_counts(dropna=False).head(30))
            before = len(gdf)
            vals = gdf[class_col].astype(str).str.lower()
            gdf = gdf[vals.isin(keep_classes)].copy()
            print(f"[DTRiver] 类型过滤前: {before}, 过滤后: {len(gdf)}")

            if gdf.empty:
                raise RuntimeError(
                    f"DTRiver 按 {class_col} 过滤后没有要素。"
                    f"请检查字段取值，或调整 keep_classes。"
                )
        else:
            print("[DTRiver] 未发现 fclass/waterway/type/class 字段，仅按线几何过滤。")

    print(f"[{task_name}] 清理后线要素数:", len(gdf))
    return gdf


def prepare_vector_to_temp_gpkg(raw_vector: Path, out_vector: Path, task_name: str) -> Path:
    if not raw_vector.exists():
        raise FileNotFoundError(f"{task_name} 原始矢量不存在: {raw_vector}")

    dem_crs, dem_bounds, _, _, _ = print_dem_info()
    dem_box = box(*dem_bounds)

    layer = choose_layer(raw_vector, task_name)
    gdf = read_vector_with_bbox(raw_vector, layer, dem_crs, dem_bounds)
    gdf = clean_line_gdf(gdf, task_name)

    # 转到 DEM CRS
    gdf = gdf.to_crs(dem_crs)
    print(f"[{task_name}] 转到 DEM CRS 后 CRS:", gdf.crs)
    print(f"[{task_name}] 转到 DEM CRS 后 bounds:", gdf.total_bounds)

    # bbox + intersects 裁剪
    minx, miny, maxx, maxy = dem_bounds
    try:
        gdf_clip = gdf.cx[minx:maxx, miny:maxy].copy()
    except Exception:
        gdf_clip = gdf.copy()

    gdf_clip = gdf_clip[gdf_clip.intersects(dem_box)].copy()
    print(f"[{task_name}] 与 DEM 范围相交的要素数:", len(gdf_clip))

    if gdf_clip.empty:
        raise RuntimeError(
            f"{task_name} 与 DEM 没有空间重叠。请检查原始 OSM 文件、图层和 DEM 坐标系。"
        )

    # 只保存 geometry，降低临时文件体积
    gdf_clip = gdf_clip[["geometry"]].copy()

    out_vector.parent.mkdir(parents=True, exist_ok=True)
    if out_vector.exists():
        out_vector.unlink()

    gdf_clip.to_file(out_vector, layer=task_name, driver="GPKG")
    print(f"[{task_name}] 临时矢量已写出:", out_vector)
    return out_vector


def count_feature_pixels(mask_path: Path, value: int = 1) -> Tuple[int, int]:
    """
    返回：目标像元数、总像元数。
    """
    total_feature = 0
    total_pixels = 0
    with rasterio.open(mask_path) as src:
        for _, win in src.block_windows(1):
            arr = src.read(1, window=win)
            total_feature += int(np.count_nonzero(arr == value))
            total_pixels += arr.size
    return total_feature, total_pixels


def remove_nodata_from_mask(mask_path: Path):
    """
    距离计算的 mask 必须满足：
    0 = 背景，1 = 目标要素；0 不能是 NoData。
    """
    ds = gdal.Open(str(mask_path), gdal.GA_Update)
    if ds is None:
        raise RuntimeError(f"无法打开 mask: {mask_path}")

    band = ds.GetRasterBand(1)
    old_nodata = band.GetNoDataValue()
    if old_nodata is not None:
        print(f"[Mask NoData 修复] 删除 NoData={old_nodata}: {mask_path}")
        band.DeleteNoDataValue()
    band.FlushCache()
    ds = None


def cleanup_file(path: Path):
    """
    删除文件及常见 sidecar。
    """
    path = Path(path)
    candidates = [
        path,
        Path(str(path) + ".aux.xml"),
        Path(str(path) + ".ovr"),
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        Path(str(path) + "-journal"),
    ]
    for p in candidates:
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            print(f"[临时文件删除警告] {p}: {e}")


# =========================================================
# 3. 临时 mask 栅格化
# =========================================================

def rasterize_vector_to_temp_mask(vector_path: Path, temp_mask: Path, task_name: str) -> Path:
    dem_crs, _, dem_transform, width, height = print_dem_info()

    if temp_mask.exists():
        temp_mask.unlink()

    print("\n" + "=" * 80)
    print(f"[{task_name}] 生成临时 mask，不会作为最终结果保留")
    print("vector:", vector_path)
    print("temp mask:", temp_mask)
    print("=" * 80)

    vec_ds = ogr.Open(str(vector_path))
    if vec_ds is None:
        raise RuntimeError(f"无法打开矢量: {vector_path}")

    layer = vec_ds.GetLayer(0)
    feature_count = layer.GetFeatureCount()
    extent = layer.GetExtent()

    print(f"[{task_name}] 矢量要素数:", feature_count)
    print(f"[{task_name}] 矢量范围:", extent)

    if feature_count <= 0:
        raise RuntimeError(f"{task_name} 矢量要素数为 0。")

    driver = gdal.GetDriverByName("GTiff")
    temp_mask.parent.mkdir(parents=True, exist_ok=True)

    dst_ds = driver.Create(
        str(temp_mask),
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
        raise RuntimeError(f"无法创建临时 mask: {temp_mask}")

    dst_ds.SetGeoTransform(dem_transform.to_gdal())
    dst_ds.SetProjection(dem_crs.to_wkt())

    band = dst_ds.GetRasterBand(1)
    # 不设置 NoData。0 是背景，1 是目标。
    band.Fill(0)

    print(f"[{task_name}] GDAL RasterizeLayer 开始...")
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
        raise RuntimeError(f"{task_name} RasterizeLayer 失败，错误码: {err}")

    remove_nodata_from_mask(temp_mask)

    feature_pixels, total_pixels = count_feature_pixels(temp_mask, value=1)
    ratio = feature_pixels / total_pixels if total_pixels else 0

    print(f"[{task_name}] 临时 mask 目标像元数: {feature_pixels}")
    print(f"[{task_name}] 临时 mask 总像元数: {total_pixels}")
    print(f"[{task_name}] 临时 mask 目标像元比例: {ratio:.8f}")

    if feature_pixels < MIN_FEATURE_PIXELS:
        raise RuntimeError(
            f"{task_name} 临时 mask 为空或目标像元过少: {feature_pixels}。"
            f"请检查矢量是否与 DEM 重叠。"
        )

    max_ratio = MAX_ROAD_RATIO if task_name.lower() == "dtroad" else MAX_RIVER_RATIO
    if ratio > max_ratio:
        raise RuntimeError(
            f"{task_name} 临时 mask 中 1 的比例过高: {ratio:.4f}。"
            f"这通常说明读错图层，或把面状水体烧进去了。"
        )

    return temp_mask


# =========================================================
# 4. 距离计算：无最大距离限制
# =========================================================

def apply_dem_mask_to_distance(distance_path: Path):
    """
    研究区外写为 NoData。
    """
    with rasterio.open(DEM_PATH) as dem, rasterio.open(distance_path, "r+") as dst:
        if dem.width != dst.width or dem.height != dst.height:
            raise ValueError("DEM 与距离栅格行列数不一致。")
        if dem.crs != dst.crs:
            raise ValueError(f"DEM 与距离栅格 CRS 不一致: {dem.crs} vs {dst.crs}")
        if not np.allclose(tuple(dem.transform), tuple(dst.transform)):
            raise ValueError("DEM 与距离栅格 transform 不一致。")

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


def build_overviews_safe(raster_path: Path):
    if not BUILD_OVERVIEWS:
        return
    try:
        with rasterio.open(raster_path, "r+") as dst:
            factors = [2, 4, 8, 16, 32, 64]
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
        print(f"[金字塔] 完成: {raster_path}")
    except Exception as e:
        print(f"[警告] 金字塔建立失败，但不影响主结果: {e}")


def compute_distance_no_maxdist(temp_mask: Path, out_distance: Path, task_name: str) -> Path:
    """
    使用 GDAL ComputeProximity 计算全域真实距离：
    - 不传 MAXDIST；
    - 不设置最大值截断；
    - 输出单位为米，因为 DEM 是米制投影坐标系。
    """
    if out_distance.exists():
        out_distance.unlink()
        print(f"[{task_name}] 已删除旧距离文件: {out_distance}")

    feature_pixels, total_pixels = count_feature_pixels(temp_mask, value=1)
    ratio = feature_pixels / total_pixels if total_pixels else 0

    print(f"[{task_name}] 距离计算前目标像元数: {feature_pixels}")
    print(f"[{task_name}] 距离计算前目标像元比例: {ratio:.8f}")

    if feature_pixels < MIN_FEATURE_PIXELS:
        raise RuntimeError(f"{task_name} 临时 mask 为空，不能计算距离。")

    print("\n" + "=" * 80)
    print(f"[{task_name}] 距离计算：无最大值限制")
    print("temp mask:", temp_mask)
    print("out:", out_distance)
    print("MAXDIST: 不使用")
    print("=" * 80)

    remove_nodata_from_mask(temp_mask)

    src_ds = gdal.Open(str(temp_mask), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"无法打开临时 mask: {temp_mask}")

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

    if dst_ds is None:
        raise RuntimeError(f"无法创建距离栅格: {out_distance}")

    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())

    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(NODATA)
    dst_band.Fill(NODATA)

    # 关键：不设置 MAXDIST，也不设置 NODATA 为最大值。
    options = [
        "VALUES=1",
        "DISTUNITS=GEO",
    ]

    pbar = tqdm(total=100, desc=f"{task_name} full proximity", unit="%")
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
    build_overviews_safe(out_distance)

    print(f"[{task_name}] 距离完成: {out_distance}")
    return out_distance


def check_distance_stats(distance_path: Path, task_name: str):
    """
    输出距离结果统计，确认不是全 NoData / 全常数。
    """
    min_v = np.inf
    max_v = -np.inf
    total_sum = 0.0
    total_count = 0

    with rasterio.open(distance_path) as src:
        nodata = src.nodata
        for _, win in tqdm(list(src.block_windows(1)), desc=f"Check {task_name} DT", unit="block"):
            arr = src.read(1, window=win).astype("float32")
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata
            if valid.any():
                vals = arr[valid]
                min_v = min(min_v, float(vals.min()))
                max_v = max(max_v, float(vals.max()))
                total_sum += float(vals.sum())
                total_count += int(vals.size)

    if total_count == 0:
        print(f"[{task_name}] 距离结果没有有效像元。")
        return

    print("\n" + "-" * 80)
    print(f"[{task_name}] 距离统计")
    print("valid_count:", total_count)
    print("min:", min_v)
    print("max:", max_v)
    print("mean:", total_sum / total_count)
    print("-" * 80)


# =========================================================
# 5. 主流程
# =========================================================

def delete_old_output_masks():
    if not DELETE_OLD_MASK_OUTPUTS:
        return
    for p in [
        OUT_RASTER_DIR / "DTRiver_mask.tif",
        OUT_RASTER_DIR / "DTRoad_mask.tif",
    ]:
        cleanup_file(p)
        print(f"[清理] 已尝试删除历史 mask: {p}")


def process_one_task(task: dict):
    task_name = task["name"]
    raw_vector = task["raw_vector"]
    out_distance = task["distance"]

    print("\n" + "#" * 80)
    print(f"[{task_name}] 开始处理：只输出 DT，不保留 mask，不设置 MAXDIST")
    print("#" * 80)

    tmp_vector = TMP_DIR / f"{task_name}_fixed_tmp.gpkg"
    tmp_mask = TMP_DIR / f"{task_name}_mask_tmp.tif"

    try:
        fixed_vector = prepare_vector_to_temp_gpkg(
            raw_vector=raw_vector,
            out_vector=tmp_vector,
            task_name=task_name,
        )

        temp_mask = rasterize_vector_to_temp_mask(
            vector_path=fixed_vector,
            temp_mask=tmp_mask,
            task_name=task_name,
        )

        result = compute_distance_no_maxdist(
            temp_mask=temp_mask,
            out_distance=out_distance,
            task_name=task_name,
        )

        check_distance_stats(result, task_name)
        return result

    finally:
        if not KEEP_TEMP_FILES:
            cleanup_file(tmp_mask)
            cleanup_file(tmp_vector)
            print(f"[{task_name}] 临时 mask / 临时矢量已清理。")
        else:
            print(f"[{task_name}] KEEP_TEMP_FILES=True，临时文件保留在: {TMP_DIR}")


def main():
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_CACHEMAX", str(GDAL_CACHEMAX_MB))

    print("\n" + "#" * 80)
    print("DTRiver / DTRoad 全域距离计算开始")
    print("只保留 DT 栅格，不保留 mask，不使用最大距离限制")
    print("#" * 80)

    print_dem_info()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    delete_old_output_masks()

    tasks = [
        {
            "name": "DTRiver",
            "raw_vector": RAW_WATERWAYS,
            "distance": OUT_RASTER_DIR / "DTRiver.tif",
        },
        {
            "name": "DTRoad",
            "raw_vector": RAW_ROADS,
            "distance": OUT_RASTER_DIR / "DTRoad.tif",
        },
    ]

    outputs = []
    for task in tasks:
        outputs.append(process_one_task(task))

    if not KEEP_TEMP_FILES:
        try:
            if TMP_DIR.exists() and not any(TMP_DIR.iterdir()):
                TMP_DIR.rmdir()
            elif TMP_DIR.exists():
                # 尝试彻底删除临时目录，若有锁定文件则保留并提示
                shutil.rmtree(TMP_DIR, ignore_errors=True)
        except Exception as e:
            print(f"[临时目录清理警告] {e}")

    print("\n" + "#" * 80)
    print("全部完成。最终输出：")
    for p in outputs:
        print("  -", p)
    print("#" * 80)

    return outputs


if __name__ == "__main__":
    main()
