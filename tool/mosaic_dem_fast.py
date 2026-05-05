import os
import math
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds, bounds as window_bounds
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from tqdm import tqdm


# =========================================================
# 1. 输入输出路径
# =========================================================

DEM_PATHS = [
    r"E:\Data\DEM\北京市_DEM_30m分辨率_SRTM数据.tif",
    r"E:\Data\DEM\河北省_DEM_30m分辨率_SRTM数据.tif",
    r"E:\Data\DEM\天津市_DEM_30m分辨率_SRTM数据.tif",
]

OUT_DIR = r"E:\Data\DEM\mosaic"
os.makedirs(OUT_DIR, exist_ok=True)

MOSAIC_OUT = os.path.join(OUT_DIR, "京津冀_DEM_30m_mosaic.tif")
FILLED_OUT = os.path.join(OUT_DIR, "京津冀_DEM_30m_mosaic_filled.tif")


# =========================================================
# 2. 参数设置
# =========================================================

# 输出 NoData，建议 float32 DEM 用 -9999
OUT_NODATA = -9999.0
OUT_DTYPE = "float32"

# 分块大小
# 2048 稳定；内存大可以改成 4096
BLOCK_SIZE = 2048

# 读取时的重采样方法
# 你的 3 个 DEM 已经同 CRS、同分辨率，所以 nearest 最快且不会改高程值
READ_RESAMPLING = Resampling.nearest

# 是否只填补已有 NoData
# True：先来的 DEM 保留，后来的 DEM 只补空洞
# False：后来的 DEM 会覆盖前面的 DEM
FILL_ONLY_NODATA = True

# 是否做补漏
DO_FILL_NODATA = True

# 补漏最大搜索距离，单位：像元
# 只建议补小洞。30m DEM 下，50 像元约 1.5km
FILL_MAX_SEARCH_DISTANCE = 50

# 分块补漏时的边缘缓冲，避免块边缘补漏不连续
FILL_HALO = 80

# 压缩设置
# 想最快：COMPRESS = None
# 想文件小一点：COMPRESS = "DEFLATE"
COMPRESS = "DEFLATE"

# GDAL 缓存
GDAL_CACHEMAX_MB = 2048


# =========================================================
# 3. 工具函数
# =========================================================

def print_src_info(src):
    print("=" * 80)
    print(f"文件: {src.name}")
    print(f"CRS: {src.crs}")
    print(f"尺寸: {src.width} x {src.height}")
    print(f"分辨率: {src.res}")
    print(f"范围: {src.bounds}")
    print(f"数据类型: {src.dtypes[0]}")
    print(f"NoData: {src.nodata}")


def check_crs(srcs):
    base_crs = srcs[0].crs
    for src in srcs[1:]:
        if src.crs != base_crs:
            raise ValueError(
                "检测到输入 DEM 的 CRS 不一致。这个稳定版脚本要求 CRS 一致。"
                "如果 CRS 不一致，需要先统一投影。"
            )


def get_target_resolution(srcs):
    """
    自动使用最细分辨率。
    你的数据是 0.000277777777777777 度。
    """
    xres = min(abs(src.res[0]) for src in srcs)
    yres = min(abs(src.res[1]) for src in srcs)
    return xres, yres


def align_bounds_to_resolution(left, bottom, right, top, xres, yres):
    """
    将输出范围对齐到像元网格，避免边界偏移。
    """
    left_aligned = math.floor(left / xres) * xres
    right_aligned = math.ceil(right / xres) * xres
    bottom_aligned = math.floor(bottom / yres) * yres
    top_aligned = math.ceil(top / yres) * yres
    return left_aligned, bottom_aligned, right_aligned, top_aligned


def get_union_bounds(srcs):
    left = min(src.bounds.left for src in srcs)
    bottom = min(src.bounds.bottom for src in srcs)
    right = max(src.bounds.right for src in srcs)
    top = max(src.bounds.top for src in srcs)
    return left, bottom, right, top


def generate_windows(width, height, block_size):
    for row_off in range(0, height, block_size):
        win_h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            win_w = min(block_size, width - col_off)
            yield Window(col_off, row_off, win_w, win_h)


def count_windows(width, height, block_size):
    return math.ceil(width / block_size) * math.ceil(height / block_size)


def make_valid_mask(data, src_nodata):
    """
    判断源 DEM 中哪些像元有效。
    """
    valid = np.isfinite(data)

    if src_nodata is not None:
        valid &= data != src_nodata

    valid &= data != OUT_NODATA

    return valid


def read_source_to_output_block(src, dst_win, dst_transform, dst_height, dst_width):
    """
    按输出窗口的空间范围，从源 DEM 读取同范围数据。
    不再手动计算源窗口偏移，减少窗口错位问题。
    """
    left, bottom, right, top = window_bounds(dst_win, dst_transform)

    src_win = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src.transform
    )

    src_nodata = src.nodata
    if src_nodata is None:
        src_nodata = OUT_NODATA

    data = src.read(
        1,
        window=src_win,
        out_shape=(dst_height, dst_width),
        boundless=True,
        fill_value=src_nodata,
        resampling=READ_RESAMPLING
    )

    return data, src_nodata


def copy_profile_for_output(src, out_width, out_height, out_transform, out_crs):
    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=out_height,
        width=out_width,
        count=1,
        crs=out_crs,
        transform=out_transform,
        dtype=OUT_DTYPE,
        nodata=OUT_NODATA,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER"
    )

    if COMPRESS is not None:
        profile.update(
            compress=COMPRESS,
            predictor=3,
            zlevel=1
        )
    else:
        profile.pop("compress", None)
        profile.pop("predictor", None)
        profile.pop("zlevel", None)

    return profile


def build_overviews(tif_path):
    """
    建立金字塔，QGIS/ArcGIS 打开会顺畅很多。
    """
    print("\n正在建立金字塔...")
    with rasterio.open(tif_path, "r+") as dst:
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
    print("金字塔建立完成。")


# =========================================================
# 4. 镶嵌
# =========================================================

def mosaic_dem():
    with rasterio.Env(
        GDAL_CACHEMAX=GDAL_CACHEMAX_MB,
        NUM_THREADS="ALL_CPUS",
        CHECK_DISK_FREE_SPACE="FALSE"
    ):
        srcs = [rasterio.open(p) for p in DEM_PATHS]

        try:
            print("\n输入 DEM 信息：")
            for src in srcs:
                print_src_info(src)

            check_crs(srcs)

            out_crs = srcs[0].crs
            xres, yres = get_target_resolution(srcs)

            left, bottom, right, top = get_union_bounds(srcs)
            left, bottom, right, top = align_bounds_to_resolution(
                left, bottom, right, top, xres, yres
            )

            out_width = int(round((right - left) / xres))
            out_height = int(round((top - bottom) / yres))
            out_transform = from_origin(left, top, xres, yres)

            print("\n输出 DEM 信息：")
            print("=" * 80)
            print(f"输出文件: {MOSAIC_OUT}")
            print(f"输出 CRS: {out_crs}")
            print(f"输出尺寸: {out_width} x {out_height}")
            print(f"输出分辨率: ({xres}, {yres})")
            print(f"输出 NoData: {OUT_NODATA}")
            print(f"输出数据类型: {OUT_DTYPE}")
            print("=" * 80)

            profile = copy_profile_for_output(
                srcs[0],
                out_width,
                out_height,
                out_transform,
                out_crs
            )

            total_windows = count_windows(out_width, out_height, BLOCK_SIZE)

            with rasterio.open(MOSAIC_OUT, "w", **profile) as dst:
                print("\n开始镶嵌 DEM...")

                for dst_win in tqdm(
                    generate_windows(out_width, out_height, BLOCK_SIZE),
                    total=total_windows,
                    desc="Mosaic",
                    unit="block"
                ):
                    h = int(dst_win.height)
                    w = int(dst_win.width)

                    out_block = np.full(
                        (h, w),
                        OUT_NODATA,
                        dtype=np.float32
                    )

                    for src in srcs:
                        data, src_nodata = read_source_to_output_block(
                            src,
                            dst_win,
                            out_transform,
                            h,
                            w
                        )

                        data = data.astype(np.float32, copy=False)
                        valid = make_valid_mask(data, src_nodata)

                        if not valid.any():
                            continue

                        if FILL_ONLY_NODATA:
                            replace_mask = valid & (out_block == OUT_NODATA)
                        else:
                            replace_mask = valid

                        if replace_mask.any():
                            out_block[replace_mask] = data[replace_mask]

                    dst.write(out_block.astype(OUT_DTYPE), 1, window=dst_win)

            print("\n镶嵌完成：")
            print(MOSAIC_OUT)

        finally:
            for src in srcs:
                src.close()


# =========================================================
# 5. 分块补漏
# =========================================================

def expand_window(win, width, height, halo):
    """
    给补漏窗口增加缓冲区。
    """
    col_off = max(0, int(win.col_off) - halo)
    row_off = max(0, int(win.row_off) - halo)

    col_end = min(width, int(win.col_off + win.width) + halo)
    row_end = min(height, int(win.row_off + win.height) + halo)

    return Window(
        col_off,
        row_off,
        col_end - col_off,
        row_end - row_off
    )


def crop_center_from_expanded(expanded_data, small_win, expanded_win):
    """
    从带 halo 的补漏结果中裁剪回原始窗口大小。
    """
    row_start = int(small_win.row_off - expanded_win.row_off)
    col_start = int(small_win.col_off - expanded_win.col_off)

    row_end = row_start + int(small_win.height)
    col_end = col_start + int(small_win.width)

    return expanded_data[row_start:row_end, col_start:col_end]


def fill_nodata_by_blocks():
    if not DO_FILL_NODATA:
        return

    print("\n开始分块补漏 NoData...")

    with rasterio.open(MOSAIC_OUT) as src:
        profile = src.profile.copy()
        width = src.width
        height = src.height

        if os.path.exists(FILLED_OUT):
            os.remove(FILLED_OUT)

        with rasterio.open(FILLED_OUT, "w", **profile) as dst:
            total_windows = count_windows(width, height, BLOCK_SIZE)

            for win in tqdm(
                generate_windows(width, height, BLOCK_SIZE),
                total=total_windows,
                desc="Fill NoData",
                unit="block"
            ):
                expanded_win = expand_window(win, width, height, FILL_HALO)

                data = src.read(1, window=expanded_win).astype(np.float32)

                valid_mask = np.isfinite(data) & (data != OUT_NODATA)

                if valid_mask.all():
                    center = crop_center_from_expanded(data, win, expanded_win)
                    dst.write(center.astype(OUT_DTYPE), 1, window=win)
                    continue

                if not valid_mask.any():
                    center = crop_center_from_expanded(data, win, expanded_win)
                    dst.write(center.astype(OUT_DTYPE), 1, window=win)
                    continue

                filled = fillnodata(
                    data,
                    mask=valid_mask.astype("uint8"),
                    max_search_distance=FILL_MAX_SEARCH_DISTANCE,
                    smoothing_iterations=0
                ).astype(np.float32)

                filled[~np.isfinite(filled)] = OUT_NODATA

                center = crop_center_from_expanded(filled, win, expanded_win)

                dst.write(center.astype(OUT_DTYPE), 1, window=win)

    print("\n补漏完成：")
    print(FILLED_OUT)


# =========================================================
# 6. 主程序
# =========================================================

def main():
    mosaic_dem()

    if DO_FILL_NODATA:
        fill_nodata_by_blocks()
        build_overviews(FILLED_OUT)
        print("\n最终推荐使用这个文件：")
        print(FILLED_OUT)
    else:
        build_overviews(MOSAIC_OUT)
        print("\n最终输出文件：")
        print(MOSAIC_OUT)

    print("\n全部完成。")


if __name__ == "__main__":
    main()