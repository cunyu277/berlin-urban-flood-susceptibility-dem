# -*- coding: utf-8 -*-
"""
将 DEM 转为 WhiteboxTools 可读取的 GeoTIFF。

原因：
WhiteboxTools 不支持带 PREDICTOR=3 的浮点型 GeoTIFF。
因此需要生成一个无 floating-point predictor 的中间 DEM。
"""

from pathlib import Path
import rasterio
from tqdm import tqdm


DEM_PATH = Path(r"D:\python\DEM_work\data\Features_resolution_30_m\DEM.tif")

OUT_PATH = Path(
    r"D:\python\DEM_work\data\Features_resolution_30_m\hydrology\DEM_wbt_input.tif"
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    with rasterio.open(DEM_PATH) as src:
        print("=" * 80)
        print("[输入 DEM]")
        print(f"路径: {DEM_PATH}")
        print(f"CRS: {src.crs}")
        print(f"Size: {src.width} x {src.height}")
        print(f"Dtype: {src.dtypes[0]}")
        print(f"NoData: {src.nodata}")
        print(f"Compression: {src.compression}")
        print("=" * 80)

        profile = src.profile.copy()

        # 关键：不要使用 predictor=3
        # 为了 WhiteboxTools 最大兼容性，这里直接不压缩
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            compress=None,
            predictor=None,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )

        # 删除 profile 中可能残留的 predictor / compress
        profile.pop("predictor", None)
        profile.pop("compress", None)

        print("[输出 DEM]")
        print(f"路径: {OUT_PATH}")
        print("压缩: 无压缩")
        print("PREDICTOR: 无")
        print("=" * 80)

        with rasterio.open(OUT_PATH, "w", **profile) as dst:
            for _, window in tqdm(
                list(src.block_windows(1)),
                desc="Copy DEM for Whitebox",
                unit="block"
            ):
                arr = src.read(1, window=window, out_dtype="float32")
                dst.write(arr, 1, window=window)

    print("\n完成。WhiteboxTools 输入 DEM:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()