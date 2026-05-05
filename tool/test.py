import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

mask_path = Path(r"D:\python\DEM_work\data\Features_resolution_30_m\DTRiver_mask.tif")

with rasterio.open(mask_path) as src:
    print("CRS:", src.crs)
    print("Size:", src.width, src.height)
    print("NoData:", src.nodata)
    print("Bounds:", src.bounds)

    total_feature_pixels = 0
    total_pixels = src.width * src.height

    for _, win in tqdm(list(src.block_windows(1)), desc="Check mask"):
        arr = src.read(1, window=win)
        total_feature_pixels += np.count_nonzero(arr == 1)

print("河流像元数:", total_feature_pixels)
print("总像元数:", total_pixels)
print("河流像元比例:", total_feature_pixels / total_pixels)