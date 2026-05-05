# -*- coding: utf-8 -*-

from pathlib import Path
import geopandas as gpd
import rasterio
from shapely.geometry import box
import fiona

DEM_PATH = Path(r"D:\python\DEM_work\data\Features_resolution_30_m\DEM.tif")

WATERWAYS_PATH = Path(
    r"D:\python\DEM_work\data\Co_work\projected_vectors\NY_NJ_waterways_to_DEM_CRS.gpkg"
)

print("=" * 80)
print("[GPKG 图层]")
print("=" * 80)
print(fiona.listlayers(WATERWAYS_PATH))

with rasterio.open(DEM_PATH) as dem:
    print("\n" + "=" * 80)
    print("[DEM]")
    print("=" * 80)
    print("CRS:", dem.crs)
    print("Bounds:", dem.bounds)

    dem_poly = box(*dem.bounds)
    dem_crs = dem.crs

gdf = gpd.read_file(WATERWAYS_PATH)

print("\n" + "=" * 80)
print("[原始水系矢量]")
print("=" * 80)
print("要素数:", len(gdf))
print("CRS:", gdf.crs)
print("Bounds:", gdf.total_bounds)

if gdf.crs is None:
    raise ValueError("水系矢量没有 CRS。")

# 强制转到 EPSG:5070，而不是转到 DEM 原来的 LOCAL_CS
gdf_5070 = gdf.to_crs("EPSG:5070")

print("\n" + "=" * 80)
print("[水系转 EPSG:5070 后]")
print("=" * 80)
print("CRS:", gdf_5070.crs)
print("Bounds:", gdf_5070.total_bounds)

intersects = gdf_5070.intersects(dem_poly)
cnt = int(intersects.sum())

print("\n" + "=" * 80)
print("[重叠检查]")
print("=" * 80)
print("与 DEM 范围相交的水系要素数:", cnt)

if cnt == 0:
    print("\n结论：水系矢量和 DEM 没有重叠，说明矢量 CRS 或 DEM CRS 仍有问题。")
else:
    print("\n结论：水系矢量和 DEM 有重叠，可以重新栅格化。")