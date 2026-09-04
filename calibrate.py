import numpy as np
import cv2
import rasterio
from rasterio.warp import transform_bounds
from dem_stitcher import stitch_dem
from scipy.ndimage import gaussian_filter
import sys

# Check if the master script passed the filename automatically
if len(sys.argv) > 1:
    img = sys.argv[1]
else:
    # Fallback to manual input if you run this script by itself
    img = input("Enter image file name: ")

ORIGINAL_GEOTIFF = f"dataset/{img}" 
RELATIVE_DSM_PATH = 'relative_dsm.tif' 
FINAL_OUTPUT_PATH = 'calibrated_absolute_dsm.tif'

def calibrate_and_fuse():
    print(f"Reading metadata from {ORIGINAL_GEOTIFF}...")
    
    # 1. Extract and convert bounds to GPS Latitude / Longitude
    with rasterio.open(ORIGINAL_GEOTIFF) as src:
        meta = src.meta.copy()
        h, w = src.shape
        gps_bounds = transform_bounds(src.crs, 'EPSG:4326', src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

    # 2. Download Global 30m Copernicus DEM
    print("Downloading Copernicus GLO-30 Elevation data...")
    dem_array, dem_profile = stitch_dem(list(gps_bounds), dem_name='glo_30', dst_ellipsoidal_height=False)

    # 3. Load your model's relative depth map
    with rasterio.open(RELATIVE_DSM_PATH) as rel_src:
        rel_depth = rel_src.read(1)

    print("Aligning spatial grids...")
    srtm_aligned = cv2.resize(dem_array, (w, h), interpolation=cv2.INTER_CUBIC)

    # 4. Phase 1: Robust Scaling
    print("Normalizing relative depth to metric scale...")
    rel_min, rel_max = np.percentile(rel_depth, 2), np.percentile(rel_depth, 98)
    dem_min, dem_max = np.percentile(srtm_aligned, 2), np.percentile(srtm_aligned, 98)
    
    scale_factor = (dem_max - dem_min) / (rel_max - rel_min + 1e-6)
    rel_scaled = dem_min + (rel_depth - rel_min) * scale_factor

    # 5. Phase 2: Spatial Frequency Extraction
    # A large sigma isolates the macro-tilt from the micro-buildings
    sigma = max(h, w) / 30.0  
    print(f"Applying High-Pass Frequency Filter (sigma={sigma:.1f})...")
    
    rel_blurred = gaussian_filter(rel_scaled, sigma=sigma)
    
    # Subtracting the blur isolates the sharp buildings and mathematically deletes the AI's fake tilt
    high_freq_details = rel_scaled - rel_blurred

    # 6. Phase 3: Final Fusion
    print("Fusing high-frequency structures with true Copernicus macro-terrain...")
    absolute_dsm = srtm_aligned + high_freq_details

    # Clean up minor statistical anomalies
    absolute_dsm = np.clip(absolute_dsm, dem_min - 15, dem_max + 50)
    absolute_dsm = absolute_dsm.astype(np.float32)

    # 7. Save the calibrated GeoTIFF
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(FINAL_OUTPUT_PATH, 'w', **meta) as dst:
        dst.write(absolute_dsm, 1)

    print(f"Success! Mathematically sound metric DSM saved to {FINAL_OUTPUT_PATH}")

if __name__ == '__main__':
    calibrate_and_fuse()