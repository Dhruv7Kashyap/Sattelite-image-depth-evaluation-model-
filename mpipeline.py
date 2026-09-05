import os
import sys
import cv2
import json
import math
import torch
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from dem_stitcher import stitch_dem
from scipy.ndimage import gaussian_filter
from depth_anything_v2.dpt import DepthAnythingV2

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'checkpoints/depth_anything_v2_vitl.pth'
OUTPUT_OBJ = 'target/terrain_mesh.obj'
OUTPUT_HEIGHTMAP = 'target/heightmap.png'
OUTPUT_MANIFEST = 'target/manifest.json'
RELATIVE_FALLBACK_HEIGHT = 50.0  # Scale used if no GPS data is present

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculates real-world meters between two GPS coordinates."""
    R = 6371000  # Radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def export_obj_mesh(dsm_array, output_path, subsample=2):
    print(f"Exporting 3D OBJ Mesh to {output_path}...")
    dsm = dsm_array[::subsample, ::subsample]
    h, w = dsm.shape
    
    with open(output_path, 'w') as f:
        f.write("o Terrain\n")
        for y in range(h):
            for x in range(w):
                z = dsm[y, x] 
                pos_x = x - (w / 2)
                pos_y = z  # Unity uses Y as the UP axis
                pos_z = y - (h / 2)
                f.write(f"v {pos_x} {pos_y} {pos_z}\n")
                
                u = x / (w - 1)
                v = 1.0 - (y / (h - 1))
                f.write(f"vt {u} {v}\n")
                
        for y in range(h - 1):
            for x in range(w - 1):
                i = y * w + x + 1 
                f.write(f"f {i}/{i} {i+w}/{i+w} {i+1}/{i+1}\n")
                f.write(f"f {i+1}/{i+1} {i+w}/{i+w} {i+w+1}/{i+w+1}\n")

def run_pipeline():
    print("=== DeepWizard: ML to Unity Master Pipeline ===")
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter image filepath: ").strip().strip("'\"")
        
    img_path="dataset/" + img_path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    ext = os.path.splitext(img_path)[1].lower()
    is_georeferenced = ext in ['.tif', '.tiff']
    
    # --- 1. Load Image for AI (Needs BGR format) ---
    print("\n[1/4] Loading Image & Extracting AI Depth...")
    if is_georeferenced:
        with rasterio.open(img_path) as src:
            bounds = src.bounds
            crs = src.crs
            # Check if TIFF actually has spatial data
            if crs is None:
                print("Warning: TIFF lacks CRS metadata. Falling back to Relative Mode.")
                is_georeferenced = False
            r, g, b = src.read(1), src.read(2), src.read(3)
            rgb = np.dstack((r, g, b))
            bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = cv2.imread(img_path)

    # --- 2. Run Foundation Model ---
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    with torch.no_grad():
        raw_depth = model.infer_image(bgr_image)
    h, w = raw_depth.shape

    # --- 3. Process Elevation (The Router) ---
    manifest = {
        "id": os.path.splitext(os.path.basename(img_path))[0],
        "source_image": os.path.basename(img_path),
        "heightmap": OUTPUT_HEIGHTMAP,
        "heightmap_encoding": "grayscale16",
        "width_px": int(w),
        "height_px": int(h)
    }

    if is_georeferenced:
        print("\n[2/4] GEOTIFF DETECTED: Running Absolute Calibration Pipeline...")
        # 1. Fetch SRTM Data
        gps_bounds = transform_bounds(crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)
        print("Downloading Copernicus GLO-30 Elevation data...")
        dem_array, _ = stitch_dem(list(gps_bounds), dem_name='glo_30', dst_ellipsoidal_height=False)
        srtm_aligned = cv2.resize(dem_array, (w, h), interpolation=cv2.INTER_CUBIC)

        # 2. Robust Scaling
        rel_min, rel_max = np.percentile(raw_depth, 2), np.percentile(raw_depth, 98)
        dem_min, dem_max = np.percentile(srtm_aligned, 2), np.percentile(srtm_aligned, 98)
        scale_factor = (dem_max - dem_min) / (rel_max - rel_min + 1e-6)
        rel_scaled = dem_min + (raw_depth - rel_min) * scale_factor

        # 3. High-Pass Filter (Remove AI Tilt)
        sigma = max(h, w) / 30.0
        rel_blurred = gaussian_filter(rel_scaled, sigma=sigma)
        high_freq = rel_scaled - rel_blurred

        # 4. Fuse Absolute Terrain
        final_dsm = srtm_aligned + high_freq
        final_dsm = np.clip(final_dsm, dem_min - 15, dem_max + 50)
        
        # 5. Populate Absolute Manifest Data
        true_min, true_max = float(final_dsm.min()), float(final_dsm.max())
        manifest["elevation_range_m"] = {"min": round(true_min, 2), "max": round(true_max, 2)}
        manifest["reference_available"] = True
        
        # Calculate Real World physical size using Haversine formula
        min_lon, min_lat, max_lon, max_lat = gps_bounds
        width_m = haversine_distance(min_lon, min_lat, max_lon, min_lat)
        height_m = haversine_distance(min_lon, min_lat, min_lon, max_lat)
        manifest["real_world_size_m"] = {"width": round(width_m, 2), "height": round(height_m, 2)}
        
        # Mesh zeroing: Subtract min so the OBJ mesh sits on Unity's floor (Y=0)
        mesh_dsm = final_dsm - true_min 

    else:
        print("\n[2/4] PNG/JPG DETECTED: Running Relative Flattening Pipeline...")
        # 1. High-Pass Filter (Remove AI Tilt)
        sigma = max(h, w) / 30.0
        tilt_map = gaussian_filter(raw_depth, sigma=sigma)
        flattened_depth = raw_depth - tilt_map
        
        # 2. Normalize and Scale to Fallback Height
        flattened_depth = np.clip(flattened_depth, 0, None)
        d_min, d_max = flattened_depth.min(), flattened_depth.max()
        if d_max - d_min > 0:
            norm_depth = (flattened_depth - d_min) / (d_max - d_min)
            
        final_dsm = norm_depth * RELATIVE_FALLBACK_HEIGHT
        
        # 3. Populate Relative Manifest Data
        manifest["elevation_range_m"] = {"min": 0.0, "max": RELATIVE_FALLBACK_HEIGHT}
        manifest["reference_available"] = False
        mesh_dsm = final_dsm

    # --- 4. Exporting Data for Unity ---
    print("\n[3/4] Exporting Data Formats...")
    
    # A. 16-bit Heightmap PNG
    # Formula: (val - min) / (max - min) * 65535
    dsm_min, dsm_max = final_dsm.min(), final_dsm.max()
    normalized_for_png = (final_dsm - dsm_min) / (dsm_max - dsm_min + 1e-6)
    heightmap_16bit = (normalized_for_png * 65535).astype(np.uint16)
    cv2.imwrite(OUTPUT_HEIGHTMAP, heightmap_16bit)
    
    # B. Generate JSON Manifest
    with open(OUTPUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=4)
        
    # C. Export OBJ Mesh
    export_obj_mesh(mesh_dsm, OUTPUT_OBJ)

    print("\n[4/4] Pipeline Complete!")
    print(f" 1. {OUTPUT_OBJ}       (The 3D Geometry)")
    print(f" 2. {OUTPUT_HEIGHTMAP} (The 16-bit Grayscale Map)")
    print(f" 3. {OUTPUT_MANIFEST}  (The Metadata Schema)")

if __name__ == '__main__':
    run_pipeline()
