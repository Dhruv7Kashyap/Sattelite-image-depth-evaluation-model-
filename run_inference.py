import os
import sys
import cv2
import torch
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from depth_anything_v2.dpt import DepthAnythingV2

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'checkpoints/depth_anything_v2_vitl.pth'
OUTPUT_OBJ = 'terrain_mesh.obj'
OUTPUT_TIFF = 'relative_dsm.tif' # Kept purely for mathematical evaluation later

def resolve_path(user_input, default_dir="dataset"):
    if os.path.isabs(user_input) or user_input.startswith(default_dir):
        return user_input
    return os.path.join(default_dir, user_input)

def export_obj_mesh(dsm_array, output_path, subsample=2, mesh_height_scale=50.0):
    print(f"Exporting 3D OBJ Mesh to {output_path}...")
    
    # Subsampling reduces polygon count so Unity doesn't lag
    dsm = dsm_array[::subsample, ::subsample]
    h, w = dsm.shape
    
    with open(output_path, 'w') as f:
        f.write("o Terrain\n")
        
        # 1. Write Vertices (X, Y, Z) and UVs (U, V)
        print("Calculating vertices...")
        for y in range(h):
            for x in range(w):
                # Apply height scale to make buildings physically tall in Unity
                z = dsm[y, x] * mesh_height_scale 
                
                # Center mesh at origin (0,0,0)
                pos_x = x - (w / 2)
                pos_y = z  # Unity uses Y as the UP axis
                pos_z = y - (h / 2)
                f.write(f"v {pos_x} {pos_y} {pos_z}\n")
                
                # UV mapping for RGB texture projection
                u = x / (w - 1)
                v = 1.0 - (y / (h - 1))
                f.write(f"vt {u} {v}\n")
                
        # 2. Write Triangles (Faces)
        print("Generating mesh faces...")
        for y in range(h - 1):
            for x in range(w - 1):
                i = y * w + x + 1 # .obj files use 1-based indexing
                f.write(f"f {i}/{i} {i+w}/{i+w} {i+1}/{i+1}\n")
                f.write(f"f {i+1}/{i+1} {i+w}/{i+w} {i+w+1}/{i+w+1}\n")
                
    print(f"✅ Saved 3D Mesh: {output_path}")

def extract_relative_dsm():
    print("=== Phase 1: Direct Image-to-Mesh Pipeline ===")
    
    if len(sys.argv) > 1:
        raw_img = sys.argv[1]
    else:
        print("(Assuming files are in the 'dataset/' directory)")
        raw_img = input("Enter satellite image filename (e.g., sample.png): ").strip().strip("'\"")
        
    img_path = resolve_path(raw_img)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Could not find image at {img_path}")

    print(f"\nInitializing Depth Anything V2 (Large) on {DEVICE}...")
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    print(f"Reading satellite imagery from {img_path}...")
    raw_image = cv2.imread(img_path)
    
    print("Extracting structural depth map...")
    with torch.no_grad():
        raw_depth = model.infer_image(raw_image)
        
    print("Applying High-Pass Filter to remove AI perspective tilt...")
    h, w = raw_depth.shape
    sigma = max(h, w) / 30.0
    tilt_map = gaussian_filter(raw_depth, sigma=sigma)
    flattened_depth = raw_depth - tilt_map
    
    flattened_depth = np.clip(flattened_depth, 0, None)
    d_min, d_max = flattened_depth.min(), flattened_depth.max()
    if d_max - d_min > 0:
        flattened_depth = (flattened_depth - d_min) / (d_max - d_min)
        
    # Save TIFF for mathematical evaluation
    tifffile.imwrite(OUTPUT_TIFF, flattened_depth.astype(np.float32))
    
    # Export directly to Unity OBJ
    export_obj_mesh(flattened_depth, OUTPUT_OBJ)
    
    print("\nPipeline Complete!")
    print(f"Hand the generated '{OUTPUT_OBJ}' and the original '{raw_img}' directly to the rendering team.")

if __name__ == '__main__':
    extract_relative_dsm()