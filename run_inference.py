import cv2
import torch
import numpy as np
import tifffile
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'checkpoints/depth_anything_v2_vitl.pth'
import sys

# Check if the master script passed the filename automatically
if len(sys.argv) > 1:
    img = sys.argv[1]
else:
    # Fallback to manual input if you run this script by itself
    img = input("Enter image file name: ")

IMAGE_PATH = f"dataset/{img}" 
# ... (keep the rest of your configuration paths the same)
OUTPUT_TIFF = 'relative_dsm.tif'

def main():
    print(f"Initializing Depth Anything V2 (Large) on {DEVICE}...")
    
    model = DepthAnythingV2(
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024]
    )
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    print(f"Reading satellite image from {IMAGE_PATH}...")
    raw_img = cv2.imread(IMAGE_PATH)
    
    if raw_img is None:
        raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}")

    print("Extracting relative depth map...")
    with torch.no_grad():
        depth_map = model.infer_image(raw_img)
        
    print(f"Saving 32-bit floating-point depth map to {OUTPUT_TIFF}...")
    tifffile.imwrite(OUTPUT_TIFF, depth_map)
    print("Done! This file is ready for Phase 2 scaling.")

if __name__ == '__main__':
    main()
