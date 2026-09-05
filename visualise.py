import rasterio
import matplotlib.pyplot as plt
import numpy as np

import sys

# Check if the master script passed the filename automatically
if len(sys.argv) > 1:
    img = sys.argv[1]
else:
    # Fallback to manual input if you run this script by itself
    img = input("Enter image file name: ")

RGB_PATH= f"dataset/{img}" 
# ... (keep the rest of your configuration paths the same)
ABSOLUTE_DSM_PATH = 'calibrated_absolute_dsm.tif'
OUTPUT_IMAGE = 'absolute_dsm_visualization.png'

def visualize_absolute_dsm():
    print(f"Loading {ABSOLUTE_DSM_PATH}...")
    
    # 1. Read the calibrated absolute DSM
    with rasterio.open(ABSOLUTE_DSM_PATH) as src:
        abs_dsm = src.read(1)
        # Optional: Mask out 'NoData' values to keep the plot clean
        if src.nodata is not None:
            abs_dsm = np.ma.masked_where(abs_dsm == src.nodata, abs_dsm)

    # 2. Read the original RGB GeoTIFF
    print(f"Loading {RGB_PATH}...")
    with rasterio.open(RGB_PATH) as src:
        # Rasterio reads bands as (channels, height, width). 
        # Matplotlib expects (height, width, channels), so we stack them.
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        rgb_image = np.dstack((r, g, b))
        
        # Satellite GeoTIFFs are often 16-bit. This normalizes them to standard 8-bit for viewing.
        if rgb_image.dtype == np.uint16 or rgb_image.max() > 255:
            rgb_image = (rgb_image / (rgb_image.max() / 255.0)).astype(np.uint8)

    # 3. Plotting Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot RGB
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Satellite RGB")
    axes[0].axis('off')
    
    # Plot Absolute DSM
    # We switch the colormap to 'terrain' as it is the geospatial standard for elevation
    im = axes[1].imshow(abs_dsm, cmap='terrain')
    axes[1].set_title("Calibrated Absolute DSM (Metric)")
    axes[1].axis('off')
    
    # Add the metric colorbar
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (Meters above sea level)')
    
    # 4. Save Output
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Saved visualization to '{OUTPUT_IMAGE}'.")
    plt.show()

if __name__ == '__main__':
    visualize_absolute_dsm()
