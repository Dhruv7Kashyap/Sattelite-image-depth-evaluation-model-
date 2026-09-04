# Depth Anything V2: Monocular Depth Estimation & 3D Elevation Pipeline

This repository contains the implementation of **Depth Anything V2** integrated with an **End-to-End 3D Elevation & Metric Digital Surface Model (DSM) Pipeline**.

It converts 2D optical images and satellite GeoTIFFs into metric scale 3D elevation models using monocular depth estimation fused with Copernicus Global 30m DEM elevation data.

---

## 🌟 Features

- **State-of-the-Art Monocular Depth**: Leverages Depth Anything V2 (DINOv2 backbone) for fine-grained relative depth prediction.
- **Copernicus DEM Fusion**: Automatically fetches regional Copernicus GLO-30 elevation data to calibrate relative depth maps into true metric elevations (meters above sea level).
- **High-Pass Spatial Frequency Filtering**: Isolates high-frequency micro-structures (buildings, trees) while removing artificial macro-tilts predicted by monocular models.
- **End-to-End Automated Pipeline**: Interactive CLI runner executing depth extraction, DEM calibration, and visualization in sequence.
- **Error Evaluation Tool**: Calculates Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) against ground truth DSMs and generates error heatmaps.
- **Gradio Web Interface**: Interactive web application featuring side-by-side image sliders and depth map downloads.
- **Metric Depth Subpackage**: Dedicated tools in `metric_depth/` for fine-tuning on custom datasets (KITTI, HyperSim, Virtual KITTI 2) and generating 3D point clouds (`.ply`).

---

## 📁 Repository Structure

```
Depth-Anything-V2/
├── app.py                      # Interactive Gradio Web Application
├── pipeline.py                 # Master End-to-End 3D Elevation Pipeline runner
├── run_inference.py            # Step 1: Relative Depth Extraction (Depth Anything V2)
├── calibrate.py                # Step 2: Copernicus GLO-30 DEM Calibration & High-Pass Fusion
├── visualise.py                # Step 3: Elevation Map & RGB Side-by-Side Visualization
├── evaluate.py                 # Step 4: DSM Accuracy Evaluation (MAE, RMSE & Heatmaps)
├── run.py                      # Standard Depth Anything V2 batch inference script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusion rules
│
├── depth_anything_v2/          # Core neural network package (DINOv2, DPT head)
│   ├── dinov2.py
│   ├── dpt.py
│   ├── dinov2_layers/
│   └── util/
│
├── metric_depth/               # Fine-tuning, metric depth models & 3D point cloud generation
│   ├── train.py
│   ├── depth_to_pointcloud.py
│   ├── dataset/
│   └── util/
│
├── assets/                     # Demo images, teaser graphics, and sample videos
├── checkpoints/                # Folder for pre-trained model weights (git-ignored)
└── dataset/                    # Local satellite input images & GeoTIFFs (git-ignored)
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Depth-Anything-V2
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📥 Download Pre-trained Model Checkpoints

Download the desired Depth Anything V2 model checkpoint(s) and place them in the `checkpoints/` directory:

| Model | Params | Checkpoint Link | Target Path |
|---|---|---|---|
| **Depth Anything V2 Small** | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth) | `checkpoints/depth_anything_v2_vits.pth` |
| **Depth Anything V2 Base** | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth) | `checkpoints/depth_anything_v2_vitb.pth` |
| **Depth Anything V2 Large** | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth) | `checkpoints/depth_anything_v2_vitl.pth` |

---

## 🚀 Usage Guide

### 1. End-to-End 3D Elevation Pipeline (CLI)
Run the master pipeline script to process a satellite image tile:
```bash
python pipeline.py
```
**Inputs:** Select an optical satellite image located in the `dataset/` directory (e.g. GeoTIFF tile).  
**Outputs Generated:**
- `relative_dsm.tif`: Raw relative depth map (32-bit floating point TIFF).
- `calibrated_absolute_dsm.tif`: Calibrated absolute metric elevation map (meters).
- `absolute_dsm_visualization.png`: Dual-panel RGB vs. Terrain Elevation plot.

---

### 2. Running Individual Pipeline Steps

#### Step 1: Extract Relative Depth
```bash
python run_inference.py <image_name>
```

#### Step 2: Calibrate Metric DSM
```bash
python calibrate.py <image_name>
```

#### Step 3: Visualize Results
```bash
python visualise.py <image_name>
```

#### Step 4: Evaluate Metric DSM against Ground Truth
```bash
python evaluate.py
```

---

### 3. Gradio Web Application
Launch the web interface for interactive monocular depth estimation:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860` to view the UI.

---

### 4. Metric Depth & Point Cloud Generation (`metric_depth/`)
For point cloud generation from metric depth predictions:
```bash
cd metric_depth
python depth_to_pointcloud.py --drive-path <path_to_images>
```

---

## 📄 License & Citation

If you find Depth Anything V2 useful in your research or application, please cite the original paper:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:2406.09414},
  year={2024}
}
```
