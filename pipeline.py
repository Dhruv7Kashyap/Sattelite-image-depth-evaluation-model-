import subprocess
import sys
import os

def run_pipeline():
    print("========================================")
    print("   END-TO-END 3D ELEVATION PIPELINE   ")
    print("========================================")
    
    # Ask for the filename ONE time
    img_name = input("\nEnter the image file name (e.g., JAX_Tile_034_RGB.tif): ")
    
    # Verify the file actually exists before wasting time running the models
    if not os.path.exists(f"dataset/{img_name}"):
        print(f"\n[ERROR] Could not find dataset/{img_name}. Check the spelling!")
        return

    try:
        print("\n>>> [1/3] RUNNING INFERENCE (Depth Anything V2)...")
        # sys.executable ensures it uses your active .venv Python
        subprocess.run([sys.executable, "run_inference.py", img_name], check=True)
        
        print("\n>>> [2/3] RUNNING CALIBRATION (Copernicus DEM Fusion)...")
        subprocess.run([sys.executable, "calibrate.py", img_name], check=True)
        
        print("\n>>> [3/3] RUNNING VISUALIZATION...")
        # Make sure the filename here matches what you actually named your script!
        subprocess.run([sys.executable, "visualise.py", img_name], check=True)
        
        print("\n========================================")
        print(" PIPELINE COMPLETE! CHECK YOUR OUTPUTS. ")
        print("========================================")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline crashed during execution. Error code: {e.returncode}")
        print("Check the logs above to see which script failed.")

if __name__ == '__main__':
    run_pipeline()