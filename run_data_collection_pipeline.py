import os
import subprocess
import sys
import time
from pathlib import Path
import yaml

work_dir = Path(__file__).parent.absolute()
print(work_dir)
sys.path.append(str(work_dir))
os.chdir(work_dir)
print(f"Working directory: {work_dir}")

def run_pipeline(config_file):
    current_dir = Path(__file__).parent.absolute()
    pipeline_dir = os.path.join(current_dir, "vitamin_b_data_collection_pipeline")
    print("Please make sure the param use_gelsight is True or False in the config file")
    input("Press Enter to continue...")

    pipeline_steps = [
        "01_crop_img.py",
        "04_get_aruco_pos.py",
        "05_get_width.py",
        "07_generate_dataset_plan.py",
        "08_generate_replay_buffer.py",
    ]

    for step in pipeline_steps:
        step_path = os.path.join(pipeline_dir, step)
        print(f"\n{'='*50}")
        print(f"Running step: {step}")
        print(f"{'='*50}\n")
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(work_dir) + os.pathsep + env.get('PYTHONPATH', '')
            
            # Add --force-regenerate-csv for step 07 to ensure coordinate transformation is applied
            cmd = [sys.executable, str(step_path), "--cfg", config_file]
            
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                check=True,
                stdout=None,
                stderr=None,
                text=True,  
                env=env
            )
                
        except subprocess.CalledProcessError as e:
            print(f"Error running {step}:")
            print(f"Exit code: {e.returncode}")
            print("\nPipeline failed. Stopping execution.")
            return False
            
        print(f"\nCompleted step: {step}\n")
        
        # add delay after 01 step
        if step == "01_bak_raw_data.py":
            print("Wait for 5 seconds to ensure file system sync...")
            time.sleep(5)
            print("Delay completed, continue to next step\n")
    
    print("\nPipeline completed successfully!")
    return True

if __name__ == "__main__":
    config_file = "./config/VB_task_config.yaml"
    run_pipeline(config_file) 