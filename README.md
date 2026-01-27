# ViTaMIn-B Data Collection Pipeline (Standalone)

This is a **standalone, self-contained** version of the ViTaMIn-B manipulation data collection and processing pipeline. It has no external dependencies on other directories or packages outside this folder.

**Supported Task Types:**
- **Bimanual**: Two-handed manipulation with dual grippers and Quest controllers
- **Mono-Manual (Single-Hand)**: Single-handed manipulation with one gripper and Quest controller


## Prerequisites

- Python 3.10 or 3.11 (recommended)
- Virtual environment (recommended)

## Quick Start

### Step 1: Setup Environment

1. **Navigate to the pipeline directory**:
   ```bash
   cd /path/to/data_collection
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Verify Installation

Run the validation script to ensure everything is set up correctly:

```bash
python test_standalone.py
```

You should see:
```
✓ ALL TESTS PASSED - Pipeline is ready to use!
```

### Step 3: Prepare Your Data

Place your raw demonstration data in the following structure:

#### Bimanual Mode (Two Hands)
```
data_collection/data/<task_name>/demos/
├── demo_<name>_<timestamp_1>/
│   ├── left_hand_img/          # Left hand camera images (JPEG sequence)
│   ├── right_hand_img/         # Right hand camera images (JPEG sequence)
│   └── all_trajectory/
│       └── quest_poses_<timestamp>.json
├── demo_<name>_<timestamp_2>/
│   └── ...
└── ...
```

#### Mono-Manual Mode (Single Hand)
```
data_collection/data/<task_name>/demos/
├── demo_<name>_<timestamp_1>/
│   ├── left_hand_img/          # OR right_hand_img/ (depending on single_hand_side config)
│   └── all_trajectory/
│       └── quest_poses_<timestamp>.json
├── demo_<name>_<timestamp_2>/
│   └── ...
└── ...
```

Or use a symlink to existing data:
```bash
ln -s /path/to/your/existing/data ./data
```

### Step 4: Configure the Pipeline

Edit `config/VB_task_config.yaml` based on your task type:

#### For Bimanual Tasks
```yaml
task:
  name: "_0118_bi_pick_and_place"
  type: bimanual  # Two-handed mode

recorder:
  camera_paths:
    left_hand: "/dev/video0"
    right_hand: "/dev/video2"
```

#### For Mono-Manual Tasks
```yaml
task:
  name: "_0118_mono_pick"
  type: single          # Single-handed mode
  single_hand_side: left  # or "right"

recorder:
  camera_paths:
    left_hand: "/dev/video0"   # Used if single_hand_side: left
    right_hand: "/dev/video2"  # Used if single_hand_side: right
```

See the Configuration section below for more details.

### Step 5: Run the Pipeline

```bash
python run_data_collection_pipeline.py
```

That's it! The pipeline will process all steps automatically.

## Input Data Structure

The pipeline expects raw demonstration data organized as follows:

### Data Collection with 00_get_data.py

The pipeline includes a data recording script `vitamin_b_data_collection_pipeline/00_get_data.py` that captures synchronized camera images and Quest controller poses in real-time.

**Recording Process:**
1. Configure your task type (`bimanual` or `single`) and cameras in `config/VB_task_config.yaml`
2. Run `python vitamin_b_data_collection_pipeline/00_get_data.py --cfg config/VB_task_config.yaml`
3. The script automatically:
   - Records from configured cameras (1 for single-hand, 2 for bimanual)
   - Captures Quest controller poses
   - Saves data in the correct folder structure

**Output Structure:**

#### For Bimanual Tasks
```
data/
└── <task_name>/                          # e.g., _0118_bi_pick_and_place
    └── demos/
        ├── demo_<timestamp_1>/
        │   ├── left_hand_img/            # Left camera JPEG sequence
        │   │   ├── frame_000000.jpg
        │   │   ├── frame_000001.jpg
        │   │   └── ...
        │   ├── right_hand_img/           # Right camera JPEG sequence
        │   │   ├── frame_000000.jpg
        │   │   └── ...
        │   └── all_trajectory/
        │       └── quest_poses_<timestamp>.json  # VR controller poses
        ├── demo_<timestamp_2>/
        │   └── ...
        └── ...
```

#### For Mono-Manual (Single-Hand) Tasks
```
data/
└── <task_name>/
    └── demos/
        └── demo_<timestamp>/
            ├── left_hand_img/            # OR right_hand_img/ (based on config)
            │   ├── frame_000000.jpg
            │   ├── frame_000001.jpg
            │   └── ...
            └── all_trajectory/
                └── quest_poses_<timestamp>.json
```

**Image Format:**
- **Format**: JPEG (`.jpg`)
- **Resolution**: 3840×800 (raw, before cropping) or as configured
- **Frame naming**: Sequential `frame_XXXXXX.jpg`
- **Content**: Raw fisheye camera view containing visual + tactile views (split in step 01)

**Quest Pose Data Format (`quest_poses_*.json`):**
```json
{
  "timestamp_1": {
    "left_controller": [x, y, z, qx, qy, qz, qw],
    "right_controller": [x, y, z, qx, qy, qz, qw],
    "left_gripper_width": 0.05,
    "right_gripper_width": 0.05
  },
  "timestamp_2": { ... }
}
```

### Pipeline Processing Modes

The pipeline automatically adapts to your task type:

**Bimanual Mode (`task.type: bimanual`)**:
- Processes **both** `left_hand_img/` and `right_hand_img/` folders
- Detects ArUco markers on **both grippers**
- Calculates width for **both grippers**
- Generates zarr with **robot0** and **robot1** arrays

**Mono-Manual Mode (`task.type: single`)**:
- Processes **only** the specified hand's image folder (set by `task.single_hand_side`)
- Detects ArUco markers on **one gripper**
- Calculates width for **one gripper**
- Generates zarr with **robot0 arrays only**

The mode is detected automatically in steps 07 and 08 by checking which image folders exist in each demo.

## Usage Guide

### Automatic Pipeline Execution (Recommended)

The `run_data_collection_pipeline.py` script automates the entire data processing workflow, executing all steps in sequence without manual intervention.

#### Running the Pipeline

```bash
# From the data_collection directory
python run_data_collection_pipeline.py
```

#### What Happens When You Run It

1. **Initial Prompt**: The script will ask you to confirm the `use_gelsight` parameter is set correctly in the config:
   ```
   Please make sure the param use_gelsight is True or False in the config file
   Press Enter to continue...
   ```
   Press **Enter** to proceed.

2. **Automatic Execution**: The pipeline will automatically run these steps in order:

   **Step 1: Image Cropping** (`01_crop_img.py`)
   ```
   ==================================================
   Running step: 01_crop_img.py
   ==================================================
   ```
   - Splits 3840×800 raw images into visual (center) and tactile (left/right) views
   - Processes images for configured hands (both for bimanual, one for mono-manual)
   - Progress: Shows tasks completed (e.g., "10/10 tasks")

   **Step 2: ArUco Detection** (`04_get_aruco_pos.py`)
   ```
   ==================================================
   Running step: 04_get_aruco_pos.py
   ==================================================
   ```
   - Detects ArUco markers on gripper(s) for pose estimation
   - Processes configured hands (both for bimanual, one for mono-manual)
   - Creates `tag_detection_*.pkl` files
   - Shows detection success rate

   **Step 3: Gripper Width Calculation** (`05_get_width.py`)
   ```
   ==================================================
   Running step: 05_get_width.py
   ==================================================
   ```
   - Calculates gripper opening width from marker pairs
   - Processes configured hands (both for bimanual, one for mono-manual)
   - Creates `gripper_width_*.csv` files
   - Reports detection rates and interpolation results

   **Step 4: Dataset Plan Generation** (`07_generate_dataset_plan.py`)
   ```
   ==================================================
   Running step: 07_generate_dataset_plan.py
   ==================================================
   ```
   - **Automatically detects** demo mode (single-hand or bimanual) for each demo
   - Synchronizes images, Quest poses, and gripper widths with timestamps
   - Applies coordinate transformations (Unity left-handed → right-handed)
   - Creates `dataset_plan.pkl` with all synchronized data

   **Step 5: Replay Buffer Creation** (`08_generate_replay_buffer.py`)
   ```
   ==================================================
   Running step: 08_generate_replay_buffer.py
   ==================================================
   ```
   - Processes all frames in parallel
   - Resizes images to target resolution
   - **Adapts zarr structure** based on number of grippers detected
   - Compresses into zarr format with JPEG-XL
   - Creates final `<task_name>.zarr.zip`

3. **Completion Message**:
   ```
   Pipeline completed successfully!
   ```

#### Expected Output

After successful completion, you'll find in `data/<task_name>/`:

```
data/<task_name>/
├── dataset_plan.pkl                      # Episode metadata (40 KB typical)
├── <task_name>.zarr.zip                  # Final dataset (71 MB typical)
└── demos/
    ├── demo_bimanual_<timestamp_1>/
    │   ├── left_hand/
    │   │   └── cropped_img/             # Cropped images
    │   ├── right_hand/
    │   │   └── cropped_img/
    │   └── pose_data/                    # Pose CSV files
    └── ...
```

#### Typical Execution Time

For a dataset with 5 bimanual demos (~1000 frames):
- **Step 1**: 3 seconds
- **Step 2**: 3 seconds
- **Step 3**: 2 seconds
- **Step 4**: 1 second
- **Step 5**: 8 seconds
- **Total**: ~20 seconds

#### Success Indicators

Look for these messages to confirm each step succeeded:
- ✓ `Completed step: 01_crop_img.py`
- ✓ `Completed step: 04_get_aruco_pos.py`
- ✓ `Completed step: 05_get_width.py`
- ✓ `Completed step: 07_generate_dataset_plan.py`
- ✓ `Completed step: 08_generate_replay_buffer.py`
- ✓ `Pipeline completed successfully!`

#### Error Handling

If any step fails:
- The pipeline will stop and display an error message
- Check the error output for details
- Fix the issue (see Troubleshooting section)
- Re-run the pipeline (it will reprocess all steps)

### Manual Step-by-Step Execution

If you need to run steps individually for debugging or partial processing:

```bash
cd vitamin_b_data_collection_pipeline

# Step 1: Crop images
python 01_crop_img.py --cfg ../config/VB_task_config.yaml

# Step 2: Detect ArUco markers
python 04_get_aruco_pos.py --cfg ../config/VB_task_config.yaml

# Step 3: Calculate gripper widths
python 05_get_width.py --cfg ../config/VB_task_config.yaml

# Step 4: Generate dataset plan
python 07_generate_dataset_plan.py --cfg ../config/VB_task_config.yaml

# Step 5: Generate replay buffer
python 08_generate_replay_buffer.py --cfg ../config/VB_task_config.yaml
```

**Note**: Steps must be run in order as each depends on outputs from previous steps.

## Configuration

The `config/VB_task_config.yaml` file controls all pipeline behavior. Here are the key parameters:

### Task Configuration
```yaml
task:
  type: bimanual              # "bimanual" or "single"
  name: _0118_bi_pick_and_place  # Your task name
  single_hand_side: left      # Only for type: single
```

### Data Locations
```yaml
recorder:
  output: "./data"            # Where to find/save data
```

### Camera Configuration
```yaml
recorder:
  camera_paths:
    left_hand: "/dev/video0"    # Left hand camera device
    right_hand: "/dev/video2"   # Right hand camera device
  camera:
    format: "MJPG"
    width: 3840
    height: 800
    fps: 30
  output: "./data"              # Where to find/save data
```

**Note**: For single-hand mode (`task.type: single`), only the camera specified by `task.single_hand_side` will be used.

### ArUco Detection
```yaml
calculate_width:
  cam_intrinsic_json_path: "../assets/intri_result/gopro_intrinsics_2_7k.json"
  aruco_dict:
    predefined: DICT_4X4_50     # ArUco dictionary type
  marker_size_map:              # Physical marker sizes in meters
    0: 0.02
    1: 0.02
    # ...
  
  # Marker IDs for left gripper fingers
  left_hand_aruco_id:
    left_id: 0
    right_id: 1
  
  # Marker IDs for right gripper fingers
  right_hand_aruco_id:
    left_id: 2
    right_id: 3
```

**Note**: For single-hand mode, only the marker IDs for the active hand (left or right) are used.
```

### Output Settings
```yaml
output_train_data:
  min_episode_length: 10      # Minimum frames per episode
  visual_out_res: [224, 224]  # Visual image output size
  tactile_out_res: [224, 224] # Tactile image output size
  
  use_tactile_img: True       # Include tactile images
  use_tactile_pc: False       # Include tactile point clouds
  use_ee_pose: False          # Include end-effector poses
  
  compression_level: 99       # Zarr compression (0-100)
  num_workers: 16             # Parallel processing workers
  
  use_inpaint_tag: True       # Inpaint ArUco markers in output
  use_mask: False             # Apply fisheye mask
```

### Coordinate Transformations
```yaml
output_train_data:
  tx_quest_2_ee_left_path: '../assets/tf_cali_result/quest_2_ee_left_hand.npy'
  tx_quest_2_ee_right_path: '../assets/tf_cali_result/quest_2_ee_right_hand.npy'
```

**Note**: For single-hand mode, only the transformation for the active hand is used.

**Important**: All paths in the config should be relative to the `data_collection` directory or use absolute paths.

## Pipeline Output

### Final Dataset

The main output is a compressed zarr archive ready for training:

**Location**: `data/<task_name>/<task_name>.zarr.zip`

**Example**: `data/_0118_bi_pick_and_place/_0118_bi_pick_and_place.zarr.zip`

**Typical Size**: 50-100 MB for 5 demos with ~500 frames

### Dataset Contents

The zarr archive contains the following arrays:

#### For Bimanual Tasks

**Robot Data (two robots):**
- `robot0_eef_pos`, `robot1_eef_pos`: End-effector position [N, 3]
  - Format: (x, y, z) in meters
- `robot0_eef_rot_axis_angle`, `robot1_eef_rot_axis_angle`: Rotation [N, 3]
  - Format: Axis-angle representation
- `robot0_gripper_width`, `robot1_gripper_width`: Gripper opening [N, 1]
  - Format: Width in meters
- `robot0_demo_start_pose`, `robot1_demo_start_pose`: Initial pose [N, 6]
  - Format: (x, y, z, rx, ry, rz)
- `robot0_demo_end_pose`, `robot1_demo_end_pose`: Target pose [N, 6]
  - Format: (x, y, z, rx, ry, rz)

#### For Mono-Manual (Single-Hand) Tasks

**Robot Data (one robot only):**
- `robot0_eef_pos`: End-effector position [N, 3]
- `robot0_eef_rot_axis_angle`: Rotation [N, 3]
- `robot0_gripper_width`: Gripper opening [N, 1]
- `robot0_demo_start_pose`: Initial pose [N, 6]
- `robot0_demo_end_pose`: Target pose [N, 6]

**Note**: No `robot1_*` arrays are created for single-hand tasks.

#### Camera Data (both modes)

#### Camera Data
- `camera0_rgb` / `camera1_rgb`: Visual images [N, H, W, 3]
  - Resolution: As configured (default 224×224)
  - Format: RGB uint8
- `camera0_left_tactile` / `camera0_right_tactile`: Tactile images [N, H, W, 3]
- `camera1_left_tactile` / `camera1_right_tactile`: Tactile images [N, H, W, 3]
  - Resolution: As configured (default 224×224)
  - Format: RGB uint8

#### Metadata
- `episode_ends`: Frame indices where episodes end [num_episodes]
- Compression: JPEG-XL at configured quality level

### Intermediate Files

During processing, these intermediate files are created:

#### In each demo folder:
```
demos/demo_bimanual_<timestamp>/
├── left_hand/
│   └── cropped_img/                    # Cropped & corrected images
│       ├── 00000.jpg
│       ├── 00001.jpg
│       └── ...
├── right_hand/
│   └── cropped_img/
│       └── ...
└── pose_data/                          # Pose CSV files
    ├── left_visual_pose.csv           # Left hand poses
    ├── right_visual_pose.csv          # Right hand poses
    ├── gripper_width_left.csv         # Left gripper widths
    └── gripper_width_right.csv        # Right gripper widths
```

#### In task root:
```
data/<task_name>/
├── dataset_plan.pkl                    # Episode metadata (~40 KB)
└── <task_name>.zarr.zip               # Final dataset (50-100 MB)
```

### Loading the Dataset

To load and use the dataset in Python:

```python
import zarr

# Load the dataset
dataset_path = "data/_0118_bi_pick_and_place/_0118_bi_pick_and_place.zarr.zip"
store = zarr.ZipStore(dataset_path, mode='r')
root = zarr.group(store=store)

# Access data
eef_pos = root['data']['robot0_eef_pos'][:]      # Shape: (N, 3)
visual = root['data']['camera0_rgb'][:]           # Shape: (N, 224, 224, 3)
tactile = root['data']['camera0_left_tactile'][:] # Shape: (N, 224, 224, 3)
episode_ends = root['meta']['episode_ends'][:]    # Episode boundaries

# Get episode indices
episode_starts = [0] + episode_ends[:-1].tolist()
episode_ends = episode_ends.tolist()

# Access specific episode
episode_idx = 0
start_idx = episode_starts[episode_idx]
end_idx = episode_ends[episode_idx]

episode_data = {
    'eef_pos': eef_pos[start_idx:end_idx],
    'visual': visual[start_idx:end_idx],
    'tactile': tactile[start_idx:end_idx],
}

print(f"Episode {episode_idx} has {end_idx - start_idx} frames")
```

## Pipeline Details

### Step 1: Image Cropping (01_crop_img.py)
- Extracts frames from MP4 videos
- Applies fisheye lens correction
- Crops to tactile sensor region
- Saves corrected images for downstream processing

### Step 2: ArUco Detection (04_get_aruco_pos.py)
- Detects ArUco markers in cropped images
- Calculates 3D pose from marker positions
- Saves pose data as CSV files

### Step 3: Gripper Width (05_get_width.py)
- Measures distance between ArUco markers
- Calculates gripper opening width
- Validates and filters measurements

### Step 4: Dataset Plan (07_generate_dataset_plan.py)
- Loads and validates all demo episodes
- Synchronizes poses across left/right hands
- Applies coordinate transformations (Unity → Robot frame)
- Filters invalid frames
- Creates episode metadata

### Step 5: Replay Buffer (08_generate_replay_buffer.py)
- Loads dataset plan
- Processes all video frames in parallel
- Resizes images to target resolution
- Applies image augmentation (inpainting for ArUco tags)
- Compresses data into zarr format
- Creates final training-ready dataset

## Coordinate Transformations

The pipeline performs several coordinate transformations:

1. **Unity → Right-handed**: Converts Unity's left-handed coordinates
2. **Hand-wrist swap**: Corrects for different hand attachment conventions
3. **Quest → EE**: Transforms VR controller poses to robot end-effector poses

All transformations use calibration data from `assets/tf_cali_result/`.

## Troubleshooting

### Common Issues and Solutions

#### 1. Missing Dependencies
**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### 2. Import Errors
**Error**: `ImportError: cannot import name 'xxx'`

**Solution**: Make sure you're running from the correct directory:
```bash
cd /path/to/data_collection
python run_data_collection_pipeline.py
```

#### 3. Low ArUco Detection Rates
**Symptoms**: 
- `ArUco Detection: 50%` or lower
- Many "NOT FOUND" messages

**Solutions**:
- Check lighting in videos (should be bright and even)
- Verify ArUco markers are visible and not occluded
- Check marker sizes in config match physical markers
- Verify camera calibration file is correct

#### 4. Low Gripper Width Detection
**Symptoms**: 
- `gripper_width_left.csv: 50/100 valid (50.0%)`
- Many invalid measurements

**Solutions**:
- Ensure ArUco markers are clearly visible
- Check that markers on gripper fingers are not damaged
- Verify marker IDs in config match physical setup

#### 5. Memory Issues
**Error**: `MemoryError` or system slowdown

**Solutions**:
- Reduce `num_workers` in config (try 8 or 4)
- Reduce output resolutions (e.g., `[128, 128]`)
- Process fewer demos at a time
- Close other applications

#### 6. Path Not Found Errors
**Error**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solutions**:
- Verify data directory structure matches expected format
- Check `task.name` in config matches data folder name
- Ensure all paths in config are relative to `data_collection/` directory
- Use absolute paths if relative paths cause issues

#### 7. Pipeline Hangs or Freezes
**Symptoms**: No output for several minutes

**Solutions**:
- Check if a step is processing (CPU/disk activity)
- Reduce `num_workers` to avoid resource contention
- Kill the process (Ctrl+C) and restart with fewer workers

#### 8. Zarr Compression Fails
**Error**: Issues during step 5

**Solutions**:
- Verify `zarr==2.16.0` and `numcodecs==0.11.0` are installed
- Check disk space (need ~2x the output size free)
- Try reducing `compression_level` (e.g., 90 instead of 99)

#### 9. Wrong Number of Cameras/Grippers Detected
**Error**: `AssertionError: n_cameras != expected` or missing gripper data

**Solutions**:
- **Check task.type in config**: Must match your data (bimanual vs single)
- **For bimanual**: Ensure both `left_hand_img/` and `right_hand_img/` exist
- **For single**: Ensure only the specified hand's folder exists
- **Mixed datasets**: If processing mixed data, ensure each demo has consistent structure
- Verify `task.single_hand_side` is set correctly for single-hand mode

#### 10. ArUco Markers Not Detected in Single-Hand Mode
**Error**: Low detection rate or no markers found

**Solutions**:
- Verify `task.single_hand_side` matches your recorded hand (left or right)
- Check that the correct ArUco IDs are configured:
  - If `single_hand_side: left`, use `left_hand_aruco_id` values
  - If `single_hand_side: right`, use `right_hand_aruco_id` values
- Ensure marker IDs in config match physical markers on your gripper

## Switching Between Bimanual and Mono-Manual Modes

The pipeline fully supports both bimanual (two-handed) and mono-manual (single-handed) data recording and processing. Here's how to switch between modes:

### 1. Update Configuration

**For Bimanual Mode:**
```yaml
task:
  type: bimanual

recorder:
  camera_paths:
    left_hand: "/dev/video0"
    right_hand: "/dev/video2"
```

**For Mono-Manual Mode:**
```yaml
task:
  type: single
  single_hand_side: left  # or "right"

recorder:
  camera_paths:
    left_hand: "/dev/video0"   # Used if single_hand_side: left
    right_hand: "/dev/video2"  # Used if single_hand_side: right
```

### 2. Ensure Correct Data Structure

**Bimanual demos must have:**
- `left_hand_img/` folder
- `right_hand_img/` folder
- `all_trajectory/` with Quest poses for both controllers

**Mono-manual demos must have:**
- **Either** `left_hand_img/` **OR** `right_hand_img/` (based on `single_hand_side`)
- `all_trajectory/` with Quest poses for the active controller

### 3. Recording New Data

When using `00_get_data.py` to record demonstrations:
- **Bimanual mode**: Records from **both cameras** simultaneously
- **Mono-manual mode**: Records from **one camera only** (specified by `single_hand_side`)

The script automatically adapts based on `task.type` in your config.

### 4. Processing Existing Data

The pipeline automatically detects the mode for each demo:
- Steps 01-03 (crop, ArUco, width) process only the hands that have data
- Steps 07-08 (dataset plan, replay buffer) **auto-detect** the mode by checking which image folders exist

**You can even mix bimanual and mono-manual demos in the same dataset!** The pipeline will process each demo according to its structure.

### 5. Verify ArUco Configuration

Make sure your ArUco marker IDs match your hardware:

```yaml
calculate_width:
  # For left gripper (used when single_hand_side: left OR in bimanual mode)
  left_hand_aruco_id:
    left_id: 0
    right_id: 1
  
  # For right gripper (used when single_hand_side: right OR in bimanual mode)
  right_hand_aruco_id:
    left_id: 2
    right_id: 3
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Each step prints detailed progress information
2. **Run validation**: `python test_standalone.py`
3. **Test individual steps**: Run each step manually to isolate the problem
4. **Check data format**: Ensure your raw data matches the expected structure

## Performance Benchmarks

Based on testing with real data (`_0118_bi_pick_and_place` dataset):

### System Configuration
- **Dataset**: 5 bimanual demos, 1034 total frames
- **Hardware**: Standard workstation (16 cores, 32GB RAM)
- **Output**: 71 MB zarr.zip file

### Execution Times

| Step | Description | Time | Success Rate |
|------|-------------|------|--------------|
| 1 | Image Cropping | 3 sec | 100% (1034/1034) |
| 2 | ArUco Detection | 3 sec | 100% (10/10 tasks) |
| 3 | Gripper Width | 2 sec | 100% (1034/1034) |
| 4 | Dataset Plan | 1 sec | 100% (5/5 demos) |
| 5 | Replay Buffer | 8 sec | 100% (517 frames) |
| **Total** | **Full Pipeline** | **~20 sec** | **100%** |

### Processing Rates
- **Image cropping**: ~345 images/second
- **ArUco detection**: ~3 tasks/second
- **Gripper width**: ~500 frames/second
- **Zarr compression**: ~65 frames/second

### Resource Usage
- **Peak Memory**: ~2-3 GB
- **Disk Space**: Needs ~2x output size free (temp + final)
- **CPU**: Utilizes all available cores (configurable)

### Scaling Expectations

| Dataset Size | Frames | Expected Time | Output Size |
|--------------|--------|---------------|-------------|
| Small | ~200 | ~8 sec | ~30 MB |
| Medium | ~500 | ~20 sec | ~70 MB |
| Large | ~1000 | ~35 sec | ~140 MB |
| Very Large | ~2000 | ~65 sec | ~280 MB |

**Note**: Times may vary based on:
- Video resolution and quality
- Number of worker processes
- Compression level
- Hardware specifications
- Disk I/O speed

## Advanced Usage

### Customizing the Pipeline

You can modify `run_data_collection_pipeline.py` to:
- Skip certain steps
- Add custom processing
- Change execution order
- Add logging or monitoring

Example: Skip gripper width calculation (if not needed):
```python
pipeline_steps = [
    "01_crop_img.py",
    "04_get_aruco_pos.py",
    # "05_get_width.py",  # Commented out
    "07_generate_dataset_plan.py",
    "08_generate_replay_buffer.py",
]
```

### Processing Subsets of Data

To process only specific demos, modify the data directory temporarily:
```bash
# Backup original
mv data/task_name/demos data/task_name/demos_all

# Create subset
mkdir data/task_name/demos
cp -r data/task_name/demos_all/demo_bimanual_001 data/task_name/demos/
cp -r data/task_name/demos_all/demo_bimanual_002 data/task_name/demos/

# Run pipeline
python run_data_collection_pipeline.py

# Restore
rm -rf data/task_name/demos
mv data/task_name/demos_all data/task_name/demos
```

### Parallel Processing Multiple Tasks

Process multiple tasks in parallel using separate terminal sessions:
```bash
# Terminal 1
cd data_collection
python run_data_collection_pipeline.py  # Task 1

# Terminal 2
cd data_collection_copy
python run_data_collection_pipeline.py  # Task 2
```

### Cleaning Intermediate Files

Use the provided `clean_processed_data.py` script to safely remove intermediate and result files:

```bash
# Preview what will be deleted (safe, no actual deletion)
python clean_processed_data.py --data-dir data/_0118_bi_pick_and_place --dry-run

# Actually delete files (asks for confirmation)
python clean_processed_data.py --data-dir data/_0118_bi_pick_and_place

# Use config file to auto-detect data directory
python clean_processed_data.py --config config/VB_task_config.yaml

# Skip confirmation prompt
python clean_processed_data.py --data-dir data/_0118_bi_pick_and_place --yes
```

**What gets removed**:
- `dataset_plan.pkl` (dataset metadata)
- `*.zarr.zip` (final compressed dataset)
- `demos/*/cropped_img/` (cropped images)
- `demos/*/pose_data/` (pose trajectories)
- `demos/*/gripper_width_*.csv` (gripper measurements)

**What gets preserved** (raw data):
- `demos/*/all_trajectory/` (Quest pose JSONs)
- `demos/*/left_hand_img/` & `right_hand_img/` (raw images)
- `demos/*/metadata.json` (demo information)

This is useful for:
- **Reprocessing with different parameters**: Clean old results before running pipeline again
- **Disk space**: Free ~50-150 MB per dataset
- **Archiving**: Keep only raw data for long-term storage

See `CLEANING_GUIDE.md` for detailed documentation.

## Quick Reference

### Essential Commands

```bash
# Setup
python test_standalone.py                    # Validate installation

# Run full pipeline
python run_data_collection_pipeline.py       # Process all steps

# Clean processed files
python clean_processed_data.py --config config/VB_task_config.yaml --dry-run  # Preview
python clean_processed_data.py --config config/VB_task_config.yaml --yes      # Execute

# Run individual steps
cd vitamin_b_data_collection_pipeline
python 01_crop_img.py --cfg ../config/VB_task_config.yaml
python 04_get_aruco_pos.py --cfg ../config/VB_task_config.yaml
python 05_get_width.py --cfg ../config/VB_task_config.yaml
python 07_generate_dataset_plan.py --cfg ../config/VB_task_config.yaml
python 08_generate_replay_buffer.py --cfg ../config/VB_task_config.yaml
```

### File Locations

| File | Location | Purpose |
|------|----------|---------|
| Configuration | `config/VB_task_config.yaml` | Pipeline settings |
| Raw data | `data/<task>/demos/` | Input videos & poses |
| Output dataset | `data/<task>/<task>.zarr.zip` | Final training data |
| Episode metadata | `data/<task>/dataset_plan.pkl` | Episode information |
| Calibration | `assets/*_result/` | Camera & transformation data |

### Key Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Task name | `task.name` | - | Dataset identifier |
| Task type | `task.type` | `bimanual` | Single or bimanual |
| Output resolution | `output_train_data.visual_out_res` | `[224, 224]` | Image size |
| Workers | `output_train_data.num_workers` | 16 | Parallel processes |
| Compression | `output_train_data.compression_level` | 99 | Zarr compression |

### Common Workflows

**New Dataset Processing**:
1. Place raw data in `data/<task_name>/demos/`
2. Update `task.name` in config
3. Run `python run_data_collection_pipeline.py`
4. Check output: `data/<task_name>/<task_name>.zarr.zip`

**Debugging Issues**:
1. Run `python test_standalone.py` to verify setup
2. Run individual steps manually to isolate problem
3. Check error messages and consult Troubleshooting section
4. Verify data structure matches expected format

**Batch Processing**:
1. Process multiple datasets by changing `task.name`
2. Or copy entire `data_collection/` folder for parallel processing
3. Each task should have its own data folder

## Citation

If you use this pipeline, please cite:
```bibtex
@article{vitaminb2024,
  title={ViTaMIn-B: Vision-Tactile Manipulation with Bimanual Robots},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Support

### Documentation
- **README.md**: This file - complete usage guide
- **SETUP_SUMMARY.md**: Installation and validation summary
- **TEST_CLEANUP_REPORT.md**: Testing and maintenance report
- **Pipeline_overview.md**: Technical implementation details

### Getting Help
1. Check the Troubleshooting section above
2. Run `python test_standalone.py` to validate setup
3. Review error messages for specific issues
4. Consult technical documentation for advanced topics

## License

[Add license information]

## Contact

For questions or issues, please open an issue or contact [maintainer email].
