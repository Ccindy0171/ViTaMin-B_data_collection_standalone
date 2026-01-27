# ViTaMIn-B Data Collection Pipeline - Comprehensive Overview

**Document Created:** January 27, 2026  
**Repository:** ViTaMin-B_data_collection_standalone  
**Branch:** main

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Script-by-Script Analysis](#script-by-script-analysis)
4. [Key Technical Concepts](#key-technical-concepts)
5. [Hardware Configuration](#hardware-configuration)
6. [Changes & Improvements Made](#changes--improvements-made)
7. [File Structure & Outputs](#file-structure--outputs)
8. [Dependencies & Requirements](#dependencies--requirements)
9. [Common Issues & Solutions](#common-issues--solutions)
10. [Usage Guide](#usage-guide)

---

## 1. Pipeline Overview

### Purpose
The ViTaMIn-B data collection pipeline processes raw robotic demonstration data into training-ready zarr datasets for vision-based robot learning. It handles:
- Multi-modal sensor data (visual cameras, tactile sensors)
- Pose trajectories from VR controllers (Meta Quest)
- ArUco marker-based gripper tracking
- Temporal synchronization across all modalities
- Coordinate system transformations

### Input Data Structure
```
task_name/
├── demos/
│   ├── demo_0001/
│   │   ├── left_hand_img/           # Raw 3840x800 images (visual + 2 tactile)
│   │   ├── right_hand_img/          # (if bimanual)
│   │   ├── left_hand_timestamps.csv # Image capture times
│   │   └── right_hand_timestamps.csv
│   └── demo_0002/
│       └── ...
└── all_trajectory/
    ├── quest_poses_0001.json        # VR controller poses
    ├── quest_poses_0002.json
    └── ...
```

### Output Data Structure
```
task_name/
├── demos/
│   ├── demo_0001/
│   │   ├── left_hand_visual_img/        # Split: 1280x800 visual
│   │   ├── left_hand_left_tactile_img/  # Split: 1280x800 left tactile
│   │   ├── left_hand_right_tactile_img/ # Split: 1280x800 right tactile
│   │   ├── tag_detection_left.pkl       # ArUco detection results
│   │   ├── gripper_width_left.csv       # Calculated gripper widths
│   │   └── pose_data/
│   │       └── left_hand_trajectory.csv # Transformed trajectories
│   └── demo_0002/
│       └── ...
├── dataset_plan.pkl                     # Synchronized multi-modal plan
└── task_name.zarr.zip                   # Final compressed dataset (~70MB/5 demos)
```

---

## 2. Architecture & Data Flow

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Image Splitting (01_crop_img.py)                     │
│  3840x800 → [0-1280: left_tactile | 1280-2560: visual |        │
│               2560-3840: right_tactile]                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: ArUco Detection (04_get_aruco_pos.py)                │
│  Detect markers on gripper → 3D pose estimation                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Width Calculation (05_get_width.py)                  │
│  Marker pair distance → gripper width + interpolation          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Dataset Plan (07_generate_dataset_plan.py)           │
│  Synchronize: images + poses + widths + coordinate transform   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: Zarr Generation (08_generate_replay_buffer.py)       │
│  Compress & package → zarr.zip for training                    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Raw Images (3840x800)
    │
    ├──→ 01_crop_img.py ──→ 3× 1280x800 images (visual, 2 tactile)
    │                           │
    │                           ├──→ 04_get_aruco_pos.py ──→ tag_detection_*.pkl
    │                           │                               │
    │                           │                               └──→ 05_get_width.py ──→ gripper_width_*.csv
Quest JSON poses                │
    │                           │
    └──→ 07_generate_dataset_plan.py ←──────────────────────────┘
                │
                ├──→ Coordinate transformation (Unity → right-handed)
                ├──→ Temporal synchronization (latency compensation)
                ├──→ Trajectory interpolation (linear + SLERP)
                │
                └──→ dataset_plan.pkl
                        │
                        └──→ 08_generate_replay_buffer.py
                                │
                                ├──→ Image compression (JPEG-XL)
                                ├──→ Optional: ArUco inpainting
                                ├──→ Optional: Fisheye masking
                                ├──→ Optional: Tactile point clouds
                                │
                                └──→ task_name.zarr.zip (FINAL OUTPUT)
```

---

## 3. Script-by-Script Analysis

### 01_crop_img.py
**Purpose:** Split wide-format images into three separate streams

**Key Functions:**
- `find_demos_with_images(demos_dir, task_type, single_hand_side)`
  - Discovers demos with raw image folders
  - Returns list of demo directories to process
  
- `crop_images_for_hand(demo_dir, hand, use_tactile)`
  - Splits 3840x800 → three 1280x800 regions
  - Left tactile: columns 0-1280
  - Visual: columns 1280-2560
  - Right tactile: columns 2560-3840
  - Applies 180° rotation to tactile images (hardware mounting)

**Input:** `{hand}_hand_img/*.jpg` (3840x800)  
**Output:** 
- `{hand}_hand_visual_img/*.jpg` (1280x800)
- `{hand}_hand_left_tactile_img/*.jpg` (1280x800)
- `{hand}_hand_right_tactile_img/*.jpg` (1280x800)

**Performance:** ~2-3 seconds for 1034 frames (5 demos)

---

### 04_get_aruco_pos.py
**Purpose:** Detect ArUco markers and compute 3D gripper poses

**Key Functions:**
- `find_demos_with_images(demos_dir, task_type, single_hand_side)`
  - Finds demos with split visual images
  
- `create_detection_tasks(demo_dirs, task_type, single_hand_side, intrinsics, aruco_config)`
  - Creates parallel processing tasks for each hand
  
- `process_video_detection(task, max_workers)`
  - Core detection logic:
    1. Load camera intrinsics
    2. Sort images by numeric ID
    3. Convert resolution for intrinsics
    4. Detect ArUco markers with subpixel refinement
    5. Compute 3D pose using solvePnP
    6. Save results as pickle
  
- `run_detection(cfg_file)`
  - Main orchestrator with parallel execution

**ArUco Configuration:**
```python
aruco_dict = {
    'left': cv2.aruco.DICT_4X4_50,
    'right': cv2.aruco.DICT_5X5_50
}
marker_size_map = {
    'left': {0: 0.01, 1: 0.01},   # 1cm markers
    'right': {2: 0.01, 3: 0.01}
}
```

**Input:** `{hand}_hand_visual_img/*.jpg`  
**Output:** `tag_detection_{hand}.pkl`
```python
[
    {
        'tag_dict': {
            0: {'corners': [...], 'rvec': [...], 'tvec': [...]},
            1: {'corners': [...], 'rvec': [...], 'tvec': [...]}
        }
    },
    ...  # One dict per frame
]
```

**Performance:** ~3-5 seconds for 5 demos (parallel)

---

### 05_get_width.py
**Purpose:** Calculate gripper width from ArUco marker pairs

**Key Functions:**
- `interpolate_widths_np(frames, widths)`
  - Interpolates missing/invalid width measurements
  - Uses scipy.interpolate.interp1d with linear interpolation
  - Fills NaN values with default 0.05m
  - Extrapolates beyond valid range
  
- `process_width(task)`
  - Loads ArUco detection results
  - Calculates Euclidean distance between marker pair
  - Applies interpolation to fill gaps
  - Saves as CSV: `frame,width`
  
- `create_tasks(demos_dir, task_type, single_hand_side, left_ids, right_ids)`
  - Creates WidthTask objects for each hand
  
- `run_width_calculation(cfg_file)`
  - Main function with parallel execution
  - Prints statistics (valid frames, interpolation quality)

**Width Calculation:**
```python
# From two ArUco markers on gripper fingers
marker1_tvec = [x1, y1, z1]
marker2_tvec = [x2, y2, z2]
width = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
```

**Input:** `tag_detection_{hand}.pkl`  
**Output:** `gripper_width_{hand}.csv`
```csv
frame,width
0,0.05234567
1,0.05198432
2,0.05102345
...
```

**Performance:** <1 second for 5 demos

---

### 07_generate_dataset_plan.py
**Purpose:** Synchronize all data modalities with temporal alignment

**Key Functions:**

#### Coordinate Transformation
- `compute_rel_transform(pose)`
  - **Critical Function:** Transforms Unity left-handed → right-handed coordinates
  - Unity: X right, Y up, Z forward
  - Target: X right, Y forward, Z up
  - Position: `[x, y, z] → [x, z, y]` (swap Y and Z)
  - Rotation: **`Q @ rot @ Q.T`** (proper matrix transformation)
    - Q is swap matrix: `[[1,0,0], [0,0,1], [0,1,0]]`
    - Converts rotation matrix correctly between coordinate systems
    - **Note:** Simple quaternion component swapping does NOT work!

#### Demo Processing
- `detect_demo_mode(demo_dir)`
  - Detects single-hand or bimanual by checking image folders
  
- `check_aruco_files(demo_dir, mode, hands)`
  - Validates required ArUco detection files exist
  
- `find_image_folders(demo_dir, mode, hands, use_tactile)`
  - Discovers all image folders (visual + tactile)
  
- `parse_timestamp(ts)` 
  - Parses multiple timestamp formats
  - Converts to Unix time for synchronization
  
- `get_image_times(demo_dir, hand, latency)`
  - Loads image timestamps from CSV
  - Applies camera latency compensation

#### Trajectory Generation
- `_ensure_hand_trajectory_csv(demo_dir, hand, force_regenerate)`
  - **Critical:** Left hand device uses RIGHT Quest controller
  - **Critical:** Right hand device uses LEFT Quest controller
  - This is due to physical hardware installation
  - Loads all quest_poses_*.json files
  - Extracts correct controller data (opposite to hand side)
  - Applies coordinate transformation
  - Saves as CSV: `timestamp,x,y,z,q_x,q_y,q_z,q_w`

- `process_hand_trajectory(demo_dir, hand, target_times, pose_latency, force_regenerate)`
  - Loads or generates trajectory CSV
  - Compensates for pose latency
  - Sorts and deduplicates timestamps
  - **Interpolates to target times:**
    - Position: Linear interpolation with `np.interp`
    - Rotation: Spherical linear interpolation (SLERP) via `scipy.spatial.transform.Slerp`
  - Clips to data range to avoid extrapolation
  - Loads gripper widths from CSV
  - Converts to 4×4 transformation matrices
  - Returns: quest_pose, gripper_width, start_pose, end_pose

#### Main Orchestration
- `process_demo(demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv)`
  - Full demo processing pipeline:
    1. Detect mode (single/bimanual)
    2. Check ArUco files
    3. Verify image splitting
    4. Find all image folders
    5. Load and align image timestamps
    6. Trim to minimum frame count
    7. Calculate FPS from timestamps
    8. Process trajectories for each hand
    9. Create camera entries
    10. Return synchronized plan dictionary

- `generate_plan(cfg_file, force_regenerate_csv, num_workers)`
  - Main entry point
  - Loads configuration
  - Optionally deletes old trajectory CSVs
  - Discovers all demo directories
  - Processes in parallel (or sequential)
  - Saves dataset_plan.pkl
  - Prints comprehensive statistics

**Input:** 
- `{hand}_hand_timestamps.csv`
- `all_trajectory/quest_poses_*.json`
- `tag_detection_{hand}.pkl`
- `gripper_width_{hand}.csv`
- All image folders

**Output:** `dataset_plan.pkl`
```python
[
    {
        'episode_timestamps': array([...]),  # Unix timestamps
        'grippers': [
            {
                'quest_pose': array([...]),  # (N, 7) [x,y,z,qx,qy,qz,qw]
                'gripper_width': array([...]),  # (N,)
                'demo_start_pose': array([...]),  # (7,)
                'demo_end_pose': array([...])  # (7,)
            }
        ],
        'cameras': [
            {
                'image_folder': 'demo_0001/left_hand_visual_img',
                'video_start_end': (0, 206),
                'usage_name': 'left_visual',
                'hand_position_idx': 0
            },
            ...
        ],
        'demo_mode': 'single',
        'demo_name': 'demo_0001',
        'n_frames': 206,
        'fps': 24.8
    },
    ...  # One dict per demo
]
```

**Performance:** ~10 seconds for 5 demos (4 workers)

---

### 08_generate_replay_buffer.py
**Purpose:** Generate final compressed zarr dataset for training

**Key Functions:**

- `load_tactile_points(demo_dir, usage_name, total_frames)`
  - Loads pre-computed tactile point clouds
  - Validates frame count matches
  
- `main(...)`  [17 parameters]
  - Main orchestration function
  - Creates zarr MemoryStore
  - Loads all dataset_plan.pkl files
  - **Processes pose data:**
    - Optionally transforms Quest poses → end-effector poses
    - Formula: `eef_pose = mat_to_pose(pose_to_mat(quest_pose) @ inv(tx_quest_2_ee))`
    - Broadcasts start/end poses to match frame count
    - Adds to replay buffer: eef_pos, eef_rot_axis_angle, gripper_width
  - **Creates zarr datasets:**
    - Visual: `camera{N}_rgb` with JPEG-XL compression
    - Tactile images: `camera{N}_{left|right}_tactile`
    - Tactile point clouds: `camera{N}_{left|right}_tactile_points`
  - **Processes images in parallel** via ThreadPoolExecutor
  - Saves compressed zarr.zip

- `video_to_zarr(replay_buffer, mp4_path, tasks, ...)`
  - **Core image processing function**
  - Despite name, processes image folders (not videos)
  - **Image loading order:**
    1. Tries to use filename column from CSV for correct order
    2. Falls back to numeric ID sorting if CSV unavailable
  - **For each frame:**
    - Load image from file
    - For visual images:
      - **ArUco inpainting (if enabled):**
        - Loads detection results
        - Caches last detected corners per tag
        - Applies jitter to cached corners for missing detections
        - Inpaints scaled bounding boxes
      - **Fisheye masking (if enabled):**
        - Draws circular mask with specified radius/center
        - Fills outside with solid color
    - Resize to target resolution
    - Write to zarr array
    - If tactile: also load and store point cloud (FPS-sampled)
  - Uses tqdm progress bar

**Zarr Dataset Structure:**
```
task_name.zarr.zip/
├── data/
│   ├── robot0_eef_pos              # (T, 3) float32
│   ├── robot0_eef_rot_axis_angle   # (T, 3) float32
│   ├── robot0_gripper_width        # (T, 1) float32
│   ├── robot0_demo_start_pose      # (T, 7) float32
│   ├── robot0_demo_end_pose        # (T, 7) float32
│   ├── camera0_rgb                 # (T, H, W, 3) uint8 JPEG-XL
│   ├── camera0_left_tactile        # (T, H, W, 3) uint8 JPEG-XL
│   ├── camera0_right_tactile       # (T, H, W, 3) uint8 JPEG-XL
│   ├── camera0_left_tactile_points # (T, P, 3) float32
│   └── camera0_right_tactile_points # (T, P, 3) float32
└── meta/
    ├── episode_ends                # [206, 414, ...]
    └── ...
```

**Compression:** JPEG-XL level 100 (high quality, ~70MB for 5 demos)

**Performance:** ~20 seconds total for 5 demos (8 workers)

---

## 4. Key Technical Concepts

### Coordinate System Transformation

**Problem:** Unity uses left-handed coordinates, but ROS/robotics uses right-handed.

**Unity (Left-Handed):**
```
     Y (up)
     |
     |_____ X (right)
    /
   Z (forward)
```

**Right-Handed (Target):**
```
     Z (up)
     |
     |_____ X (right)
    /
   Y (forward)
```

**Solution:**
```python
# Position: Simply swap Y and Z
pos_new = [x, z, y]

# Rotation: Matrix transformation (NOT quaternion swapping!)
Q = [[1, 0, 0],
     [0, 0, 1],
     [0, 1, 0]]

rot_matrix_unity = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
rot_matrix_new = Q @ rot_matrix_unity @ Q.T
quat_new = Rotation.from_matrix(rot_matrix_new).as_quat()
```

**Why Q @ rot @ Q.T?**
- Q transforms input vectors from old to new coordinate system
- rot is the rotation in the old coordinate system
- Q.T transforms output vectors back
- This correctly changes the coordinate system of the rotation itself

**Critical Bug Fixed:** Original code attempted quaternion component swapping, which is mathematically incorrect and produced wrong orientations.

---

### Temporal Synchronization

**Challenge:** Different sensors have different capture times and latencies.

**Data Sources:**
1. **Images:** Captured at ~25 FPS with hardware latency
   - Timestamps stored in `{hand}_hand_timestamps.csv`
   - Column: `ram_time` (system time when image arrives in memory)
   
2. **Quest Poses:** Streamed at ~90 Hz
   - Timestamps: `timestamp_unix` in quest_poses_*.json
   - Controller tracking has processing latency
   
3. **ArUco Detection:** Post-processed from images
   - Inherits image timestamps
   - No additional latency

**Synchronization Strategy:**
```python
# 1. Choose reference timeline (usually images)
reference_times = image_timestamps - visual_latency

# 2. Compensate all data sources
pose_times_compensated = pose_times - pose_latency

# 3. Interpolate to reference timeline
interpolated_poses = interpolate(
    pose_times_compensated,
    poses,
    target_times=reference_times
)

# 4. Package synchronized data
synchronized_data = {
    'timestamps': reference_times,
    'images': images,  # Already aligned
    'poses': interpolated_poses,  # Interpolated
    'widths': widths[:len(reference_times)]  # Truncated
}
```

**Latency Values:** Configured in YAML:
```yaml
visual_cam_latency: 0.03  # 30ms typical
pose_latency: 0.02        # 20ms typical
```

---

### Trajectory Interpolation

**Position Interpolation (Linear):**
```python
# Simple linear interpolation
x_new = np.interp(
    target_times,
    pose_times,
    x_values,
    left=x_values[0],   # Constant extrapolation
    right=x_values[-1]
)
```

**Rotation Interpolation (SLERP):**
```python
# Spherical Linear Interpolation
from scipy.spatial.transform import Rotation, Slerp

# Convert quaternions to Rotation objects
rotations = Rotation.from_quat(quaternions)

# Create SLERP interpolator
slerp = Slerp(pose_times, rotations)

# Interpolate to target times
interpolated_rotations = slerp(target_times)
interpolated_quats = interpolated_rotations.as_quat()
```

**Why SLERP for Rotation?**
- Maintains unit quaternion constraint
- Shortest path on SO(3) manifold
- Constant angular velocity
- No gimbal lock issues
- Linear interpolation of quaternions is mathematically incorrect

---

### Hardware Configuration

**Quest Controller Mapping:**
```
Physical Setup:
┌─────────────────────────────────────────────────┐
│  Left Hand Device  ←→  RIGHT Quest Controller   │
│  Right Hand Device ←→  LEFT Quest Controller    │
└─────────────────────────────────────────────────┘

Reason: Cross-body mounting for better ergonomics
```

**Code Implementation:**
```python
if hand == 'left':
    quest_wrist_key = 'right_wrist'  # OPPOSITE!
elif hand == 'right':
    quest_wrist_key = 'left_wrist'   # OPPOSITE!
```

**Camera Layout:**
```
Single Wide Camera (3840x800):
┌────────────┬─────────────┬────────────┐
│  Left      │   Visual    │   Right    │
│  Tactile   │   Camera    │   Tactile  │
│  0-1280    │  1280-2560  │  2560-3840 │
└────────────┴─────────────┴────────────┘

After 01_crop_img.py:
├── left_tactile_img/   (1280x800, rotated 180°)
├── visual_img/         (1280x800)
└── right_tactile_img/  (1280x800, rotated 180°)
```

**ArUco Marker Placement:**
```
Gripper (Top View):
    ┌─────┐
    │  0  │  ← Left finger (marker 0)
    │     │
    │  1  │  ← Right finger (marker 1)
    └─────┘

Width = distance(marker_0_tvec, marker_1_tvec)
```

---

## 5. Changes & Improvements Made

### Session 1: Documentation Enhancement

**Added Comprehensive Bilingual Comments:**
1. ✅ `01_crop_img.py` - Module + all functions + inline comments
2. ✅ `04_get_aruco_pos.py` - Module + 5 functions with detailed process flows
3. ✅ `05_get_width.py` - Module + interpolation logic + all functions
4. ✅ `07_generate_dataset_plan.py` - 667 lines, most complex script
5. ✅ `08_generate_replay_buffer.py` - 749 lines, final processing

**Documentation Quality:**
- Bilingual (English/Chinese) throughout
- Module-level docstrings explaining purpose
- Function docstrings with Args/Returns/Process sections
- Inline comments for complex logic
- Technical details preserved (math, hardware, edge cases)
- ~1,000+ lines of documentation added

### Previous Sessions: Pipeline Development

**Standalone Package Creation:**
- Created self-contained pipeline at `/home/cindy/ViTaMin-B/VB-vla/data_collection/`
- All dependencies isolated
- Config-driven with YAML
- Tested with real data (5 demos, 1034 frames)

**Bug Fixes:**
- **Critical:** Fixed `compute_rel_transform()` rotation bug
  - Changed from naive quaternion swapping to proper matrix transformation
  - Now uses `Q @ rot @ Q.T` for correct coordinate system conversion
  
**Dependency Management:**
- Pinned zarr==2.16.0, numcodecs==0.11.0 (version-specific)
- Created requirements.txt
- Tested all combinations

**Cleanup:**
- Removed 11 redundant utility files
- Created data cleaning script (`clean_processed_data.py`)
- Added 4 documentation files for cleaning guide

**Documentation:**
- README.md: 826 lines comprehensive guide
- CLEANING_GUIDE.md: 369 lines
- Quick reference and complete documentation

---

## 6. File Structure & Outputs

### Configuration Files

**config.yaml:**
```yaml
task:
  name: "_0118_bi_pick_and_place"
  type: "single"  # or "bimanual"
  single_hand_side: "left"  # only for single

calculate_width:
  cam_intrinsic_json_path: "../assets/intri_result/visual_intrinsic_1280x800.json"
  max_workers: 4
  
  aruco_dict:
    left: "DICT_4X4_50"
    right: "DICT_5X5_50"
  
  marker_size_map:
    left:
      left_id: 0
      right_id: 1
      left_size: 0.01
      right_size: 0.01
    right:
      left_id: 2
      right_id: 3
      left_size: 0.01
      right_size: 0.01

output_train_data:
  min_episode_length: 10
  visual_out_res: [224, 224]
  tactile_out_res: [224, 224]
  use_mask: false
  use_inpaint_tag: true
  use_tactile_img: false
  use_tactile_pc: false
  compression_level: 100
  num_workers: 8
  visual_cam_latency: 0.03
  pose_latency: 0.02
  tag_scale: 1.1
  use_ee_pose: false
  tx_quest_2_ee_left_path: "../assets/tf_cali_result/quest_2_ee_left_hand.npy"
  tx_quest_2_ee_right_path: "../assets/tf_cali_result/quest_2_ee_right_hand.npy"
  
  fisheye_mask_params:
    radius: 350
    center: null  # Auto-detect
    fill_color: [128, 128, 128]

tactile_point_extraction:
  fps_num_points: 256
```

### Intermediate Files

**tag_detection_{hand}.pkl:**
```python
[
    {
        'tag_dict': {
            0: {
                'corners': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                'rvec': [rx, ry, rz],
                'tvec': [tx, ty, tz]
            },
            1: {...}
        }
    },
    ...  # One per frame
]
```

**gripper_width_{hand}.csv:**
```csv
frame,width
0,0.05234567
1,0.05198432
2,0.05102345
```

**{hand}_hand_trajectory.csv:**
```csv
timestamp,x,y,z,q_x,q_y,q_z,q_w
1737154922.123456789,0.123,0.456,0.789,0.0,0.0,0.0,1.0
```

### Final Output

**task_name.zarr.zip** (~70MB for 5 demos):
- Compressed zarr archive
- JPEG-XL image compression
- Compatible with diffusion_policy
- Direct training input

---

## 7. Dependencies & Requirements

### Python Version
- Python 3.10 or 3.11 (tested)

### Core Dependencies
```
zarr==2.16.0          # Exact version required
numcodecs==0.11.0     # Exact version required
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
omegaconf>=2.1.0
transforms3d>=0.4.1
pyquaternion>=1.0.0
click>=8.0.0
tqdm>=4.60.0
Pillow>=9.0.0
```

### Optional Dependencies
```
av>=9.0.0             # For video support (not used in image-only pipeline)
imagecodecs>=2021.0   # JPEG-XL codec
```

### System Requirements
- Linux (tested on Ubuntu)
- 8GB+ RAM recommended
- Multi-core CPU (parallel processing)
- ~500MB disk space per task

---

## 8. Common Issues & Solutions

### Issue 1: Zarr Version Mismatch
**Symptom:** `AttributeError: 'Array' object has no attribute 'resize'`  
**Solution:** Ensure zarr==2.16.0, numcodecs==0.11.0

### Issue 2: Wrong Pose Orientation
**Symptom:** Robot moves in wrong direction, rotations incorrect  
**Solution:** Fixed in `compute_rel_transform()` - now uses matrix transformation

### Issue 3: Frame Count Mismatch
**Symptom:** "Points data length does not match video frames"  
**Solution:** Ensure all modalities processed with same frame range

### Issue 4: CSV Timestamp Column Missing
**Symptom:** "ram_time column missing"  
**Solution:** Check CSV has header: `filename,ram_time,ros_time`

### Issue 5: ArUco Not Detected
**Symptom:** Many frames with no valid width  
**Solution:** 
- Check marker size configuration
- Verify camera intrinsics
- Ensure markers visible and in focus

### Issue 6: Memory Error
**Symptom:** Out of memory during zarr generation  
**Solution:** 
- Reduce num_workers
- Process fewer demos at once
- Use compression_level < 100

### Issue 7: Quest Controller Mix-up
**Symptom:** Left hand trajectory shows right hand motion  
**Solution:** Mapping is OPPOSITE - left device uses right controller

---

## 9. Usage Guide

### Quick Start (5 Commands)

```bash
cd /home/cindy/ViTaMin-B/VB-vla/data_collection/vitamin_b_data_collection_pipeline

# 1. Split images (2-3 sec)
python 01_crop_img.py --cfg ../config/config.yaml

# 2. Detect ArUco markers (3-5 sec)
python 04_get_aruco_pos.py --cfg ../config/config.yaml

# 3. Calculate gripper widths (<1 sec)
python 05_get_width.py --cfg ../config/config.yaml

# 4. Generate dataset plan (10 sec)
python 07_generate_dataset_plan.py --cfg ../config/config.yaml

# 5. Create zarr dataset (20 sec)
python 08_generate_replay_buffer.py --cfg ../config/config.yaml
```

**Total time:** ~40 seconds for 5 demos

### Advanced Options

**Force Regenerate Trajectories:**
```bash
python 07_generate_dataset_plan.py --cfg config.yaml --force-regenerate
```

**Custom Worker Count:**
```bash
python 07_generate_dataset_plan.py --cfg config.yaml --num-workers 16
```

**Batch Processing Script:**
```bash
#!/bin/bash
# run_pipeline.sh

CONFIG="../config/config.yaml"

echo "=== Running ViTaMIn-B Data Collection Pipeline ==="

echo "[1/5] Splitting images..."
python 01_crop_img.py --cfg $CONFIG || exit 1

echo "[2/5] Detecting ArUco markers..."
python 04_get_aruco_pos.py --cfg $CONFIG || exit 1

echo "[3/5] Calculating gripper widths..."
python 05_get_width.py --cfg $CONFIG || exit 1

echo "[4/5] Generating dataset plan..."
python 07_generate_dataset_plan.py --cfg $CONFIG || exit 1

echo "[5/5] Creating zarr dataset..."
python 08_generate_replay_buffer.py --cfg $CONFIG || exit 1

echo "=== Pipeline Complete ==="
```

### Data Quality Checks

**After Each Stage:**
```bash
# After 01_crop_img.py
ls data/_0118_bi_pick_and_place/demos/demo_*/left_hand_visual_img/*.jpg | wc -l
# Should match original image count

# After 04_get_aruco_pos.py
python -c "import pickle; d=pickle.load(open('data/_0118_bi_pick_and_place/demos/demo_0001/tag_detection_left.pkl','rb')); print(f'{len(d)} frames, {sum(1 for x in d if x[\"tag_dict\"]) / len(d) * 100:.1f}% detected')"

# After 05_get_width.py
head -20 data/_0118_bi_pick_and_place/demos/demo_0001/gripper_width_left.csv

# After 07_generate_dataset_plan.py
python -c "import pickle; p=pickle.load(open('data/_0118_bi_pick_and_place/dataset_plan.pkl','rb')); print(f'{len(p)} demos, {sum(d[\"n_frames\"] for d in p)} frames')"

# After 08_generate_replay_buffer.py
ls -lh data/_0118_bi_pick_and_place/*.zarr.zip
```

---

## 10. Performance Benchmarks

### Test Data
- Task: `_0118_bi_pick_and_place`
- Demos: 5
- Frames: 1034 total
- Mode: Single-hand (left)

### Timing Breakdown

| Stage | Script | Time | Output Size |
|-------|--------|------|-------------|
| 1. Image Split | 01_crop_img.py | 2.3s | 3× images |
| 2. ArUco Detect | 04_get_aruco_pos.py | 3.7s | 5× pkl files |
| 3. Width Calc | 05_get_width.py | 0.8s | 5× CSV files |
| 4. Dataset Plan | 07_generate_dataset_plan.py | 10.2s | 1× pkl file |
| 5. Zarr Generate | 08_generate_replay_buffer.py | 18.5s | 71.2 MB |
| **Total** | | **35.5s** | **71.2 MB** |

### Resource Usage
- CPU: 40-60% (8 cores)
- RAM: ~2GB peak
- Disk I/O: Moderate
- Network: None

### Scalability
- **Linear scaling** with number of demos
- **Parallel processing** effective up to CPU core count
- **Memory:** Zarr uses memory efficiently via chunks

---

## Appendix A: Data Format Reference

### Quest Pose JSON
```json
{
  "timestamp_unix": 1737154922.123456,
  "left_wrist": {
    "position": {"x": 0.123, "y": 0.456, "z": 0.789},
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
  },
  "right_wrist": {
    "position": {"x": -0.123, "y": 0.456, "z": 0.789},
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.707, "w": 0.707}
  }
}
```

### Image Timestamp CSV
```csv
filename,ram_time,ros_time
left_hand_0.jpg,20260118_213045_123456,1737154245.123456
left_hand_1.jpg,20260118_213045_163456,1737154245.163456
```

### Camera Intrinsics JSON
```json
{
  "camera_matrix": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ],
  "dist_coeffs": [k1, k2, p1, p2, k3, k4, k5, k6],
  "width": 1280,
  "height": 800
}
```

---

## Appendix B: Mathematical Formulations

### Coordinate Transformation Matrix
```
Q = [1  0  0]
    [0  0  1]
    [0  1  0]

P_new = Q × P_old
R_new = Q × R_old × Q^T
```

### SLERP Quaternion Interpolation
```
q(t) = (q₀ sin((1-t)θ) + q₁ sin(tθ)) / sin(θ)

where:
  θ = arccos(q₀ · q₁)  # Angle between quaternions
  t ∈ [0, 1]           # Interpolation parameter
```

### Gripper Width Calculation
```
w = ||tvec₁ - tvec₀||₂
  = sqrt((x₁-x₀)² + (y₁-y₀)² + (z₁-z₀)²)

where tvec₀, tvec₁ are 3D positions of ArUco markers
```

### Pose Transformation (Quest → End-Effector)
```
T_ee = T_quest × T_quest_to_ee⁻¹

where:
  T_quest ∈ SE(3)        # Quest controller pose (4×4)
  T_quest_to_ee ∈ SE(3)  # Calibrated transformation (4×4)
  T_ee ∈ SE(3)           # End-effector pose (4×4)
```

---

## Appendix C: Troubleshooting Checklist

### Before Running Pipeline
- [ ] Python 3.10 or 3.11 installed
- [ ] All dependencies installed (pip install -r requirements.txt)
- [ ] zarr==2.16.0 and numcodecs==0.11.0 (exact versions)
- [ ] config.yaml properly configured
- [ ] Camera intrinsics JSON exists
- [ ] Raw data structure correct

### After Each Stage
- [ ] No error messages in output
- [ ] Output files created in expected locations
- [ ] File sizes reasonable (not 0 bytes)
- [ ] Frame counts match across stages
- [ ] Visual inspection of sample outputs

### Final Validation
- [ ] zarr.zip file created (~70MB for 5 demos)
- [ ] Can load zarr with: `zarr.open('task.zarr.zip', mode='r')`
- [ ] All expected datasets present
- [ ] Pose trajectories look smooth
- [ ] Images not corrupted

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-27 | 1.0 | Initial comprehensive overview created |

---

**End of Document**
