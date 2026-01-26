# ViTaMIn-B Data Collection Pipeline - Technical Overview

**Author:** Claude (AI Assistant)  
**Date:** January 26, 2026  
**Purpose:** Comprehensive technical documentation of the ViTaMIn-B data collection pipeline architecture, data flow, and pose transformations

> **Note:** For quick start and usage instructions, see [README.md](README.md). This document provides in-depth technical details.

---

## Folder Organization

This pipeline folder is **self-contained** with all necessary utilities included:

```
vitamin_b_data_collection_pipeline/
├── README.md                      # Quick start guide and usage
├── Pipeline_overview.md           # This file - technical details
│
├── Core Scripts (run by automated pipeline)
├── 00_get_data.py                 # Data collection (manual)
├── 01_crop_img.py                 # Step 1: Image splitting
├── 04_get_aruco_pos.py            # Step 2: ArUco detection  
├── 05_get_width.py                # Step 3: Gripper width
├── 07_generate_dataset_plan.py    # Step 5: Trajectory processing
├── 08_generate_replay_buffer.py   # Step 6: Dataset assembly
│
├── Optional Scripts
├── 06_get_tac_point.py            # Step 4: Tactile points (optional)
├── 01_get_pose_data_manual_save.py
├── 05_vis_aruco_detection.py
├── preview_cameras.py
├── convert_quest_poses.py
├── pose_only_pipeline.py
│
└── utils/                         # Self-contained utilities package
    ├── __init__.py
    ├── camera_device.py           # V4L2 camera interface
    ├── cv_util.py                 # Computer vision functions
    ├── pose_util.py               # Pose transformations
    ├── replay_buffer.py           # Zarr dataset interface
    ├── config_utils.py            # Configuration helpers
    └── imagecodecs_numcodecs.py   # Image compression codecs
```

All imports from `utils.*` now reference the local utilities package within this folder, making the pipeline independent of external dependencies.

---

## Table of Contents
1. [Pipeline Architecture](#pipeline-architecture)
2. [Data Collection Scripts](#data-collection-scripts)
3. [Processing Scripts](#processing-scripts)
4. [Coordinate System Transformations](#coordinate-system-transformations)
5. [Data Flow Diagram](#data-flow-diagram)
6. [Key Technical Details](#key-technical-details)

---

## Pipeline Architecture

The ViTaMIn-B data collection pipeline consists of three main stages:

### Stage 1: Data Acquisition (Real-time)
- **00_get_data.py**: Combined image and pose recorder
- **01_get_pose_data_manual_save.py**: Manual pose snapshot capture (optional, for calibration)

### Stage 2: Data Processing (Post-recording)
**Executed via `run_data_collection_pipeline.py` in sequence:**
1. **01_crop_img.py**: Image cropping and splitting (3840×800 → 3×1280×800)
2. **04_get_aruco_pos.py**: ArUco marker detection on visual images
3. **05_get_width.py**: Gripper width calculation from ArUco markers
4. **06_get_tac_point.py**: Tactile point cloud extraction (**OPTIONAL** - not in default pipeline)
5. **07_generate_dataset_plan.py**: Episode metadata and trajectory processing
6. **08_generate_replay_buffer.py**: Final dataset assembly in Zarr format

**Note:** Script 06_get_tac_point.py is NOT included in the default `run_data_collection_pipeline.py` execution. It must be run separately if tactile point clouds are needed (`use_tactile_pc: true` in config).

---

## Data Collection Scripts

### 00_get_data.py - Main Data Recorder

**Purpose:** Simultaneously captures images from V4L2 cameras and Quest pose data via TCP socket.

#### Key Components:

1. **StatusBoard**
   - Real-time FPS monitoring for cameras and pose
   - Split-screen terminal display
   - Demo status tracking

2. **BatchSaver**
   - Saves Quest pose data in batches (5000 frames per JSON file)
   - Session management with timestamps
   - Metadata tracking

3. **ControlClient**
   - TCP client for Quest communication (default: localhost:50010)
   - Sends demo start/stop signals
   - Sends FPS updates back to Quest for monitoring

4. **QuestPoseReceiver** (Thread)
   - TCP socket connection (default: localhost:7777)
   - Receives JSON-formatted pose data from Quest
   - Automatic recording on connection
   - Data structure:
     ```json
     {
       "head_pose": {...},
       "left_wrist": {
         "position": {"x": float, "y": float, "z": float},
         "rotation": {"x": float, "y": float, "z": float, "w": float}
       },
       "right_wrist": {...},
       "timestamp": float
     }
     ```

5. **CameraWorker** (Thread per camera)
   - Reads frames from V4L2 cameras using custom V4L2Camera class
   - Buffered queue (max 64 frames)
   - Auto-exposure, white balance, brightness, gain, gamma control
   - Captures RGB images at configurable resolution (typically 3840x800)

6. **RecordingSession**
   - Manages demo lifecycle (start/stop)
   - Saves images as JPG files: `{camera_name}_{frame_id}.jpg`
   - Generates timestamp CSV files: `{camera_name}_timestamps.csv`
     - Columns: `frame_id`, `ram_time`, `filename`
     - `ram_time` format: `YYYYmmdd_HHMMSS_ffffff`

7. **CombinedRunner**
   - Unified control with keyboard input
   - Press 'S' to start/stop recording
   - Press 'Q' to quit
   - Synchronizes image and pose recording

#### Data Output Structure:
```
data/
└── {task_name}/
    ├── demos/
    │   └── demo_{type}_{timestamp}/
    │       ├── {hand}_hand_img/              # Raw images (3840x800)
    │       │   ├── {hand}_hand_0.jpg
    │       │   └── ...
    │       └── {hand}_hand_timestamps.csv    # Image timestamps
    └── all_trajectory/
        ├── quest_poses_{session}_part001.json
        ├── quest_poses_{session}_part002.json
        └── session_summary_{session}.json
```

---

### 01_get_pose_data_manual_save.py - Manual Pose Capture

**Purpose:** Interactive tool for capturing specific pose snapshots.

#### Features:
- Press 's' + Enter to save current pose
- Press 'q' + Enter to quit
- Saves snapshots to `.npy` file
- Useful for calibration and validation

---

## Processing Scripts

### 01_crop_img.py - Image Splitting

**Purpose:** Crops raw 3840x800 images into three 1280x800 sections: visual + 2 tactile sensors.

#### Process:
1. **Input:** `{hand}_hand_img/` folder (3840x800 JPG images)
2. **Output:** Three folders per hand:
   - `{hand}_hand_visual_img/` - Middle 1280 pixels (fisheye camera)
   - `{hand}_hand_left_tactile_img/` - Left 1280 pixels (left GelSight)
   - `{hand}_hand_right_tactile_img/` - Right 1280 pixels (right GelSight)

3. **Rotation Logic:**
   - Left tactile sensor: Rotated 180°
   - Visual (fisheye): No rotation
   - Right tactile sensor: No rotation

4. **Parallelization:** Uses multiprocessing for faster batch processing

#### Why This Step?
- Separates visual and tactile modalities
- Enables independent processing of each sensor
- Reduces file I/O by processing once

---

### 04_get_aruco_pos.py - ArUco Marker Detection

**Purpose:** Detects ArUco markers on gripper fingers to enable gripper width calculation.

#### Key Features:

1. **Detection Parameters (cv_util.py):**
   - Uses OpenCV's ArUco detector with fisheye camera intrinsics
   - Configurable threshold parameters for robust detection
   - Sub-pixel corner refinement for accuracy
   - Supports DICT_4X4_50 dictionary (configurable)

2. **Fisheye Calibration:**
   - Loads camera intrinsics from JSON (Kannala-Brandt model)
   - Converts resolution if needed
   - Parameters: K (camera matrix), D (distortion coefficients)

3. **Detection Process:**
   - Iterates through `{hand}_hand_visual_img/` folders
   - Detects markers in each frame
   - Computes 3D pose using PnP algorithm with fisheye model
   - Handles missing detections gracefully

4. **Output Format (tag_detection_{hand}.pkl):**
   ```python
   [
     {
       'frame_idx': int,
       'time': float,  # From timestamps CSV or frame_idx/30.0
       'tag_dict': {
         marker_id: {
           'corners': [[x, y], ...],  # 2D pixel coordinates
           'rvec': np.array,          # Rotation vector
           'tvec': np.array,          # Translation vector
           'marker_size': float
         }
       }
     },
     ...
   ]
   ```

5. **Timestamp Synchronization:**
   - Reads `{hand}_hand_timestamps.csv`
   - Uses `ram_time` column for accurate frame timing
   - Falls back to frame index / 30.0 if timestamps unavailable

#### Parallelization:
- Concurrent processing with ThreadPoolExecutor
- Each demo processed independently
- Configurable max_workers (default: 4)

---

### 05_get_width.py - Gripper Width Calculation

**Purpose:** Calculates gripper width from ArUco marker positions.

#### Algorithm:

1. **Marker Pair Selection:**
   - Left gripper: Uses two markers (left_id, right_id) on left fingers
   - Right gripper: Uses two markers on right fingers
   - Marker IDs configured per task

2. **Width Calculation (cv_util.py - get_gripper_width):**
   ```python
   # Pseudo-code
   tag1_pos = tag_dict[left_id]['tvec']
   tag2_pos = tag_dict[right_id]['tvec']
   width = euclidean_distance(tag1_pos, tag2_pos)
   ```

3. **Interpolation (interpolate_widths_np):**
   - Uses scipy.interpolate.interp1d for linear interpolation
   - Fills missing values (NaN or ≤0) with interpolated values
   - Extrapolates beyond valid range
   - Default value for all-NaN case: 0.05 meters

4. **Output Format (gripper_width_{hand}.csv):**
   ```csv
   frame,width
   0,0.08234567
   1,0.08123456
   ...
   ```

#### Quality Metrics:
- Reports valid detection rate per demo
- Identifies frames with missing/invalid widths
- Overall statistics across all demos

---

### 6. Tactile Point Cloud Extraction

**Purpose:** Converts GelSight tactile images to 3D point clouds representing contact geometry.

**IMPORTANT:** This step is **OPTIONAL** and **NOT included** in the default `run_data_collection_pipeline.py`. 
- Must be run manually: `python 06_get_tac_point.py --cfg config.yaml`
- Required only if `use_tactile_pc: true` in configuration
- Should be run **before** step 07_generate_dataset_plan.py
- If skipped, the pipeline will only process tactile images, not point clouds

#### Algorithm Overview:

1. **Gel Surface Detection:**
   - Converts RGB → grayscale → binary threshold
   - Threshold values are sensor-specific (25-40 lower bound)
   - Gaussian blur (3x3, sigma=50) for noise reduction
   - Finds contours using RETR_EXTERNAL
   - Filters contours by length (>500 points)

2. **Contour Smoothing:**
   - Applies moving average filter (window size: 4)
   - Rejects outliers (>5 pixels from average)

3. **Camera Center Calibration:**
   - First frame establishes center reference
   - Computes pixel center from all contour points
   - Calculates deviation from optical center
   - Projects to physical coordinates at z=30mm (gel bottom)

4. **3D Reconstruction:**
   - Uses pinhole camera model with intrinsics K:
     ```
     K = [[506.36, 0, 309.94],
          [0, 506.54, 239.48],
          [0, 0, 1]]
     ```
   - Intersects ray from camera through pixel with CAD-defined gel side plane
   - Two plane equations (left/right sides):
     - Normal vectors: (nx, ny, nz) from CAD
     - Reference point: (15mm, 16.5mm, 30mm)
   - Filters invalid points (|z| > 63mm)

5. **Farthest Point Sampling (FPS):**
   - Samples fixed number of points (fps_num_points, default: 256)
   - Greedy algorithm: iteratively selects farthest point from existing set
   - Pads with zeros if < fps_num_points
   - Ensures consistent point cloud size for neural network input

6. **Output Format:**
   ```
   tactile_points/{usage_name}_points.npy
   Shape: (num_frames, fps_num_points, 3)
   Dtype: float32
   ```

#### Coordinate System:
- **Optical center system:** Origin at camera optical center
- **Frame center system:** Origin at gel pad center (calibrated)
- **Transformation:** err_x, err_y computed from first frame calibration

#### Parallelization:
- ProcessPoolExecutor for video-level parallelism
- Each tactile video (left/right per hand) processed independently

---

## Coordinate System Transformations

### Critical: Quest-to-Robot Coordinate Transformation

The pipeline handles **multiple coordinate system transformations**:

### 1. Unity (Quest) → Right-Handed Coordinate System

**Location:** `07_generate_dataset_plan.py::compute_rel_transform()`

#### Unity Left-Handed System:
- X: Right
- Y: Up
- Z: Forward (into scene)

#### Target Right-Handed System:
- X: Right (unchanged)
- Y: Forward (was Unity Z)
- Z: Up (was Unity Y)

#### Transformation Process:

```python
def compute_rel_transform(pose: np.ndarray) -> tuple:
    # pose = [x, y, z, q_x, q_y, q_z, q_w] in Unity coordinates
    
    # Step 1: Swap Y and Z axes for position
    pose[:3] = [pose[0], pose[2], pose[1]]  # [x, z, y]
    
    # Step 2: Define swap matrix Q
    Q = [[1, 0, 0],
         [0, 0, 1],   # Y → Z
         [0, 1, 0]]   # Z → Y
    
    # Step 3: Transform rotation matrix
    rot_unity = Rotation.from_quat(pose[3:]).as_matrix()
    rot_rh = Q @ rot_unity @ Q.T
    rel_rot = Rotation.from_matrix(rot_rh)
    
    # Step 4: Transform position (relative to world frame)
    rel_pos = Q @ (pose[:3] - world_frame[:3])
    
    return rel_pos, rel_quat
```

**Key Insight:** This transformation is applied **once** during trajectory CSV generation, not during final replay buffer creation.

---

### 2. Hand-Wrist Mapping (CRITICAL!)

**Location:** `07_generate_dataset_plan.py::_ensure_hand_trajectory_csv()`

#### Physical Setup:
- **Left hand device** is tracked by **right Quest controller**
- **Right hand device** is tracked by **left Quest controller**

#### Mapping Logic:
```python
if hand == 'left':
    quest_wrist_key = 'right_wrist'  # Left device uses right Quest tracker
elif hand == 'right':
    quest_wrist_key = 'left_wrist'   # Right device uses left Quest tracker
```

**Why?** The physical mounting of Quest controllers on the robot hands requires this swap for correct correspondence.

---

### 3. Quest Pose → End-Effector Pose

**Location:** `08_generate_replay_buffer.py::main()`

#### Purpose:
Transform from Quest controller position to actual robot end-effector position.

#### Transformation:
```python
if use_ee_pose:
    tx_quest_2_ee = np.load(tx_quest_2_ee_path)  # 4x4 transformation matrix
    
    # Convert pose to homogeneous matrix
    quest_mat = pose_to_mat(quest_pose)  # Nx4x4
    
    # Apply transformation: T_world_ee = T_world_quest @ inv(T_quest_ee)
    eef_mat = quest_mat @ np.linalg.inv(tx_quest_2_ee)
    
    # Convert back to pose representation
    eef_pose = mat_to_pose(eef_mat)
```

#### Calibration Files:
- `quest_2_ee_left_hand.npy`: Left hand transformation matrix
- `quest_2_ee_right_hand.npy`: Right hand transformation matrix
- These are pre-calibrated 4x4 homogeneous transformation matrices

---

### 4. Pose Representation Formats

The pipeline uses multiple pose representations:

#### Format 1: Position + Quaternion (Quest JSON)
```python
{
  "position": {"x": float, "y": float, "z": float},
  "rotation": {"x": float, "y": float, "z": float, "w": float}
}
# 7 values total: [x, y, z, q_x, q_y, q_z, q_w]
```

#### Format 2: Position + Rotation Vector (CSV, Replay Buffer)
```python
pose = [x, y, z, rx, ry, rz]
# 6 values: position (3) + axis-angle rotation (3)
```

#### Format 3: Homogeneous Transformation Matrix (Internal)
```python
mat = [[r11, r12, r13, tx],
       [r21, r22, r23, ty],
       [r31, r32, r33, tz],
       [0,   0,   0,   1]]
# 4x4 matrix for pose composition
```

#### Conversion Utilities (pose_util.py):
```python
# Pose (6D) ↔ Mat (4x4)
pose_to_mat(pose)  # 6D → 4x4
mat_to_pose(mat)   # 4x4 → 6D

# Position + Rotation ↔ Mat
pos_rot_to_mat(pos, rot)
mat_to_pos_rot(mat)

# Rotation formats
Rotation.from_rotvec(rotvec)  # Axis-angle
Rotation.from_quat(quat)      # Quaternion
Rotation.from_matrix(mat)     # 3x3 rotation matrix
```

---

### 5. Image Coordinate Transformations

#### Fisheye Undistortion (cv_util.py):
```python
class FisheyeRectConverter:
    # Uses Kannala-Brandt fisheye model
    # Converts fisheye image → rectified perspective image
    # cv2.fisheye.initUndistortRectifyMap(K, D, R, P, size, m1type)
```

#### ArUco 3D Localization:
```python
# Projects 2D corners → 3D pose using PnP with fisheye model
cv2.fisheye.estimatePoseSingleMarkers(
    corners, marker_size, K, D
)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1: DATA ACQUISITION (Manual)              │
│                    Run: 00_get_data.py                       │
└─────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
            [Quest via TCP]          [V4L2 Cameras]
                 │                         │
                 │ JSON pose data          │ Raw RGB images
                 │ (left_wrist,            │ (3840x800)
                 │  right_wrist,           │
                 │  head_pose)             │
                 │                         │
                 ▼                         ▼
    ┌──────────────────────┐    ┌──────────────────────┐
    │ all_trajectory/      │    │ demos/demo_xxx/      │
    │ quest_poses_*.json   │    │ {hand}_hand_img/*.jpg│
    │                      │    │ {hand}_timestamps.csv│
    └──────────────────────┘    └──────────────────────┘
                 │                         │
                 │                         │
┌─────────────────────────────────────────────────────────────┐
│         STAGE 2: DATA PROCESSING (Automated via             │
│            run_data_collection_pipeline.py)                  │
└─────────────────────────────────────────────────────────────┘
                 │                         │
                 │                         ▼
                 │              ┌────────────────────────┐
                 │              │ Step 1: 01_crop_img.py │
                 │              │ Split 3840→3×1280      │
                 │              └────────────────────────┘
                 │                         │
                 │              ┌──────────┴──────────┐
                 │              │                     │
                 │              ▼                     ▼
                 │   {hand}_hand_visual_img/  {hand}_hand_*_tactile_img/
                 │              │                     │
                 │              ▼                     │
                 │   ┌────────────────────────┐      │
                 │   │ Step 2: 04_get_aruco   │      │
                 │   │ Detect ArUco markers   │      │
                 │   └────────────────────────┘      │
                 │              │                     │
                 │              ▼                     │
                 │   tag_detection_{hand}.pkl        │
                 │              │                     │
                 │              ▼                     │
                 │   ┌────────────────────────┐      │
                 │   │ Step 3: 05_get_width   │      │
                 │   │ Compute gripper width  │      │
                 │   └────────────────────────┘      │
                 │              │                     │
                 │              ▼                     │
                 │   gripper_width_{hand}.csv        │
                 │                                    │
                 │   [Step 4: 06_get_tac_point - OPTIONAL, run manually]
                 │                                    │
                 │                                    ▼
                 │                    ┌────────────────────────┐
                 │                    │ 06_get_tac_point.py    │
                 │                    │ Extract point clouds   │
                 │                    │ (if use_tactile_pc)    │
                 │                    └────────────────────────┘
                 │                                    │
                 │                                    ▼
                 │                    tactile_points/*_points.npy
                 │                                    │
┌─────────────────────────────────────────────────────────────┐
│            STAGE 3: DATASET GENERATION (Automated)           │
└─────────────────────────────────────────────────────────────┘
                 │                                    │
                 ▼                                    │
    ┌──────────────────────────┐                     │
    │ Step 5: 07_generate_plan │◄────────────────────┘
    │ • Load quest_poses JSON  │
    │ • Transform coordinates  │
    │ • Interpolate to images  │
    │ • Generate CSV per hand  │
    └──────────────────────────┘
                 │
                 ▼
    {hand}_hand_trajectory.csv
    dataset_plan.pkl
                 │
                 ▼
    ┌──────────────────────────┐
    │ Step 6: 08_generate_buf  │
    │ • Load all data sources  │
    │ • Apply EE transform     │
    │ • Resize/process images  │
    │ • Pack into Zarr format  │
    └──────────────────────────┘
                 │
                 ▼
         {task_name}.zarr.zip
    ┌──────────────────────────┐
    │ Final Training Dataset   │
    │ • robot*_eef_pos         │
    │ • robot*_eef_rot_axis_angle│
    │ • robot*_gripper_width   │
    │ • camera*_rgb            │
    │ • camera*_*_tactile (opt)│
    │ • camera*_*_tactile_points (opt)│
    └──────────────────────────┘
```

**Pipeline Execution:**
1. **Manual:** Run `00_get_data.py` to collect data
2. **Automated:** Run `run_data_collection_pipeline.py` to process collected data
3. **Optional:** Run `06_get_tac_point.py` separately if tactile point clouds needed (before step 5)

---

## Key Technical Details

### 1. Timestamp Synchronization Strategy

**Challenge:** Synchronize 30 Hz cameras with 60+ Hz Quest pose stream.

**Solution:**
1. **Image timestamps** captured at acquisition time (ram_time in CSV)
2. **Quest pose timestamps** from Unity (timestamp_unix)
3. **Latency compensation:**
   - `visual_cam_latency`: Camera shutter delay (default: 0.0s)
   - `pose_latency`: Quest tracking delay (default: 0.0s)
4. **Interpolation:** Linear for position, SLERP for rotation
5. **Clipping:** Target times clipped to trajectory range to avoid extrapolation

#### Key Bug Fix (07_generate_dataset_plan.py):
```python
# BUGFIX: Clip target times to avoid extrapolation
clipped_times = np.clip(target_times, pose_times[0], pose_times[-1])

# Use clipped_times consistently for both position and rotation
interp_pos = np.interp(clipped_times, pose_times, pos)
slerp = Slerp(pose_times, Rotation.from_quat(quat))
interp_quat = slerp(clipped_times).as_quat()
```

---

### 2. Gripper Width Interpolation

**Challenge:** ArUco detection is not 100% reliable, causing missing width values.

**Solution (05_get_width.py::interpolate_widths_np):**
```python
def interpolate_widths_np(frames, widths):
    # 1. Find valid samples (>0 and not NaN)
    valid_mask = (~np.isnan(widths)) & (widths > 0)
    
    # 2. Fit linear interpolation on valid points
    interp = interp1d(frames[valid_mask], widths[valid_mask],
                      kind='linear', bounds_error=False,
                      fill_value='extrapolate')
    
    # 3. Fill invalid positions
    widths[~valid_mask] = interp(frames[~valid_mask])
    
    # 4. Replace remaining NaN with default (0.05m)
    widths = np.where(np.isnan(widths), 0.05, widths)
    
    return widths
```

**Quality Metrics:**
- Reports valid detection ratio per demo
- Overall statistics across dataset
- Identifies problematic demos with low detection rates

---

### 3. Trajectory CSV Format (Generated in Stage 3)

**File:** `demos/demo_xxx/pose_data/{hand}_hand_trajectory.csv`

```csv
timestamp,x,y,z,q_x,q_y,q_z,q_w
1704931234.567890,0.12345678,-0.23456789,0.34567890,0.01234567,0.02345678,0.03456789,0.99123456
...
```

**Columns:**
- `timestamp`: Unix timestamp (seconds since epoch)
- `x, y, z`: Position in right-handed coordinate system (meters)
- `q_x, q_y, q_z, q_w`: Orientation quaternion

**Generation Process:**
1. Load all `quest_poses_*.json` files from `all_trajectory/`
2. Extract appropriate wrist data (with hand-wrist swap)
3. Apply coordinate transformation (Unity → right-handed)
4. Write to CSV with NumPy for speed (vs pandas row-by-row)

**Caching:** Generated once and reused unless `force_regenerate_csv=True`

---

### 4. Replay Buffer Structure (Zarr)

**File Format:** Zarr compressed as ZIP

**Root Groups:**
- `data/`: Main data storage
- `meta/`: Episode metadata

**Data Arrays:**

#### Per-Robot Arrays (N_episodes × episode_length × dims):
```python
robot0_eef_pos               # (N, T, 3) - End-effector position
robot0_eef_rot_axis_angle    # (N, T, 3) - Rotation as axis-angle
robot0_gripper_width         # (N, T, 1) - Gripper opening
robot0_demo_start_pose       # (N, T, 6) - First pose (broadcasted)
robot0_demo_end_pose         # (N, T, 6) - Last pose (broadcasted)

# For bimanual tasks:
robot1_eef_pos               # (N, T, 3)
robot1_eef_rot_axis_angle    # (N, T, 3)
robot1_gripper_width         # (N, T, 1)
robot1_demo_start_pose       # (N, T, 6)
robot1_demo_end_pose         # (N, T, 6)
```

#### Per-Camera Arrays:
```python
# Visual cameras (fisheye)
camera0_rgb                  # (N, T, H, W, 3) - Resized & undistorted
camera1_rgb                  # (N, T, H, W, 3)

# Tactile images (if use_tactile_img=True)
camera0_left_tactile         # (N, T, H, W, 3) - Resized GelSight
camera0_right_tactile        # (N, T, H, W, 3)
camera1_left_tactile         # (N, T, H, W, 3)
camera1_right_tactile        # (N, T, H, W, 3)

# Tactile point clouds (if use_tactile_pc=True)
camera0_left_tactile_points  # (N, T, 256, 3) - FPS sampled points
camera0_right_tactile_points # (N, T, 256, 3)
camera1_left_tactile_points  # (N, T, 256, 3)
camera1_right_tactile_points # (N, T, 256, 3)
```

**Compression:**
- Images: JpegXL codec (configurable level, default: 3)
- Pose/width: No compression (small size)
- Point clouds: No compression (float32 precision needed)

**Chunking Strategy:**
- Chunk size: (1, T, ...) - One episode per chunk
- Enables efficient random episode access during training

---

### 5. Image Processing Pipeline (08_generate_replay_buffer.py)

#### Visual Images (Fisheye):
```python
1. Load raw image (1280×800)
2. Optional: Inpaint ArUco markers
   - Uses cached corner positions
   - Jitters position slightly when marker not detected
   - Inpainting radius scaled by tag_scale parameter
3. Optional: Apply fisheye mask
   - Draws circular mask (configurable radius, center, color)
   - Fills outside region with black or specified color
4. Resize to output resolution (default: 640×480)
   - Uses get_fisheye_image_transform()
   - Maintains aspect ratio with padding/cropping
5. Compress with JpegXL
```

#### Tactile Images:
```python
1. Load cropped tactile image (1280×800)
2. Resize to output resolution (default: 160×120)
   - Uses get_tactile_image_transform()
3. Compress with JpegXL
```

#### Tactile Point Clouds:
```python
1. Load pre-computed points from 06_get_tac_point.py
2. Verify shape: (fps_num_points, 3)
3. Pad with zeros if insufficient points
4. Truncate if excessive points
5. Store as float32 array (no compression)
```

---

### 6. Configuration Management

**Format:** YAML with OmegaConf

**Key Sections:**

```yaml
task:
  name: "my_task"
  type: "bimanual"  # or "single"
  single_hand_side: "left"  # if type="single"

recorder:
  output: "./data"
  control_host: "localhost"
  control_port: 50010
  camera:
    width: 3840
    height: 800
    format: "MJPG"
    auto_exposure: 1
    brightness: 0
    gain: 100
    gamma: 100
  camera_paths:
    left_hand: "/dev/video0"
    right_hand: "/dev/video2"

calculate_width:
  cam_intrinsic_json_path: "./config/fisheye_intrinsics.json"
  aruco_dict:
    predefined: "DICT_4X4_50"
  marker_size_map:
    default: 0.015  # meters
  left_hand_aruco_id:
    left_id: 0
    right_id: 1
  right_hand_aruco_id:
    left_id: 2
    right_id: 3
  max_workers: 4

tactile_point_extraction:
  fps_num_points: 256

output_train_data:
  min_episode_length: 30
  visual_out_res: [640, 480]
  tactile_out_res: [160, 120]
  use_mask: true
  use_inpaint_tag: true
  use_tactile_img: true
  use_tactile_pc: true
  compression_level: 3
  num_workers: 4
  visual_cam_latency: 0.0
  pose_latency: 0.0
  use_ee_pose: true
  tx_quest_2_ee_left_path: "./tf_cali_result/quest_2_ee_left_hand.npy"
  tx_quest_2_ee_right_path: "./tf_cali_result/quest_2_ee_right_hand.npy"
  tag_scale: 1.0
  fisheye_mask_params:
    radius: 580
    center: null  # Auto-detect
    fill_color: [0, 0, 0]
```

---

### 7. Parallelization Strategy

**Data Acquisition (00_get_data.py):**
- Main thread: UI and control
- Per-camera threads: Frame capture
- Pose receiver thread: TCP socket
- No explicit synchronization (timestamps handle alignment)

**Image Cropping (01_crop_img.py):**
- ProcessPoolExecutor with multiple workers
- Task: (demo_dir, hand) tuple
- Independent processing per task

**ArUco Detection (04_get_aruco_pos.py):**
- ThreadPoolExecutor (max_workers=4)
- Task: one hand per demo
- Thread-safe OpenCV (cv2.setNumThreads per worker)

**Width Calculation (05_get_width.py):**
- ThreadPoolExecutor
- Task: one gripper width CSV generation
- NumPy-based, CPU-efficient

**Tactile Point Extraction (06_get_tac_point.py):**
- ProcessPoolExecutor (CPU-intensive)
- Task: one tactile video
- Progress tracking with tqdm

**Dataset Plan Generation (07_generate_dataset_plan.py):**
- ProcessPoolExecutor (max_workers=8)
- Task: one demo directory
- JSON loading and CSV generation parallelized
- Results collected and printed in order

**Replay Buffer Generation (08_generate_replay_buffer.py):**
- ThreadPoolExecutor (configurable workers)
- Task: one video (image folder)
- Zarr arrays support concurrent writes to different indices

---

### 8. Error Handling and Validation

**Data Acquisition:**
- Camera connection failures logged, continue with available cameras
- Quest connection retry with timeout
- Buffer overflow protection (max 64 frames, 10MB pose buffer)

**Processing:**
- Missing files/folders: Skip demo with warning
- ArUco detection failures: Interpolate widths
- Timestamp parsing: Multiple format fallbacks
- Invalid images: Skip frame, continue processing

**Dataset Generation:**
- Minimum episode length filter (default: 30 frames)
- Validate ArUco files exist before processing
- Check pose data availability
- Verify timestamp CSV format

**Replay Buffer:**
- Verify input resolution consistency
- Check dataset array sizes match
- Validate point cloud shapes
- Confirm file writes with size check

---

## Summary of Coordinate Transformations (Critical for Understanding)

### Complete Transformation Chain:

```
Unity Quest Pose (JSON)
    ↓ [compute_rel_transform - Stage 3, Step 5]
Right-Handed Quest Pose
    ↓ [Hand-wrist swap - Stage 3, Step 5]
Correct Hand Quest Pose
    ↓ [Save to CSV - Stage 3, Step 5]
{hand}_hand_trajectory.csv
    ↓ [Interpolation to image times - Stage 3, Step 5]
Interpolated Quest Pose
    ↓ [Quest-to-EE transform - Stage 3, Step 6]
End-Effector Pose
    ↓ [Convert to axis-angle - Stage 3, Step 6]
Final Training Pose Format
    ↓ [Store in Zarr - Stage 3, Step 6]
robot*_eef_pos, robot*_eef_rot_axis_angle
```

### Key Takeaways:

1. **Unity → Right-Handed:** Applied during trajectory CSV generation (Step 5: 07_generate_dataset_plan.py)
2. **Hand-Wrist Swap:** Left device uses right_wrist, right uses left_wrist
3. **Quest → EE:** Optional transformation using calibration matrices (Step 6: 08_generate_replay_buffer.py)
4. **Pose Representation:** Final format is position + axis-angle rotation

---

## Pipeline Execution Workflow

### Manual Execution (Recommended for understanding):

```bash
# Stage 1: Data Collection (manual)
cd Data_collection/vitamin_b_data_collection_pipeline
python 00_get_data.py --cfg ../config/VB_task_config.yaml

# Stage 2 & 3: Automated Processing
cd ..
python run_data_collection_pipeline.py
# This runs steps 01, 04, 05, 07, 08 automatically

# Optional: If tactile point clouds needed
cd vitamin_b_data_collection_pipeline
python 06_get_tac_point.py --cfg ../config/VB_task_config.yaml
# Then re-run step 08:
python 08_generate_replay_buffer.py --cfg ../config/VB_task_config.yaml
```

### What `run_data_collection_pipeline.py` Does:

1. **Prompts user** to confirm GelSight configuration
2. **Executes 5 steps sequentially:**
   - 01_crop_img.py
   - 04_get_aruco_pos.py
   - 05_get_width.py
   - 07_generate_dataset_plan.py
   - 08_generate_replay_buffer.py
3. **Stops on error** - no subsequent steps run if one fails
4. **Passes config file** to all steps via `--cfg` argument
5. **Sets PYTHONPATH** to ensure imports work correctly

### Individual Step Execution:

```bash
# Each step can be run independently with:
python {script_name}.py --cfg ../config/VB_task_config.yaml

# Example - regenerate trajectory CSVs:
python 07_generate_dataset_plan.py --cfg ../config/VB_task_config.yaml --force-regenerate-csv
```

---

## Best Practices and Recommendations

### 1. Calibration
- Calibrate fisheye camera intrinsics with GoPro calibration tool
- Calibrate Quest-to-EE transformation using manual waypoint method
- Verify ArUco marker sizes match physical dimensions
- Test tactile camera intrinsics on flat surface

### 2. Data Collection
- Ensure stable TCP connection (use `adb forward` for Quest via USB)
- Monitor FPS during recording (should be 25-30 Hz for cameras)
- Keep demos >30 frames for training viability
- Check lighting conditions for ArUco detection

### 3. Processing
- **Run `run_data_collection_pipeline.py`** for standard processing (steps 01, 04, 05, 07, 08)
- **Manually run `06_get_tac_point.py`** only if tactile point clouds are needed
- Verify ArUco detection rate (>80% recommended)
- Check gripper width interpolation quality
- Inspect tactile point clouds for outliers (if using)

### 4. Pipeline Execution Order
- **CRITICAL:** If using tactile point clouds, run 06_get_tac_point.py **before** the automated pipeline or at least before step 08
- The automated pipeline (run_data_collection_pipeline.py) does NOT include tactile point cloud extraction
- Configure `use_tactile_pc: true/false` in config based on whether you ran step 06

### 5. Troubleshooting
- If ArUco detection fails: Adjust threshold parameters in config
- If gripper widths are wrong: Check marker_size_map in config
- If pose trajectories jump: Check timestamp synchronization
- If tactile points are noisy: Adjust camera intrinsics or threshold
- If pipeline stops: Check error messages - each step must complete before next begins

---

## Appendix A: Automated Pipeline Execution

### run_data_collection_pipeline.py

**Purpose:** Automates the post-collection processing pipeline from raw data to training-ready dataset.

**Location:** `Data_collection/run_data_collection_pipeline.py`

#### Execution Steps (in order):

1. **01_crop_img.py**
   - Splits raw 3840×800 images into visual + tactile components
   - Parallel processing for speed
   - Creates: `{hand}_hand_visual_img/`, `{hand}_hand_left_tactile_img/`, `{hand}_hand_right_tactile_img/`

2. **04_get_aruco_pos.py**
   - Detects ArUco markers on gripper fingers
   - Runs on visual images only
   - Creates: `tag_detection_{hand}.pkl`

3. **05_get_width.py**
   - Calculates gripper width from marker positions
   - Interpolates missing values
   - Creates: `gripper_width_{hand}.csv`

4. **07_generate_dataset_plan.py**
   - Loads Quest pose JSON files
   - Applies coordinate transformations
   - Generates trajectory CSVs
   - Creates episode metadata
   - Creates: `{hand}_hand_trajectory.csv`, `dataset_plan.pkl`

5. **08_generate_replay_buffer.py**
   - Loads all processed data
   - Applies end-effector transformation
   - Resizes/processes images
   - Assembles final Zarr dataset
   - Creates: `{task_name}.zarr.zip`

#### Features:

- **Sequential execution:** Each step must complete successfully before next begins
- **Error handling:** Pipeline stops on first error, reports exit code
- **Environment setup:** Automatically sets PYTHONPATH for correct imports
- **Config propagation:** Passes same config file to all steps
- **User confirmation:** Prompts to verify GelSight configuration before starting

#### What's NOT Included:

- **06_get_tac_point.py** - Tactile point cloud extraction must be run manually if needed
  - Run separately if `use_tactile_pc: true` in config
  - Should run before step 08 if point clouds are desired

#### Usage:

```bash
cd Data_collection
python run_data_collection_pipeline.py
# Uses default config: ./config/VB_task_config.yaml
```

Or modify the script to use a different config file:
```python
if __name__ == "__main__":
    config_file = "./config/my_custom_config.yaml"
    run_pipeline(config_file)
```

#### Error Recovery:

If a step fails:
1. Check the error message and exit code
2. Fix the issue (e.g., missing files, incorrect config)
3. Re-run the entire pipeline, or
4. Run individual steps manually from the failure point:
```bash
cd vitamin_b_data_collection_pipeline
python {failed_step}.py --cfg ../config/VB_task_config.yaml
# Continue with subsequent steps...
```

---

## Appendix B: File Formats

### A. Quest Pose JSON
```json
{
  "head_pose": {
    "position": {"x": 0.0, "y": 1.6, "z": 0.0},
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
  },
  "left_wrist": {
    "position": {"x": -0.3, "y": 1.2, "z": 0.5},
    "rotation": {"x": 0.1, "y": 0.0, "z": 0.0, "w": 0.99}
  },
  "right_wrist": {
    "position": {"x": 0.3, "y": 1.2, "z": 0.5},
    "rotation": {"x": -0.1, "y": 0.0, "z": 0.0, "w": 0.99}
  },
  "timestamp": 1704931234.567890,
  "timestamp_unix": 1704931234.567890,
  "timestamp_readable": "2025.01.11_12.34.56.567890"
}
```

### B. Timestamp CSV
```csv
frame_id,ram_time,filename
0,20250111_123456_000000,left_hand_0.jpg
1,20250111_123456_033333,left_hand_1.jpg
2,20250111_123456_066666,left_hand_2.jpg
```

### C. ArUco Detection Pickle
```python
[
  {
    'frame_idx': 0,
    'time': 1704931234.567890,
    'tag_dict': {
      0: {
        'corners': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        'rvec': np.array([rx, ry, rz]),
        'tvec': np.array([tx, ty, tz]),
        'marker_size': 0.015
      },
      1: { ... }
    }
  },
  ...
]
```

### D. Dataset Plan Pickle
```python
[
  {
    'episode_timestamps': np.array([t0, t1, ..., tN]),
    'grippers': [
      {
        'quest_pose': np.array([[x,y,z,rx,ry,rz], ...]),  # (N, 6)
        'gripper_width': np.array([w0, w1, ..., wN]),
        'demo_start_pose': np.array([x,y,z,rx,ry,rz]),
        'demo_end_pose': np.array([x,y,z,rx,ry,rz])
      },
      ...  # One per hand
    ],
    'cameras': [
      {
        'image_folder': 'demos/demo_xxx/left_hand_visual_img',
        'video_start_end': (0, 120),
        'usage_name': 'left_visual',
        'hand_position_idx': 0
      },
      ...
    ],
    'demo_mode': 'bimanual',
    'demo_name': 'demo_bimanual_20250111_123456',
    'n_frames': 120,
    'fps': 30.0
  },
  ...
]
```

---

## Conclusion

The ViTaMIn-B data collection pipeline is a sophisticated system that:

1. **Captures** multi-modal data (visual, tactile images, tactile point clouds, pose) in real-time
2. **Processes** raw sensor data with robust error handling and interpolation
3. **Transforms** coordinate systems correctly for robot control
4. **Generates** training-ready datasets in efficient Zarr format

The most critical aspects for correct operation are:
- Understanding the hand-wrist swap in Quest tracking
- Properly applying coordinate transformations (Unity → right-handed)
- Maintaining timestamp synchronization across modalities
- Ensuring calibration files (camera intrinsics, Quest-to-EE) are accurate

This pipeline enables collection of large-scale bimanual manipulation datasets with rich sensory modalities for training vision-language-action models.

### Self-Contained Design

As of January 2026, this pipeline folder is **self-contained**:
- All required utilities are in the local `utils/` package
- No external dependencies on parent `utils/` folder
- Can be copied/moved independently
- All imports reference local utilities: `from utils.module import function`

This design ensures:
- ✅ **Portability:** Folder can be used standalone
- ✅ **Clarity:** All dependencies are explicit and local
- ✅ **Maintainability:** Changes don't affect other parts of the codebase
- ✅ **Reproducibility:** Complete pipeline in one place

For usage instructions, troubleshooting, and quick reference, see [README.md](README.md).

---

**Document Version:** 2.0  
**Last Updated:** January 26, 2026  
**Maintainer:** VB-VLA Team  
**Changes in v2.0:**
- Reorganized folder structure for self-contained operation
- Added local utils/ package with all required utilities
- Removed deprecated files (00_get_data_old.py, 00_get_data_gopro.py)
- Created comprehensive README.md for quick reference
- Clarified automated vs manual pipeline execution

