# ViTaMIn-B Data Collection Pipeline

A comprehensive pipeline for collecting and processing multi-modal robotic manipulation data, including visual images, tactile sensing, and pose tracking from Meta Quest controllers.

## Overview

This pipeline enables collection of bimanual and single-hand manipulation demonstrations with:
- **Visual sensing:** Fisheye cameras (1280×800)
- **Tactile sensing:** GelSight sensors (2×1280×800 per hand)
- **Pose tracking:** Meta Quest 3 controllers via TCP
- **Gripper state:** ArUco marker-based width estimation

## Quick Start

### 1. Data Collection

```bash
# Start data collection with Quest pose tracking
python 00_get_data.py --cfg ../config/VB_task_config.yaml
# Press 'S' to start/stop recording, 'Q' to quit
```

### 2. Data Processing

```bash
# Run the automated processing pipeline
cd ..
python run_data_collection_pipeline.py
```

This will execute all processing steps automatically:
1. Image cropping and splitting
2. ArUco marker detection
3. Gripper width calculation
4. Dataset plan generation
5. Replay buffer creation

## Pipeline Scripts

### Core Pipeline (Automated)

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `00_get_data.py` | Real-time data recorder | Quest TCP + V4L2 cameras | Raw images + pose JSON |
| `01_crop_img.py` | Image splitting | 3840×800 images | Visual + tactile images |
| `04_get_aruco_pos.py` | Marker detection | Visual images | Tag detection PKL |
| `05_get_width.py` | Gripper width | ArUco detections | Width CSV |
| `07_generate_dataset_plan.py` | Trajectory processing | Pose JSON + images | Trajectory CSV + plan PKL |
| `08_generate_replay_buffer.py` | Dataset assembly | All processed data | Zarr training dataset |

### Optional Scripts

| Script | Description | When to Use |
|--------|-------------|-------------|
| `preview_cameras.py` | Camera preview | Testing camera setup |
| `convert_quest_poses.py` | Standalone pose converter | Manual trajectory processing |

### Deprecated/Alternative Versions

- `pose_only_pipeline.py` - Pose-only dataset generation

## Directory Structure

```
vitamin_b_data_collection_pipeline/
├── README.md                          # This file
├── Pipeline_overview.md               # Detailed technical documentation
│
├── Core Pipeline Scripts
├── 00_get_data.py                     # Data collection
├── 01_crop_img.py                     # Image splitting
├── 04_get_aruco_pos.py                # ArUco detection
├── 05_get_width.py                    # Gripper width
├── 07_generate_dataset_plan.py        # Trajectory processing
├── 08_generate_replay_buffer.py       # Dataset assembly
│
├── Utility Scripts
├── 01_get_pose_data_manual_save.py    # Manual pose capture
├── 05_vis_aruco_detection.py          # Detection visualization
├── preview_cameras.py                 # Camera preview
├── convert_quest_poses.py             # Pose conversion
│
└── utils/                             # Self-contained utilities
    ├── __init__.py
    ├── camera_device.py               # V4L2 camera interface
    ├── cv_util.py                     # Computer vision utilities
    ├── pose_util.py                   # Pose transformations
    ├── replay_buffer.py               # Zarr dataset interface
    ├── config_utils.py                # Configuration helpers
    └── imagecodecs_numcodecs.py       # Image compression codecs
```

## Data Flow

```
Raw Data Collection
    ↓
[00_get_data.py] → Quest poses (JSON) + Raw images (3840×800)
    ↓
Automated Processing (run_data_collection_pipeline.py)
    ↓
[01_crop_img] → Visual (1280×800) + 2× Tactile (1280×800)
    ↓
[04_get_aruco] → ArUco marker detections (PKL)
    ↓
[05_get_width] → Gripper widths (CSV)
    ↓
[07_generate_plan] → Trajectory CSV + Dataset plan (PKL)
    ↓
[08_generate_buffer] → Training dataset (Zarr)
```

## Configuration

The pipeline uses YAML configuration files. Example:

```yaml
task:
  name: "my_task"
  type: "bimanual"  # or "single"

recorder:
  output: "./data"
  camera:
    width: 3840
    height: 800
    format: "MJPG"
  camera_paths:
    left_hand: "/dev/video0"
    right_hand: "/dev/video2"

output_train_data:
  visual_out_res: [640, 480]
  tactile_out_res: [160, 120]
  use_tactile_img: true
  use_tactile_pc: false  # Set true to use tactile point clouds
  use_ee_pose: true
  tx_quest_2_ee_left_path: "./tf_cali_result/quest_2_ee_left_hand.npy"
  tx_quest_2_ee_right_path: "./tf_cali_result/quest_2_ee_right_hand.npy"
```

See `../config/VB_task_config.yaml` for full configuration options.

## Output Dataset Format

The final dataset (`{task_name}.zarr.zip`) contains:

### Pose Data (per robot)
- `robot{i}_eef_pos`: End-effector position (N×T×3)
- `robot{i}_eef_rot_axis_angle`: End-effector rotation (N×T×3)
- `robot{i}_gripper_width`: Gripper opening (N×T×1)

### Image Data (per camera)
- `camera{i}_rgb`: Visual images (N×T×H×W×3)
- `camera{i}_left_tactile`: Left tactile images (N×T×H×W×3)
- `camera{i}_right_tactile`: Right tactile images (N×T×H×W×3)

### Point Cloud Data (optional)
- `camera{i}_left_tactile_points`: Left tactile points (N×T×256×3)
- `camera{i}_right_tactile_points`: Right tactile points (N×T×256×3)

Where:
- N = number of episodes
- T = episode length (frames)
- H×W = image resolution

## Requirements

### Hardware
- V4L2-compatible USB cameras (3 per hand: 1 fisheye + 2 GelSight)
- Meta Quest 3 with custom tracking app
- USB connection or WiFi for Quest communication

### Software Dependencies
```bash
# Core
numpy
scipy
opencv-python
omegaconf

# Camera
v4l2
pyudev

# Dataset
zarr
numcodecs
imagecodecs

# Optional
pandas  # For legacy CSV processing
```

## Coordinate Systems

### Quest Pose Transformation

The pipeline performs critical coordinate transformations:

1. **Unity (Quest) → Right-Handed**
   - Unity: X=right, Y=up, Z=forward
   - Target: X=right, Y=forward, Z=up
   - Applied in `07_generate_dataset_plan.py`

2. **Hand-Wrist Mapping**
   - Left hand device uses RIGHT Quest controller
   - Right hand device uses LEFT Quest controller
   - This swap is intentional due to physical mounting

3. **Quest → End-Effector**
   - Optional transformation using calibration matrices
   - Applied in `08_generate_replay_buffer.py`

See `Pipeline_overview.md` for detailed transformation mathematics.

## Troubleshooting

### Camera Issues
```bash
# List available cameras
python preview_cameras.py

# Check camera permissions
sudo chmod 666 /dev/video*
```

### Quest Connection
```bash
# For USB connection, forward TCP port
adb forward tcp:7777 tcp:7777
adb forward tcp:50010 tcp:50010

# Test connection
python 01_get_pose_data_manual_save.py
```

### ArUco Detection
- Ensure good lighting (avoid shadows)
- Check marker sizes match config
- Visualize detections: `python 05_vis_aruco_detection.py`
- Adjust threshold parameters in config if needed

### Processing Errors
```bash
# Run individual steps for debugging
python 01_crop_img.py --cfg ../config/VB_task_config.yaml
python 04_get_aruco_pos.py --cfg ../config/VB_task_config.yaml
# ... etc
```

## Best Practices

1. **Before Collection:**
   - Calibrate cameras (fisheye intrinsics)
   - Calibrate Quest-to-EE transformation
   - Test ArUco detection with sample images
   - Verify TCP connections

2. **During Collection:**
   - Monitor FPS (should be 25-30 Hz)
   - Keep demos >30 frames
   - Maintain consistent lighting
   - Check real-time status display

3. **After Collection:**
   - Run automated pipeline first
   - Check ArUco detection rate (>80% recommended)
   - Verify gripper width interpolation
   - Inspect final dataset with Zarr tools

## Development

### Adding New Processing Steps

1. Create script `XX_step_name.py` in pipeline folder
2. Follow naming convention: `XX_` prefix with step number
3. Use `--cfg` argument for configuration
4. Add to `run_data_collection_pipeline.py` if part of core pipeline
5. Update this README and `Pipeline_overview.md`

### Utility Functions

All utilities are in the `utils/` folder:
- Import: `from utils.module import function`
- Keep utilities self-contained
- Document with docstrings

## Documentation

- **README.md** (this file): Quick reference and usage guide
- **Pipeline_overview.md**: Detailed technical documentation
  - Complete architecture description
  - Data flow diagrams
  - Pose transformation mathematics
  - File format specifications
  - Troubleshooting guide

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{vitaminb2024,
  title={ViTaMIn-B: Vision-Touch-Action Model for Bimanual Manipulation},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

[Specify license here]

## Contact

For issues and questions:
- GitHub Issues: [repository URL]
- Email: [contact email]

---

**Version:** 1.0  
**Last Updated:** January 26, 2026  
**Maintainers:** VB-VLA Team
