#!/usr/bin/env python3
"""
Generate dataset plan by synchronizing all data modalities.
生成数据集计划,同步所有数据模态

This script is the core of the pipeline, combining:
该脚本是管道的核心,组合以下内容:

1. Image timestamps from visual and tactile cameras / 视觉和触觉相机的图像时间戳
2. Pose trajectories from Quest controllers / Quest控制器的姿态轨迹
3. Gripper widths from ArUco marker detection / ArUco标记检测的抓手宽度
4. Coordinate system transformations (Unity→right-handed) / 坐标系转换(Unity→右手系)

Key features / 关键特性:
- Temporal synchronization with latency compensation / 带延迟补偿的时间同步
- Trajectory interpolation (linear for position, SLERP for rotation) / 轨迹插值(位置线性,旋转SLERP)
- Multi-processing for parallel demo processing / 多进程并行处理demos
- Support for both single-hand and bimanual tasks / 支持单手和双手任务

The output dataset_plan.pkl contains all synchronized data ready for
final zarr dataset generation.
输出dataset_plan.pkl包含所有同步数据,可用于最终zarr数据集生成
"""
import sys
import os
import argparse
import pickle
import json
import csv
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pose_util import mat_to_pose


def compute_rel_transform(pose: np.ndarray) -> tuple:
    """
    Coordinate system transformation: Unity left-handed → right-handed
    坐标系转换:Unity左手系 → 右手系
    
    Unity coordinate system: X right, Y up, Z forward
    Right-handed system: X right, Y forward, Z up
    Transformation method: Swap Y and Z axes
    
    Unity坐标系: X右 Y上 Z前
    右手坐标系: X右 Y前 Z上
    转换方法: 交换Y和Z轴
    
    Args:
        pose: [x, y, z, qx, qy, qz, qw] Unity coordinates / Unity坐标
    
    Returns:
        (position, quaternion) in right-handed system / 右手坐标系中的(位置,四元数)
    
    Note / 注意:
        This function correctly transforms rotation using the rotation matrix method.
        Simple component swapping of quaternions does NOT work!
        该函数使用旋转矩阵方法正确转换旋转
        简单地交换四元数分量是不正确的!
    """
    # Position: [x, y, z] -> [x, z, y] (swap Y and Z) / 位置:[x, y, z] -> [x, z, y] (交换Y和Z)
    pos = np.array([pose[0], pose[2], pose[1]], dtype=float)
    
    # Rotation: Requires proper transformation, not simple component swapping
    # Use rotation matrix Q for coordinate system transformation
    # 旋转:需要正确变换,不能简单交换四元数分量
    # 使用旋转矩阵Q进行坐标系变换
    
    # Q swaps y and z axes: [x, y, z] -> [x, z, y]
    # Q交换y和z轴: [x, y, z] -> [x, z, y]
    Q = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0]], dtype=float)
    
    # Convert Unity quaternion to rotation matrix / 将Unity的四元数转换为旋转矩阵
    rot = Rotation.from_quat(pose[3:7]).as_matrix()
    
    # Apply coordinate system transformation: Q @ rot @ Q.T
    # This correctly transforms rotation from Unity to right-handed system
    # 应用坐标系变换: Q @ rot @ Q.T
    # 这正确地将Unity坐标系中的旋转转换到右手坐标系
    transformed_rot = Q @ rot @ Q.T
    
    # Convert back to quaternion / 转换回四元数
    quat = Rotation.from_matrix(transformed_rot).as_quat()
    
    return pos, quat


def detect_demo_mode(demo_dir: Path):
    """
    Detect demo mode (single-hand or bimanual) by checking image folders.
    通过检查图像文件夹检测demo模式(单手或双手)
    
    Args:
        demo_dir: Path to demo directory / demo目录路径
    
    Returns:
        (mode, hands): mode is 'single' or 'bimanual', hands is list of hand names
        (模式, 手列表): 模式为'single'或'bimanual',手列表为手名称列表
    
    Note / 注意:
        Compatible with both old and new folder naming conventions
        兼容新旧两种文件夹命名约定
    """
    # Check for both old and new folder naming / 兼容新旧两种文件夹命名
    left_exists = (demo_dir / 'left_hand_img').exists() or \
                  (demo_dir / 'left_hand_visual_img').exists()
    right_exists = (demo_dir / 'right_hand_img').exists() or \
                   (demo_dir / 'right_hand_visual_img').exists()
    
    if left_exists and right_exists:
        return "bimanual", ['left', 'right']
    elif left_exists:
        return "single", ['left']
    elif right_exists:
        return "single", ['right']
    
    return None, []


def check_aruco_files(demo_dir: Path, mode: str, hands: list) -> bool:
    """
    Check if ArUco detection result files exist.
    检查ArUco检测结果文件是否存在
    
    Args:
        demo_dir: Path to demo directory / demo目录路径
        mode: 'single' or 'bimanual' / 单手或双手
        hands: List of hand names / 手名称列表
    
    Returns:
        True if all required files exist / 如果所有必需文件存在则返回True
    """
    if mode == 'single':
        pkl = demo_dir / f'tag_detection_{hands[0]}.pkl'
        return pkl.exists()
    
    # Bimanual: check both hands / 双手:检查两只手
    for hand in hands:
        if not (demo_dir / f'tag_detection_{hand}.pkl').exists():
            return False
    return True


def find_image_folders(demo_dir: Path, mode: str, hands: list, use_tactile: bool):
    """
    Find all image folders including visual cameras and tactile sensors.
    查找所有图像文件夹,包括视觉相机和触觉传感器的图像
    
    Args:
        demo_dir: Path to demo directory / demo目录路径
        mode: 'single' or 'bimanual' / 单手或双手
        hands: List of hand names / 手名称列表
        use_tactile: Whether to include tactile images / 是否包含触觉图像
    
    Returns:
        Dictionary mapping usage names to folder paths / 将用途名称映射到文件夹路径的字典
    """
    folders = {}
    
    for hand in hands:
        # Prefer split visual images, otherwise use raw images
        # 优先用分割后的visual图像,否则用原始图像
        visual_folder = demo_dir / f'{hand}_hand_visual_img'
        raw_folder = demo_dir / f'{hand}_hand_img'
        img_folder = visual_folder if visual_folder.exists() else raw_folder
        
        if img_folder.exists():
            folders[f'{hand}_visual'] = img_folder
            
            # Add tactile image folders / 添加触觉图像文件夹
            if use_tactile:
                for side in ['left', 'right']:
                    tac_folder = demo_dir / f'{hand}_hand_{side}_tactile_img'
                    if tac_folder.exists():
                        folders[f'{hand}_hand_{side}_tactile'] = tac_folder
    
    return folders


def parse_timestamp(ts: str) -> float:
    """
    Parse timestamp string to Unix time.
    解析时间戳字符串为Unix时间
    
    Args:
        ts: Timestamp string / 时间戳字符串
    
    Returns:
        Unix timestamp (float) / Unix时间戳(浮点数)
    
    Supports multiple formats / 支持多种格式:
    - "%Y%m%d_%H%M%S_%f"
    - "%Y.%m.%d_%H.%M.%S.%f"
    - "%Y-%m-%d_%H-%M-%S-%f"
    """
    fmts = (
        "%Y%m%d_%H%M%S_%f",
        "%Y.%m.%d_%H.%M.%S.%f",
        "%Y-%m-%d_%H-%M-%S-%f"
    )
    
    last_err = None
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt).timestamp()
        except Exception as e:
            last_err = e

    # Try removing separators and parsing again / 尝试移除分隔符后再解析
    ts_compact = ts.replace(".", "").replace("-", "")
    try:
        return datetime.strptime(ts_compact, "%Y%m%d_%H%M%S_%f").timestamp()
    except Exception:
        raise ValueError(f"Unknown timestamp format: {ts}") from last_err


def get_image_times(demo_dir: Path, hand: str, latency: float) -> np.ndarray:
    """
    Read image timestamps from CSV with camera latency compensation.
    从CSV读取图像时间戳并补偿相机延迟
    
    Args:
        demo_dir: Path to demo directory / demo目录路径
        hand: 'left' or 'right' / 左手或右手
        latency: Camera latency in seconds / 相机延迟(秒)
    
    Returns:
        Array of timestamps with latency compensation / 带延迟补偿的时间戳数组
    """
    csv_file = demo_dir / f'{hand}_hand_timestamps.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"Timestamps not found: {csv_file}")

    ram_times = []
    with csv_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "ram_time" not in reader.fieldnames:
            raise ValueError(f"'ram_time' column missing in {csv_file}")
        for row in reader:
            ts = row.get("ram_time")
            if not ts:
                continue
            try:
                ram_times.append(parse_timestamp(ts))
            except Exception:
                continue

    if not ram_times:
        raise RuntimeError(f"No valid ram_time entries in {csv_file}")

    times = np.asarray(ram_times, dtype=np.float64)
    # Compensate for camera latency / 补偿相机延迟
    return times - latency


def ensure_split_images(demo_dir: Path, hand: str):
    """
    Check if images have been split.
    检查图像是否已分割
    
    Args:
        demo_dir: Path to demo directory / demo目录路径
        hand: 'left' or 'right' / 左手或右手
    
    Note / 注意:
        Actual image splitting is done by 01_crop_img.py
        This function just verifies split images exist
        实际分割由01_crop_img.py完成
        该函数仅验证分割后的图像是否存在
    """
    visual_dir = demo_dir / f'{hand}_hand_visual_img'
    left_dir = demo_dir / f'{hand}_hand_left_tactile_img'
    right_dir = demo_dir / f'{hand}_hand_right_tactile_img'
    
    target_dirs = [visual_dir, left_dir, right_dir]
    all_exist = all(d.exists() and d.is_dir() for d in target_dirs)
    
    if all_exist:
        all_have_images = all(sum(1 for _ in d.glob('*.jpg')) > 0 for d in target_dirs)
        if all_have_images:
            return
    
    return


def _ensure_hand_trajectory_csv(demo_dir: Path, hand: str, force_regenerate: bool = False):
    """
    Generate hand trajectory CSV file from Quest controller data.
    从Quest控制器数据生成手的轨迹CSV文件
    
    Args:
        demo_dir: Path to demo directory / demo目录路径
        hand: 'left' or 'right' / 左手或右手
        force_regenerate: Force regenerate even if file exists / 强制重新生成即使文件已存在
    
    Returns:
        Path to trajectory CSV file / 轨迹CSV文件路径
    
    Important hardware configuration / 重要硬件配置:
        Left hand device uses RIGHT Quest controller / 左手设备用右Quest控制器
        Right hand device uses LEFT Quest controller / 右手设备用左Quest控制器
        This is determined by physical installation / 这是物理安装决定的
    
    Process / 处理过程:
    1. Load all quest_poses_*.json from all_trajectory folder / 从all_trajectory文件夹加载所有quest_poses_*.json
    2. Extract correct controller data (opposite to hand side) / 提取正确的控制器数据(与手侧相反)
    3. Apply coordinate system transformation / 应用坐标系转换
    4. Save as CSV with timestamp and 7D pose / 保存为带时间戳和7D姿态的CSV
    """
    traj_file = demo_dir / 'pose_data' / f'{hand}_hand_trajectory.csv'
    
    # Handle force regeneration / 处理强制重新生成
    if force_regenerate and traj_file.exists():
        traj_file.unlink()
        print(f"  [INFO] Deleted old CSV: {traj_file.name}")
    
    if traj_file.exists() and not force_regenerate:
        return traj_file
    
    if force_regenerate:
        print(f"  [INFO] Regenerating CSV for {hand} hand...")
    else:
        print(f"  [INFO] Generating CSV for {hand} hand...")
    
    # Find quest pose JSON files / 查找quest姿态JSON文件
    task_dir = demo_dir.parent.parent
    all_traj_dir = task_dir / "all_trajectory"
    if not all_traj_dir.exists():
        raise FileNotFoundError(f"all_trajectory directory not found: {all_traj_dir}")
    
    json_files = sorted(all_traj_dir.glob("quest_poses_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No quest_poses_*.json found in {all_traj_dir}")
    
    # Map hand to Quest controller (OPPOSITE sides) / 映射手到Quest控制器(相反侧)
    if hand == 'left':
        quest_wrist_key = 'right_wrist'  # Left hand uses right controller / 左手用右控制器
    elif hand == 'right':
        quest_wrist_key = 'left_wrist'   # Right hand uses left controller / 右手用左控制器
    else:
        raise ValueError(f"Unknown hand: {hand}, expected 'left' or 'right'")
    
    # Batch load JSON files / 批量读取JSON
    print(f"  [INFO] Loading {len(json_files)} JSON files for {hand} hand...")
    all_entries = []
    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                pose_list = json.load(f)
                all_entries.extend(pose_list)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {json_path}: {e}.")
        except Exception as e:
            raise RuntimeError(f"Error reading JSON file {json_path}: {e}")
    
    # Process pose data / 处理姿态数据
    print(f"  [INFO] Processing {len(all_entries)} pose entries...")
    timestamps = []
    poses = []
    
    for entry in all_entries:
        if quest_wrist_key not in entry:
            continue
        wrist = entry[quest_wrist_key]
        pos = wrist.get("position", {})
        rot = wrist.get("rotation", {})
        
        # Extract position / 提取位置
        if isinstance(pos, dict):
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))
        else:
            try:
                x, y, z = map(float, pos[:3])
            except Exception:
                x = y = z = 0.0
        
        # Extract rotation / 提取旋转
        if isinstance(rot, dict):
            q_x = float(rot.get("x", 0.0))
            q_y = float(rot.get("y", 0.0))
            q_z = float(rot.get("z", 0.0))
            q_w = float(rot.get("w", 1.0))
        else:
            try:
                q_x, q_y, q_z, q_w = map(float, rot[:4])
            except Exception:
                q_x = q_y = q_z = 0.0
                q_w = 1.0
        
        # Apply coordinate system transformation / 坐标系转换
        rel_pos, rel_quat = compute_rel_transform(
            np.array([x, y, z, q_x, q_y, q_z, q_w], dtype=float)
        )
        
        # Extract timestamp / 提取时间戳
        ts = entry.get("timestamp_unix", entry.get("timestamp", 0.0))
        timestamps.append(ts)
        poses.append([rel_pos[0], rel_pos[1], rel_pos[2], 
                     rel_quat[0], rel_quat[1], rel_quat[2], rel_quat[3]])
    
    if not timestamps:
        raise RuntimeError(f"No valid pose data for hand '{hand}' (using {quest_wrist_key})")
    
    # Write CSV / 写入CSV
    pose_data_dir = demo_dir / "pose_data"
    pose_data_dir.mkdir(exist_ok=True)
    
    print(f"  [INFO] Writing CSV with {len(timestamps)} rows...")
    data_array = np.column_stack([timestamps, poses])
    
    header = "timestamp,x,y,z,q_x,q_y,q_z,q_w"
    np.savetxt(traj_file, data_array, delimiter=",", header=header, 
               comments="", fmt="%.9f")
    
    print(f"  [INFO] Generated trajectory CSV for {hand} hand: {traj_file}")
    return traj_file


def process_hand_trajectory(demo_dir: Path, hand: str, target_times: np.ndarray, 
                           pose_latency: float, force_regenerate: bool = False):
    """
    Process hand trajectory with temporal interpolation.
    处理单只手的轨迹并进行时间插值
    
    Args:
        demo_dir: Path to demo directory / demo目录路径
        hand: 'left' or 'right' / 左手或右手
        target_times: Target timestamps for synchronization / 用于同步的目标时间戳
        pose_latency: Pose latency in seconds / 姿态延迟(秒)
        force_regenerate: Force regenerate trajectory CSV / 强制重新生成轨迹CSV
    
    Returns:
        Dictionary containing / 包含以下内容的字典:
            - quest_pose: Interpolated 7D poses (N, 7) / 插值后的7D姿态(N, 7)
            - gripper_width: Gripper widths (N,) / 抓手宽度(N,)
            - demo_start_pose: First pose / 首姿态
            - demo_end_pose: Last pose / 末姿态
    
    Process / 处理过程:
    1. Load or generate trajectory CSV / 加载或生成轨迹CSV
    2. Compensate for pose latency / 补偿姿态延迟
    3. Sort and deduplicate timestamps / 排序和去重时间戳
    4. Interpolate poses to target times / 插值姿态到目标时间
       - Position: Linear interpolation / 位置:线性插值
       - Rotation: Spherical linear interpolation (SLERP) / 旋转:球面线性插值
    5. Load gripper widths / 加载抓手宽度
    6. Convert to 4x4 transformation matrices / 转换为4x4变换矩阵
    """
    # Ensure trajectory CSV exists / 确保轨迹CSV存在
    traj_file = _ensure_hand_trajectory_csv(demo_dir, hand, force_regenerate=force_regenerate)

    # Read trajectory CSV / 读取轨迹CSV
    try:
        data = np.genfromtxt(traj_file, delimiter=",", names=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read trajectory CSV {traj_file}: {e}.")
    if data.size == 0:
        raise RuntimeError(f"No pose samples for {hand}")

    # Extract data and compensate for latency / 提取数据并补偿延迟
    pose_times = np.asarray(data["timestamp"], dtype=np.float64) - pose_latency
    pos = np.column_stack([
        np.asarray(data["x"], dtype=np.float64),
        np.asarray(data["y"], dtype=np.float64),
        np.asarray(data["z"], dtype=np.float64),
    ])
    quat = np.column_stack([
        np.asarray(data["q_x"], dtype=np.float64),
        np.asarray(data["q_y"], dtype=np.float64),
        np.asarray(data["q_z"], dtype=np.float64),
        np.asarray(data["q_w"], dtype=np.float64),
    ])
    
    # Sort and deduplicate / 排序和去重
    order = np.argsort(pose_times)
    pose_times = pose_times[order]
    pos = pos[order]
    quat = quat[order]
    
    pose_times, unique_idx = np.unique(pose_times, return_index=True)
    pos = pos[unique_idx]
    quat = quat[unique_idx]
    
    if len(pose_times) == 0:
        raise RuntimeError(f"No pose samples for {hand}")

    # Interpolate to target times / 插值到目标时间
    if len(pose_times) == 1:
        # Single sample: repeat / 单个样本:重复使用
        interp_pos = np.repeat(pos, len(target_times), axis=0)
        interp_quat = np.repeat(quat, len(target_times), axis=0)
    else:
        # Multiple samples: interpolate / 多个样本:插值
        # Clip to data range to avoid extrapolation / 裁剪到数据范围避免外推
        clipped_times = np.clip(target_times, pose_times[0], pose_times[-1])
        
        # Position: Linear interpolation / 位置:线性插值
        interp_pos = np.column_stack([
            np.interp(clipped_times, pose_times, pos[:, i], 
                     left=pos[0, i], right=pos[-1, i])
            for i in range(3)
        ])
        
        # Rotation: Spherical linear interpolation (SLERP) / 旋转:球面线性插值
        rot = Rotation.from_quat(quat)
        slerp = Slerp(pose_times, rot)
        interp_quat = slerp(clipped_times).as_quat()
    
    # Convert to 4x4 transformation matrices / 转换为4x4变换矩阵
    rot_mats = Rotation.from_quat(interp_quat).as_matrix()
    n_frames = len(target_times)
    pose_mat = np.zeros((n_frames, 4, 4), dtype=np.float32)
    pose_mat[:, 3, 3] = 1
    pose_mat[:, :3, 3] = interp_pos
    pose_mat[:, :3, :3] = rot_mats
    
    # Read gripper width / 读取抓手宽度
    gripper_file = demo_dir / f'gripper_width_{hand}.csv'
    if not gripper_file.exists():
        raise FileNotFoundError(f"Gripper width not found: {gripper_file}")

    try:
        gdata = np.genfromtxt(gripper_file, delimiter=",", names=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read gripper width CSV {gripper_file}: {e}.")
    if gdata.size == 0:
        raise RuntimeError(f"No gripper width samples for {hand}")

    # Compatible with old and new column names / 兼容新旧列名
    if "width" in gdata.dtype.names:
        widths_arr = gdata["width"]
    elif "gripper_width" in gdata.dtype.names:
        widths_arr = gdata["gripper_width"]
    else:
        raise RuntimeError(f"No 'width' or 'gripper_width' column in {gripper_file}")

    widths = np.asarray(widths_arr, dtype=np.float32)
    if len(widths) < n_frames:
        raise ValueError(f"Not enough gripper width samples for {hand}")
    widths = widths[:n_frames]

    # Convert to 7D pose format / 转换为7D姿态格式
    quest_pose = mat_to_pose(pose_mat)
    
    return {
        "quest_pose": quest_pose,
        "gripper_width": widths,
        "demo_start_pose": quest_pose[0],
        "demo_end_pose": quest_pose[-1]
    }


def create_camera_entries(image_folders: dict, demo_dir: Path, n_frames: int, 
                         mode: str, hands: list):
    """
    Create camera entries for dataset plan.
    为数据集计划创建相机条目
    
    Args:
        image_folders: Dictionary mapping usage names to folder paths / 将用途名称映射到文件夹路径的字典
        demo_dir: Path to demo directory / demo目录路径
        n_frames: Number of frames / 帧数
        mode: 'single' or 'bimanual' / 单手或双手
        hands: List of hand names / 手名称列表
    
    Returns:
        List of camera entry dictionaries / 相机条目字典列表
    
    Each entry contains / 每个条目包含:
        - image_folder: Relative path to image folder / 图像文件夹相对路径
        - video_start_end: (start_frame, end_frame) / (起始帧, 结束帧)
        - usage_name: Camera usage identifier / 相机用途标识符
        - position: 'left' or 'right' (bimanual only) / 左或右(仅双手)
        - hand_position_idx: Hand index (0 or 1) / 手索引(0或1)
    """
    cameras = []
    hand_idx_map = {'left': 0, 'right': 1}
    
    for usage_name, img_folder in image_folders.items():
        # Use relative path from parent directory / 使用从父目录的相对路径
        rel_path = img_folder.relative_to(demo_dir.parent)
        
        entry = {
            "image_folder": str(rel_path),
            "video_start_end": (0, n_frames),
            "usage_name": usage_name,
        }
        
        if mode == "single":
            entry["hand_position_idx"] = 0
        else:
            # For bimanual, determine which hand this camera belongs to
            # 对于双手,确定该相机属于哪只手
            for hand in hands:
                if hand in usage_name:
                    entry["position"] = hand
                    entry["hand_position_idx"] = hand_idx_map[hand]
                    break
        
        cameras.append(entry)
    
    return cameras


def process_demo(demo_dir: Path, min_length: int, use_tactile: bool, 
                visual_latency: float, pose_latency: float, 
                force_regenerate_csv: bool = False):
    """
    Process a single demo to create synchronized dataset plan entry.
    处理单个demo以创建同步的数据集计划条目
    
    Args:
        demo_dir: Path to demo directory / demo目录路径
        min_length: Minimum episode length to accept / 接受的最小episode长度
        use_tactile: Whether to include tactile images / 是否包含触觉图像
        visual_latency: Visual camera latency / 视觉相机延迟
        pose_latency: Pose latency / 姿态延迟
        force_regenerate_csv: Force regenerate trajectory CSV / 强制重新生成轨迹CSV
    
    Returns:
        Dataset plan entry dictionary or None if skipped / 数据集计划条目字典或None(如果跳过)
    
    Process / 处理过程:
    1. Detect demo mode (single/bimanual) / 检测demo模式(单手/双手)
    2. Check ArUco detection files / 检查ArUco检测文件
    3. Verify image splitting / 验证图像分割
    4. Find all image folders / 查找所有图像文件夹
    5. Load and align image timestamps / 加载并对齐图像时间戳
    6. Process trajectories with interpolation / 处理轨迹并插值
    7. Create camera entries / 创建相机条目
    8. Return synchronized plan / 返回同步计划
    """
    try:
        # 1. Detect demo mode / 检测demo模式
        mode, hands = detect_demo_mode(demo_dir)
        if not mode:
            print(f"  [SKIP] {demo_dir.name}: No image folders")
            return None
        
        # 2. Check ArUco files / 检查ArUco文件
        if not check_aruco_files(demo_dir, mode, hands):
            print(f"  [SKIP] {demo_dir.name}: Missing ArUco files")
            return None
        
        # 3. Verify image splitting / 验证图像分割
        for hand in hands:
            ensure_split_images(demo_dir, hand)
        
        # 4. Find image folders / 查找图像文件夹
        image_folders = find_image_folders(demo_dir, mode, hands, use_tactile)
        if not image_folders:
            print(f"  [SKIP] {demo_dir.name}: No valid images")
            return None
        
        # 5. Load and align image timestamps / 加载并对齐图像时间戳
        image_times = {}
        for hand in hands:
            image_times[hand] = get_image_times(demo_dir, hand, visual_latency)
        
        # Align frame counts / 对齐帧数
        n_frames = min(len(times) for times in image_times.values())
        if n_frames < min_length:
            print(f"  [SKIP] {demo_dir.name}: Too short ({n_frames}<{min_length})")
            return None
        
        # Trim to minimum length / 裁剪到最小长度
        for hand in hands:
            if len(image_times[hand]) != n_frames:
                print(f"  [WARN] {demo_dir.name}: trimming {hand} from {len(image_times[hand])} to {n_frames}")
            image_times[hand] = image_times[hand][:n_frames]
        
        # Calculate FPS / 计算FPS
        ref_times = image_times[hands[0]]
        if len(ref_times) > 1:
            duration = ref_times[-1] - ref_times[0]
            fps = (len(ref_times) - 1) / duration if duration > 0 else 25.0
        else:
            fps = 25.0
        
        # 6. Process trajectories / 处理轨迹
        grippers = []
        for hand in hands:
            data = process_hand_trajectory(demo_dir, hand, image_times[hand], 
                                         pose_latency, force_regenerate=force_regenerate_csv)
            grippers.append(data)
        
        # 7. Create camera entries / 创建相机条目
        cameras = create_camera_entries(image_folders, demo_dir, n_frames, mode, hands)
        timestamps = ref_times
        
        print(f"  [OK] {demo_dir.name}: {len(hands)} hand(s), {len(cameras)} cam(s), {n_frames} frames")
        
        # 8. Return synchronized plan / 返回同步计划
        return {
            "episode_timestamps": timestamps,
            "grippers": grippers,
            "cameras": cameras,
            "demo_mode": mode,
            "demo_name": demo_dir.name,
            "n_frames": n_frames,
            "fps": fps
        }
    
    except Exception as e:
        print(f"  [ERROR] {demo_dir.name}: {e}")
        return None


def _process_demo_wrapper(args):
    """
    Wrapper function for parallel processing.
    并行处理的包装函数
    
    Args:
        args: Tuple of arguments for process_demo / process_demo的参数元组
    
    Returns:
        (demo_dir, plan, error): Result tuple / 结果元组
    """
    demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv = args
    try:
        plan = process_demo(demo_dir, min_length, use_tactile, visual_latency, 
                          pose_latency, force_regenerate_csv=force_regenerate_csv)
        return demo_dir, plan, None
    except Exception as e:
        return demo_dir, None, str(e)


def generate_plan(cfg_file: str, force_regenerate_csv: bool = False, num_workers: int = None):
    """
    Generate complete dataset plan for all demos.
    为所有demos生成完整的数据集计划
    
    Args:
        cfg_file: Path to configuration YAML file / 配置YAML文件路径
        force_regenerate_csv: Force regenerate trajectory CSVs / 强制重新生成轨迹CSV
        num_workers: Number of parallel workers (None=auto) / 并行worker数量(None=自动)
    
    Process / 处理过程:
    1. Load configuration / 加载配置
    2. Optionally delete old trajectory CSVs / 可选删除旧轨迹CSV
    3. Discover all demo directories / 发现所有demo目录
    4. Process demos in parallel or sequentially / 并行或顺序处理demos
    5. Collect results and compute statistics / 收集结果并计算统计信息
    6. Save dataset_plan.pkl / 保存dataset_plan.pkl
    
    Output / 输出:
        dataset_plan.pkl containing list of demo plans / 包含demo计划列表的dataset_plan.pkl
        Statistics printed to console / 打印到控制台的统计信息
    """
    cfg = OmegaConf.load(cfg_file)
    
    task_name = cfg.task.name
    min_length = cfg.output_train_data.min_episode_length
    use_tactile = cfg.output_train_data.get("use_tactile_img", False) or \
                  cfg.output_train_data.get("use_tactile_pc", False)
    
    visual_latency = cfg.output_train_data.get("visual_cam_latency", 0.0)
    pose_latency = cfg.output_train_data.get("pose_latency", 0.0)
    
    demos_dir = DATA_DIR / task_name / 'demos'
    output_file = DATA_DIR / task_name / 'dataset_plan.pkl'
    
    # Print configuration / 打印配置
    print(f"Task: {task_name}")
    print(f"Min length: {min_length}")
    print(f"Tactile: {use_tactile}")
    if force_regenerate_csv:
        print(f"Force regenerate CSV: YES")
    print()
    
    # Delete old CSVs if requested / 如果请求则删除旧CSV
    if force_regenerate_csv:
        print("[INFO] Deleting old trajectory CSV files...")
        demo_dirs_preview = sorted([d for d in demos_dir.glob('demo_*') if d.is_dir()])
        deleted_count = 0
        for demo_dir in demo_dirs_preview:
            pose_data_dir = demo_dir / 'pose_data'
            if pose_data_dir.exists():
                for csv_file in pose_data_dir.glob('*_hand_trajectory.csv'):
                    csv_file.unlink()
                    deleted_count += 1
        print(f"[INFO] Deleted {deleted_count} old CSV files\n")
    
    # Discover demos / 发现demos
    demo_dirs = sorted([d for d in demos_dir.glob('demo_*') if d.is_dir()])
    print(f"Found {len(demo_dirs)} demos")
    
    # Set worker count / 设置worker数量
    if num_workers is None:
        num_workers = min(cpu_count(), len(demo_dirs), 8)
    print(f"Using {num_workers} parallel workers\n")
    
    plans = []
    stats = {
        'total': len(demo_dirs),
        'processed': 0,
        'skipped': 0,
        'single': 0,
        'bimanual': 0,
        'frames': 0,
        'duration': 0.0
    }
    
    # Prepare arguments for parallel processing / 准备并行处理的参数
    process_args = [
        (demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv)
        for demo_dir in demo_dirs
    ]
    
    # Process demos / 处理demos
    if num_workers > 1 and len(demo_dirs) > 1:
        # Parallel processing / 并行处理
        print(f"[INFO] Processing {len(demo_dirs)} demos in parallel...")
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_demo_wrapper, process_args)
        
        for demo_dir, plan, error in results:
            if error:
                print(f"  [ERROR] {demo_dir.name}: {error}")
                stats['skipped'] += 1
            elif plan:
                plans.append(plan)
                stats['processed'] += 1
                stats['frames'] += plan['n_frames']
                stats['duration'] += plan['n_frames'] / plan['fps']
                
                if plan['demo_mode'] == 'single':
                    stats['single'] += 1
                else:
                    stats['bimanual'] += 1
                
                print(f"  [OK] {demo_dir.name}: {len(plan.get('grippers', []))} hand(s), "
                     f"{len(plan.get('cameras', []))} cam(s), {plan['n_frames']} frames")
            else:
                stats['skipped'] += 1
    else:
        # Sequential processing / 顺序处理
        print(f"[INFO] Processing {len(demo_dirs)} demos sequentially...")
        for demo_dir in demo_dirs:
            plan = process_demo(demo_dir, min_length, use_tactile, visual_latency, 
                              pose_latency, force_regenerate_csv=force_regenerate_csv)
            if plan:
                plans.append(plan)
                stats['processed'] += 1
                stats['frames'] += plan['n_frames']
                stats['duration'] += plan['n_frames'] / plan['fps']
                
                if plan['demo_mode'] == 'single':
                    stats['single'] += 1
                else:
                    stats['bimanual'] += 1
            else:
                stats['skipped'] += 1
    
    if not plans:
        print("\n[ERROR] No valid demos!")
        return
    
    # Save result / 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump(plans, f)
    
    # Print summary statistics / 打印摘要统计信息
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Single-hand: {stats['single']}")
    print(f"Bimanual: {stats['bimanual']}")
    print(f"Total frames: {stats['frames']:,}")
    print(f"Total duration: {stats['duration']:.1f}s ({stats['duration']/60:.1f}min)")
    print(f"Success rate: {stats['processed']/stats['total']*100:.1f}%")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset plan by synchronizing all data modalities"
    )
    parser.add_argument('--cfg', type=str, required=True, 
                       help='Path to configuration YAML file')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regenerate trajectory CSV files')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    args = parser.parse_args()

    generate_plan(args.cfg, force_regenerate_csv=args.force_regenerate,
                 num_workers=args.num_workers)