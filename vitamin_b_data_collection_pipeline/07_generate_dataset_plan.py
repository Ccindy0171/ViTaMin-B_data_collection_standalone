#!/usr/bin/env python3
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
    坐标系转换：Unity左手系 -> 右手系
    
    Unity坐标系: X右 Y上 Z前
    右手坐标系: X右 Y前 Z上
    转换方法: 交换Y和Z轴
    
    参数:
        pose: [x, y, z, qx, qy, qz, qw] Unity坐标
    
    返回:
        (位置, 四元数) 右手坐标系
    """
    # 位置: [x, y, z] -> [x, z, y] (交换Y和Z)
    pos = np.array([pose[0], pose[2], pose[1]], dtype=float)
    
    # 旋转: 需要正确变换，不能简单交换四元数分量
    # 使用旋转矩阵Q进行坐标系变换
    # Q交换y和z轴: [x, y, z] -> [x, z, y]
    Q = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0]], dtype=float)
    
    # 将Unity的四元数转换为旋转矩阵
    rot = Rotation.from_quat(pose[3:7]).as_matrix()
    
    # 应用坐标系变换: Q @ rot @ Q.T
    # 这正确地将Unity坐标系中的旋转转换到右手坐标系
    transformed_rot = Q @ rot @ Q.T
    
    # 转换回四元数
    quat = Rotation.from_matrix(transformed_rot).as_quat()
    
    return pos, quat


def detect_demo_mode(demo_dir: Path):
    """
    检测demo是单手还是双手模式
    通过查找图像文件夹判断
    """
    # 兼容新旧两种文件夹命名
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
    """检查ArUco检测结果文件是否存在"""
    if mode == 'single':
        pkl = demo_dir / f'tag_detection_{hands[0]}.pkl'
        return pkl.exists()
    
    for hand in hands:
        if not (demo_dir / f'tag_detection_{hand}.pkl').exists():
            return False
    return True


def find_image_folders(demo_dir: Path, mode: str, hands: list, use_tactile: bool):
    """
    查找所有图像文件夹
    包括视觉相机和触觉传感器的图像
    """
    folders = {}
    
    for hand in hands:
        # 优先用分割后的visual图像，否则用原始图像
        visual_folder = demo_dir / f'{hand}_hand_visual_img'
        raw_folder = demo_dir / f'{hand}_hand_img'
        img_folder = visual_folder if visual_folder.exists() else raw_folder
        
        if img_folder.exists():
            folders[f'{hand}_visual'] = img_folder
            
            # 添加触觉图像文件夹
            if use_tactile:
                for side in ['left', 'right']:
                    tac_folder = demo_dir / f'{hand}_hand_{side}_tactile_img'
                    if tac_folder.exists():
                        folders[f'{hand}_hand_{side}_tactile'] = tac_folder
    
    return folders


def parse_timestamp(ts: str) -> float:
    """
    解析时间戳字符串为Unix时间
    支持多种格式
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

    # 尝试移除分隔符后再解析
    ts_compact = ts.replace(".", "").replace("-", "")
    try:
        return datetime.strptime(ts_compact, "%Y%m%d_%H%M%S_%f").timestamp()
    except Exception:
        raise ValueError(f"Unknown timestamp format: {ts}") from last_err


def get_image_times(demo_dir: Path, hand: str, latency: float) -> np.ndarray:
    """
    从CSV读取图像时间戳
    补偿相机延迟
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
    return times - latency


def ensure_split_images(demo_dir: Path, hand: str):
    """
    检查图像是否已分割
    实际分割由01_crop_img.py完成
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
    生成手的轨迹CSV文件
    
    重要: 左手设备用右Quest控制器，右手设备用左Quest控制器
    这是物理安装决定的
    """
    traj_file = demo_dir / 'pose_data' / f'{hand}_hand_trajectory.csv'
    
    if force_regenerate and traj_file.exists():
        traj_file.unlink()
        print(f"  [INFO] Deleted old CSV: {traj_file.name}")
    
    if traj_file.exists() and not force_regenerate:
        return traj_file
    
    if force_regenerate:
        print(f"  [INFO] Regenerating CSV for {hand} hand...")
    else:
        print(f"  [INFO] Generating CSV for {hand} hand...")
    
    task_dir = demo_dir.parent.parent
    all_traj_dir = task_dir / "all_trajectory"
    if not all_traj_dir.exists():
        raise FileNotFoundError(f"all_trajectory directory not found: {all_traj_dir}")
    
    json_files = sorted(all_traj_dir.glob("quest_poses_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No quest_poses_*.json found in {all_traj_dir}")
    
    # 左右手和Quest控制器的映射
    if hand == 'left':
        quest_wrist_key = 'right_wrist'  # 左手用右控制器
    elif hand == 'right':
        quest_wrist_key = 'left_wrist'   # 右手用左控制器
    else:
        raise ValueError(f"Unknown hand: {hand}, expected 'left' or 'right'")
    
    # 批量读取JSON
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
    
    # 处理姿态数据
    print(f"  [INFO] Processing {len(all_entries)} pose entries...")
    timestamps = []
    poses = []
    
    for entry in all_entries:
        if quest_wrist_key not in entry:
            continue
        wrist = entry[quest_wrist_key]
        pos = wrist.get("position", {})
        rot = wrist.get("rotation", {})
        
        # 提取位置
        if isinstance(pos, dict):
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))
        else:
            try:
                x, y, z = map(float, pos[:3])
            except Exception:
                x = y = z = 0.0
        
        # 提取旋转
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
        
        # 坐标系转换
        rel_pos, rel_quat = compute_rel_transform(
            np.array([x, y, z, q_x, q_y, q_z, q_w], dtype=float)
        )
        
        ts = entry.get("timestamp_unix", entry.get("timestamp", 0.0))
        timestamps.append(ts)
        poses.append([rel_pos[0], rel_pos[1], rel_pos[2], 
                     rel_quat[0], rel_quat[1], rel_quat[2], rel_quat[3]])
    
    if not timestamps:
        raise RuntimeError(f"No valid pose data for hand '{hand}' (using {quest_wrist_key})")
    
    # 写入CSV
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
    处理单只手的轨迹
    包括姿态插值和抓手宽度读取
    """
    traj_file = _ensure_hand_trajectory_csv(demo_dir, hand, force_regenerate=force_regenerate)

    # 读取轨迹CSV
    try:
        data = np.genfromtxt(traj_file, delimiter=",", names=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read trajectory CSV {traj_file}: {e}.")
    if data.size == 0:
        raise RuntimeError(f"No pose samples for {hand}")

    # 提取数据并补偿延迟
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
    
    # 排序和去重
    order = np.argsort(pose_times)
    pose_times = pose_times[order]
    pos = pos[order]
    quat = quat[order]
    
    pose_times, unique_idx = np.unique(pose_times, return_index=True)
    pos = pos[unique_idx]
    quat = quat[unique_idx]
    
    if len(pose_times) == 0:
        raise RuntimeError(f"No pose samples for {hand}")

    # 插值到目标时间
    if len(pose_times) == 1:
        # 单个样本: 重复使用
        interp_pos = np.repeat(pos, len(target_times), axis=0)
        interp_quat = np.repeat(quat, len(target_times), axis=0)
    else:
        # 多个样本: 插值
        # 裁剪到数据范围避免外推
        clipped_times = np.clip(target_times, pose_times[0], pose_times[-1])
        
        # 位置: 线性插值
        interp_pos = np.column_stack([
            np.interp(clipped_times, pose_times, pos[:, i], 
                     left=pos[0, i], right=pos[-1, i])
            for i in range(3)
        ])
        
        # 旋转: 球面线性插值(SLERP)
        rot = Rotation.from_quat(quat)
        slerp = Slerp(pose_times, rot)
        interp_quat = slerp(clipped_times).as_quat()
    
    # 转换为4x4变换矩阵
    rot_mats = Rotation.from_quat(interp_quat).as_matrix()
    n_frames = len(target_times)
    pose_mat = np.zeros((n_frames, 4, 4), dtype=np.float32)
    pose_mat[:, 3, 3] = 1
    pose_mat[:, :3, 3] = interp_pos
    pose_mat[:, :3, :3] = rot_mats
    
    # 读取抓手宽度
    gripper_file = demo_dir / f'gripper_width_{hand}.csv'
    if not gripper_file.exists():
        raise FileNotFoundError(f"Gripper width not found: {gripper_file}")

    try:
        gdata = np.genfromtxt(gripper_file, delimiter=",", names=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read gripper width CSV {gripper_file}: {e}.")
    if gdata.size == 0:
        raise RuntimeError(f"No gripper width samples for {hand}")

    # 兼容新旧列名
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

    quest_pose = mat_to_pose(pose_mat)
    
    return {
        "quest_pose": quest_pose,
        "gripper_width": widths,
        "demo_start_pose": quest_pose[0],
        "demo_end_pose": quest_pose[-1]
    }


def create_camera_entries(image_folders: dict, demo_dir: Path, n_frames: int, 
                         mode: str, hands: list):
    """创建相机条目"""
    cameras = []
    hand_idx_map = {'left': 0, 'right': 1}
    
    for usage_name, img_folder in image_folders.items():
        rel_path = img_folder.relative_to(demo_dir.parent)
        
        entry = {
            "image_folder": str(rel_path),
            "video_start_end": (0, n_frames),
            "usage_name": usage_name,
        }
        
        if mode == "single":
            entry["hand_position_idx"] = 0
        else:
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
    """处理单个demo"""
    try:
        mode, hands = detect_demo_mode(demo_dir)
        if not mode:
            print(f"  [SKIP] {demo_dir.name}: No image folders")
            return None
        
        if not check_aruco_files(demo_dir, mode, hands):
            print(f"  [SKIP] {demo_dir.name}: Missing ArUco files")
            return None
        
        for hand in hands:
            ensure_split_images(demo_dir, hand)
        
        image_folders = find_image_folders(demo_dir, mode, hands, use_tactile)
        if not image_folders:
            print(f"  [SKIP] {demo_dir.name}: No valid images")
            return None
        
        # 读取图像时间戳
        image_times = {}
        for hand in hands:
            image_times[hand] = get_image_times(demo_dir, hand, visual_latency)
        
        # 对齐帧数
        n_frames = min(len(times) for times in image_times.values())
        if n_frames < min_length:
            print(f"  [SKIP] {demo_dir.name}: Too short ({n_frames}<{min_length})")
            return None
        
        for hand in hands:
            if len(image_times[hand]) != n_frames:
                print(f"  [WARN] {demo_dir.name}: trimming {hand} from {len(image_times[hand])} to {n_frames}")
            image_times[hand] = image_times[hand][:n_frames]
        
        # 计算FPS
        ref_times = image_times[hands[0]]
        if len(ref_times) > 1:
            duration = ref_times[-1] - ref_times[0]
            fps = (len(ref_times) - 1) / duration if duration > 0 else 25.0
        else:
            fps = 25.0
        
        # 处理轨迹
        grippers = []
        for hand in hands:
            data = process_hand_trajectory(demo_dir, hand, image_times[hand], 
                                         pose_latency, force_regenerate=force_regenerate_csv)
            grippers.append(data)
        
        cameras = create_camera_entries(image_folders, demo_dir, n_frames, mode, hands)
        timestamps = ref_times
        
        print(f"  [OK] {demo_dir.name}: {len(hands)} hand(s), {len(cameras)} cam(s), {n_frames} frames")
        
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
    """并行处理的包装函数"""
    demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv = args
    try:
        plan = process_demo(demo_dir, min_length, use_tactile, visual_latency, 
                          pose_latency, force_regenerate_csv=force_regenerate_csv)
        return demo_dir, plan, None
    except Exception as e:
        return demo_dir, None, str(e)


def generate_plan(cfg_file: str, force_regenerate_csv: bool = False, num_workers: int = None):
    """生成数据集计划"""
    cfg = OmegaConf.load(cfg_file)
    
    task_name = cfg.task.name
    min_length = cfg.output_train_data.min_episode_length
    use_tactile = cfg.output_train_data.get("use_tactile_img", False) or \
                  cfg.output_train_data.get("use_tactile_pc", False)
    
    visual_latency = cfg.output_train_data.get("visual_cam_latency", 0.0)
    pose_latency = cfg.output_train_data.get("pose_latency", 0.0)
    
    demos_dir = DATA_DIR / task_name / 'demos'
    output_file = DATA_DIR / task_name / 'dataset_plan.pkl'
    
    print(f"Task: {task_name}")
    print(f"Min length: {min_length}")
    print(f"Tactile: {use_tactile}")
    if force_regenerate_csv:
        print(f"Force regenerate CSV: YES")
    print()
    
    # 删除旧CSV
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
    
    demo_dirs = sorted([d for d in demos_dir.glob('demo_*') if d.is_dir()])
    print(f"Found {len(demo_dirs)} demos")
    
    # 设置worker数量
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
    
    # 准备参数
    process_args = [
        (demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv)
        for demo_dir in demo_dirs
    ]
    
    # 并行处理
    if num_workers > 1 and len(demo_dirs) > 1:
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
    
    # 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump(plans, f)
    
    # 统计信息
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--force-regenerate', action='store_true')
    parser.add_argument('--num-workers', type=int, default=None)
    args = parser.parse_args()

    generate_plan(args.cfg, force_regenerate_csv=args.force_regenerate,
                 num_workers=args.num_workers)