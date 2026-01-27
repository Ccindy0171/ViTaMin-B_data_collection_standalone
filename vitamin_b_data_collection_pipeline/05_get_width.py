#!/usr/bin/env python3
"""
Calculate gripper width from ArUco marker detections.
从ArUco标记检测结果计算抓手宽度

This script processes ArUco detection pickle files to compute gripper width
based on the distance between two markers on each gripper finger. The process:
该脚本处理ArUco检测pickle文件,基于抓手两指上的两个标记之间的距离计算抓手宽度。处理流程:

1. Load tag detections from pickle files / 从pickle文件加载标记检测结果
2. Calculate width from marker pair positions / 从标记对位置计算宽度
3. Interpolate missing/invalid widths / 插值缺失/无效的宽度值
4. Save results as CSV with frame-by-frame widths / 将结果保存为带逐帧宽度的CSV

The output CSV files are used in later stages to synchronize gripper state
with trajectory data.
输出CSV文件在后续阶段用于将抓手状态与轨迹数据同步
"""
import sys
import pickle
import argparse
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from scipy.interpolate import interp1d
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cv_util import get_gripper_width


@dataclass
class WidthTask:
    """
    Task definition for width calculation.
    宽度计算任务定义
    
    Attributes:
        pkl_file: Path to ArUco detection pickle file / ArUco检测pickle文件路径
        csv_file: Output CSV file path / 输出CSV文件路径
        hand: 'left' or 'right' / 左手或右手
        aruco_ids: Tuple of two marker IDs on gripper fingers / 抓手两指上的标记ID元组
    """
    pkl_file: Path
    csv_file: Path
    hand: str
    aruco_ids: Tuple[int, int]


def interpolate_widths_np(frames: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    Interpolate gripper widths with gap filling.
    插值抓手宽度并填补空缺
    
    This function performs linear interpolation on valid width measurements
    to fill in missing or invalid values. The behavior is equivalent to the
    original pandas version:
    该函数对有效的宽度测量值进行线性插值以填补缺失或无效值。行为等价于原始pandas版本:
    
    - Fit only on valid widths (>0 and not NaN) / 仅在有效宽度(>0且非NaN)上拟合
    - Linearly interpolate missing values / 线性插值缺失值
    - Fill remaining NaN with default 0.05 / 用默认值0.05填充剩余NaN
    
    Args:
        frames: Frame indices (integer array) / 帧索引(整数数组)
        widths: Gripper width measurements (may contain NaN) / 抓手宽度测量值(可能包含NaN)
    
    Returns:
        Interpolated widths with all NaN filled / 插值后的宽度,所有NaN已填充
    """
    widths = widths.astype(float)

    # Valid values: not NaN and > 0 / 有效值:非NaN且>0
    valid_mask = (~np.isnan(widths)) & (widths > 0)
    valid_frames = frames[valid_mask]
    valid_widths = widths[valid_mask]

    if valid_frames.size == 0:
        # No valid values, set all to default / 完全没有有效值,全部设为默认值
        return np.full_like(widths, 0.05, dtype=float)
    if valid_frames.size == 1:
        # Only one valid value, fill all with it / 只有一个有效值,全部填成这个值
        return np.full_like(widths, float(valid_widths[0]), dtype=float)

    # Try linear interpolation / 尝试线性插值
    try:
        interp = interp1d(
            valid_frames,
            valid_widths,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",  # Extrapolate beyond valid range / 超出有效范围时外推
        )
        # Interpolate only invalid positions / 仅对无效位置进行插值
        invalid_mask = ~valid_mask
        if np.any(invalid_mask):
            widths[invalid_mask] = interp(frames[invalid_mask])
    except Exception:
        # If interpolation fails, keep original widths / 插值失败时保留原始widths
        pass

    # Fill remaining NaN with default / 对仍为NaN的位置填默认值
    widths = np.where(np.isnan(widths), 0.05, widths)
    return widths


def process_width(task: WidthTask) -> Path:
    """
    Process one detection file to calculate gripper widths.
    处理一个检测文件以计算抓手宽度
    
    Args:
        task: WidthTask containing input/output paths and parameters / 包含输入输出路径和参数的WidthTask
    
    Returns:
        Path to saved CSV file / 保存的CSV文件路径
    
    Process / 处理流程:
    1. Load ArUco detections from pickle / 从pickle加载ArUco检测结果
    2. Calculate width for each frame using marker pair / 使用标记对计算每帧宽度
    3. Interpolate missing widths / 插值缺失的宽度
    4. Save as CSV with 'frame,width' columns / 保存为带'frame,width'列的CSV
    """
    detections = pickle.load(task.pkl_file.open('rb'))

    widths_list = []
    valid_count = 0

    # Calculate width for each frame / 计算每帧的宽度
    for i, det in enumerate(detections):
        width = get_gripper_width(det['tag_dict'], task.aruco_ids[0], task.aruco_ids[1])
        if width is not None and not np.isnan(width) and width > 0:
            valid_count += 1
        widths_list.append(width)

    print(f"  {task.hand}: {valid_count}/{len(detections)} valid")

    # Construct frame numbers and width array / 构造帧号与宽度数组
    frames = np.arange(len(detections), dtype=int)
    widths = np.array(
        [np.nan if w is None else float(w) for w in widths_list],
        dtype=float,
    )
    
    # Interpolate to fill gaps / 插值填补空缺
    widths = interpolate_widths_np(frames, widths)

    # Write as CSV: two columns frame,width / 写出为CSV:两列frame,width
    data = np.column_stack([frames, widths])
    np.savetxt(
        task.csv_file,
        data,
        delimiter=",",
        header="frame,width",
        comments="",  # No comment prefix / 无注释前缀
        fmt=["%d", "%.8f"],  # Integer frame, float width / 整数帧号,浮点宽度
    )

    return task.csv_file


def create_tasks(demos_dir: Path, task_type: str, single_hand_side: str, left_ids: Tuple, right_ids: Tuple) -> List[WidthTask]:
    """
    Create width calculation tasks for all demos.
    为所有demos创建宽度计算任务
    
    Args:
        demos_dir: Directory containing demo folders / 包含demo文件夹的目录
        task_type: 'single' or 'bimanual' / 单手或双手
        single_hand_side: 'left' or 'right' (only used for single) / 左手或右手(仅用于单手)
        left_ids: Tuple of ArUco IDs for left gripper / 左抓手的ArUco ID元组
        right_ids: Tuple of ArUco IDs for right gripper / 右抓手的ArUco ID元组
    
    Returns:
        List of WidthTask objects / WidthTask对象列表
    """
    tasks = []
    
    for demo_dir in demos_dir.glob('demo_*'):
        if not demo_dir.is_dir():
            continue
        
        if task_type == "single":
            # Single-hand: process only specified side / 单手:仅处理指定侧
            pkl = demo_dir / f'tag_detection_{single_hand_side}.pkl'
            if pkl.exists():
                tasks.append(WidthTask(
                    pkl_file=pkl,
                    csv_file=demo_dir / f'gripper_width_{single_hand_side}.csv',
                    hand=single_hand_side,
                    aruco_ids=left_ids if single_hand_side == 'left' else right_ids
                ))
        else:
            # Bimanual: process both hands / 双手:处理两只手
            for hand, ids in [('left', left_ids), ('right', right_ids)]:
                pkl = demo_dir / f'tag_detection_{hand}.pkl'
                if pkl.exists():
                    tasks.append(WidthTask(
                        pkl_file=pkl,
                        csv_file=demo_dir / f'gripper_width_{hand}.csv',
                        hand=hand,
                        aruco_ids=ids
                    ))
    
    return tasks


def run_width_calculation(cfg_file: str):
    """
    Main function to run width calculation on all demos.
    主函数,对所有demos运行宽度计算
    
    Args:
        cfg_file: Path to configuration YAML file / 配置YAML文件路径
    
    Process / 处理过程:
    1. Load configuration / 加载配置
    2. Parse ArUco marker IDs for both hands / 解析两只手的ArUco标记ID
    3. Create width calculation tasks / 创建宽度计算任务
    4. Run parallel processing / 运行并行处理
    5. Print statistics for all results / 打印所有结果的统计信息
    """
    cfg = OmegaConf.load(cfg_file)
    
    task_name = cfg.task.name
    task_type = cfg.task.type
    single_hand_side = cfg.task.get("single_hand_side", "left")
    
    # Get ArUco IDs for left/right grippers / 获取左/右抓手的ArUco ID
    left_ids = (
        cfg.calculate_width.left_hand_aruco_id.left_id,
        cfg.calculate_width.left_hand_aruco_id.right_id
    )
    right_ids = (
        cfg.calculate_width.right_hand_aruco_id.left_id,
        cfg.calculate_width.right_hand_aruco_id.right_id
    )
    
    demos_dir = DATA_DIR / task_name / "demos"
    tasks = create_tasks(demos_dir, task_type, single_hand_side, left_ids, right_ids)
    
    print(f"[{task_type}] Processing {len(tasks)} tasks")
    
    if not tasks:
        print("[ERROR] No tasks!")
        return
    
    # Execute tasks in parallel / 并行执行任务
    saved = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_width, t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                csv_file = future.result()
                saved.append(csv_file)
            except Exception as e:
                task = futures[future]
                print(f"[ERROR] {task.pkl_file.name}: {e}")
    
    print(f"\n[DONE] {len(saved)}/{len(tasks)} successful")

    # Print statistics for all saved files / 打印所有保存文件的统计信息
    if saved:
        total_frames = 0
        total_valid = 0
        for csv in sorted(saved):
            # Read CSV with header / 读取带header的CSV
            arr = np.genfromtxt(csv, delimiter=",", names=True)
            if arr.size == 0:
                continue

            widths = arr["width"]
            # Ensure 1D array / 统一成1D数组
            widths = np.atleast_1d(widths)

            # Count valid widths (>0 and not NaN) / 统计有效宽度(>0且非NaN)
            mask_valid = (~np.isnan(widths)) & (widths > 0)
            n_frames = widths.size
            n_valid = int(mask_valid.sum())

            total_frames += n_frames
            total_valid += n_valid
            ratio = (n_valid / n_frames * 100) if n_frames > 0 else 0.0
            print(f"  {csv.name}: {n_frames} frames, {n_valid} valid ({ratio:.1f}%)")

        if total_frames > 0:
            total_ratio = total_valid / total_frames * 100
            print(f"\nTotal: {total_frames} frames, {total_valid} valid ({total_ratio:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    run_width_calculation(args.cfg)
