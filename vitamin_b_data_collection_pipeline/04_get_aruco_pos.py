#!/usr/bin/env python3
"""
ArUco Marker Detection Script / ArUco标记检测脚本
=================================================
Detects and localizes ArUco markers in fisheye camera images.
Used to track gripper positions for width calculation.

在鱼眼相机图像中检测和定位ArUco标记。
用于追踪夹爪位置以计算宽度。

Process flow / 处理流程:
1. Load camera intrinsics and ArUco configuration / 加载相机内参和ArUco配置
2. Detect ArUco markers in each visual image / 检测每张视觉图像中的ArUco标记
3. Compute 3D pose of each detected marker / 计算每个检测到的标记的3D姿态
4. Save detection results as pickle files / 保存检测结果为pickle文件

Output / 输出:
- tag_detection_{hand}.pkl: Contains frame-by-frame detection results
  包含逐帧检测结果
  
Image Format Notes / 图像格式说明:
- OpenCV loads images in BGR format by default / OpenCV默认以BGR格式加载图像
- ArUco detection works with BGR, RGB, or grayscale / ArUco检测支持BGR、RGB或灰度图
- ArUco internally converts to grayscale for detection / ArUco内部转换为灰度进行检测
- We use BGR directly to avoid unnecessary conversion / 直接使用BGR避免不必要的转换
"""
import sys
import os
import concurrent.futures
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import json
import pickle
import numpy as np
import re
import cv2
import pandas as pd
from datetime import datetime

# Configuration Constants / 配置常量
DEFAULT_OPENCV_FORMAT = 'BGR'  # OpenCV's default color format / OpenCV默认颜色格式
ARUCO_ACCEPTS_ANY_FORMAT = True  # ArUco works with BGR/RGB/Gray / ArUco支持BGR/RGB/灰度

# Project path setup / 项目路径设置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT))

from utils.cv_util import (
    parse_aruco_config,
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags
)

def find_demos_with_images(demos_dir: Path, task_type: str, single_hand_side: str):
    """
    Find all demo directories with visual image folders.
    查找所有包含视觉图像文件夹的demo目录
    
    Args:
        demos_dir: Directory containing demo folders / 包含demo文件夹的目录
        task_type: "single" or "bimanual" / "单手"或"双手"
        single_hand_side: "left" or "right" (for single-hand tasks) / "左"或"右"(用于单手任务)
    
    Returns:
        List of demo directories with visual images / 包含视觉图像的demo目录列表
    """
    demo_dirs = []
    for demo_dir in demos_dir.glob('demo_*'):
        if not demo_dir.is_dir():
            continue
        
        if task_type == "single":
            # Check for visual images after cropping / 检查裁剪后的视觉图像
            if (demo_dir / f'{single_hand_side}_hand_visual_img').exists():
                demo_dirs.append(demo_dir)
        else:
            # Check for either hand's visual images / 检查任一只手的视觉图像
            if (demo_dir / 'left_hand_visual_img').exists() or (demo_dir / 'right_hand_visual_img').exists():
                demo_dirs.append(demo_dir)
    
    return sorted(demo_dirs)

def create_detection_tasks(demo_dirs, task_type, single_hand_side, intrinsics, aruco_config):
    """
    Create detection tasks for parallel processing.
    创建用于并行处理的检测任务
    
    Args:
        demo_dirs: List of demo directories / demo目录列表
        task_type: "single" or "bimanual" / "单手"或"双手"
        single_hand_side: Hand side for single-hand mode / 单手模式的手的方向
        intrinsics: Path to camera intrinsics file / 相机内参文件路径
        aruco_config: ArUco configuration dict / ArUco配置字典
    
    Returns:
        List of task dictionaries / 任务字典列表
    """
    tasks = []
    
    for demo_dir in demo_dirs:
        if task_type == "single":
            img_folder = demo_dir / f'{single_hand_side}_hand_visual_img'
            if img_folder.exists():
                tasks.append({
                    'input': str(img_folder),
                    'output': str(demo_dir / f'tag_detection_{single_hand_side}.pkl'),
                    'intrinsics': intrinsics,
                    'aruco_config': aruco_config,
                    'demo': demo_dir.name,
                    'hand': single_hand_side,
                    'demo_dir': str(demo_dir)
                })
        else:
            # Create tasks for both hands / 为双手创建任务
            for hand in ['left', 'right']:
                img_folder = demo_dir / f'{hand}_hand_visual_img'
                if img_folder.exists():
                    tasks.append({
                        'input': str(img_folder),
                        'output': str(demo_dir / f'tag_detection_{hand}.pkl'),
                        'intrinsics': intrinsics,
                        'aruco_config': aruco_config,
                        'demo': demo_dir.name,
                        'hand': hand,
                        'demo_dir': str(demo_dir)
                    })
    
    return tasks

def process_video_detection(task, num_workers):
    """
    Process ArUco detection for one video (image sequence).
    处理一个视频(图像序列)的ArUco检测
    
    Args:
        task: Task dictionary containing input/output paths and config
              任务字典,包含输入/输出路径和配置
        num_workers: Number of OpenCV threads to use / OpenCV线程数
    
    Returns:
        tuple: (success: bool, error_message: str or None)
               (成功标志, 错误消息或None)
    
    Process / 处理过程:
    1. Load camera intrinsics / 加载相机内参
    2. Adjust intrinsics for image resolution / 调整内参以适应图像分辨率
    3. Detect ArUco markers in each frame / 检测每帧中的ArUco标记
    4. Save results with timestamps / 保存带时间戳的结果
    """
    cv2.setNumThreads(num_workers)
    
    aruco_dict = task['aruco_config']['aruco_dict']
    marker_size_map = task['aruco_config']['marker_size_map']
    
    # Load and parse fisheye intrinsics / 加载和解析鱼眼内参
    with open(task['intrinsics'], 'r') as f:
        raw_fisheye_intr = parse_fisheye_intrinsics(json.load(f))
    
    results = []
    input_path = Path(os.path.expanduser(task['input']))
    
    try:
        if not input_path.is_dir():
            return False, f"Input path is not a directory: {input_path}"
        
        # Sort image files by numeric ID in filename / 按文件名中的数字ID排序
        # Ensures chronological order / 确保按时间顺序
        img_files = sorted(
            input_path.glob('*.jpg'),
            key=lambda p: int(re.search(r'(\d+)(?=\.jpg$)', p.name).group(1))
            if re.search(r'(\d+)(?=\.jpg$)', p.name) else p.name
        )
        if not img_files:
            return False, f"No jpg images found in {input_path}"
        
        # Read first image to get resolution / 读取第一张图像获取分辨率
        first_img = cv2.imread(str(img_files[0]))
        if first_img is None:
            return False, f"Failed to read first image: {img_files[0]}"
        
        h, w = first_img.shape[:2]
        in_res = np.array([h, w])[::-1]  # Convert to (width, height) / 转换为(宽度,高度)
        
        # Convert intrinsics to match image resolution / 转换内参以匹配图像分辨率
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)
        
        # Load timestamps if available / 如果可用,加载时间戳
        demo_dir = Path(task['demo_dir'])
        hand = task['hand']
        timestamps_file = demo_dir / f'{hand}_hand_timestamps.csv'
        timestamps = None
        if timestamps_file.exists():
            try:
                df = pd.read_csv(timestamps_file)
                if 'ram_time' in df.columns:
                    def parse_time(ts_str):
                        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S_%f").timestamp()
                    timestamps = [parse_time(ts) for ts in df['ram_time']]
            except:
                pass
        
        # Process each image / 处理每张图像
        for i, img_file in enumerate(img_files):
            # Load image in BGR format (OpenCV default)
            # BGR格式加载图像(OpenCV默认)
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                continue
            
            # Note: ArUco detection accepts BGR, RGB, or grayscale
            # It internally converts to grayscale for marker detection
            # We keep it in BGR to avoid unnecessary conversion overhead
            # 注意: ArUco检测接受BGR、RGB或灰度图像
            # 内部会转换为灰度图进行标记检测
            # 保持BGR格式以避免不必要的转换开销
            
            # Detect and localize ArUco tags / 检测和定位ArUco标记
            tag_dict = detect_localize_aruco_tags(
                img=img_bgr,  # Pass BGR directly (ArUco converts internally)
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True  # Subpixel refinement for accuracy / 亚像素精化以提高精度
            )
            
            # Use timestamp from CSV if available, otherwise estimate / 如果可用使用CSV时间戳,否则估计
            time_val = timestamps[i] if timestamps and i < len(timestamps) else float(i) / 30.0
            
            result = {
                'frame_idx': i,
                'time': time_val,
                'tag_dict': tag_dict
            }
            results.append(result)
        
        # Save detection results / 保存检测结果
        output_path = os.path.expanduser(task['output'])
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        return True, None
    except Exception as e:
        return False, str(e)

def run_detection(cfg_file):
    """
    Main function to run ArUco detection on all demos.
    主函数,对所有demo运行ArUco检测
    
    Args:
        cfg_file: Path to configuration YAML file / 配置YAML文件路径
    
    Process / 处理过程:
    1. Load configuration / 加载配置
    2. Parse ArUco and camera parameters / 解析ArUco和相机参数
    3. Create detection tasks / 创建检测任务
    4. Run parallel detection / 运行并行检测
    5. Report results / 报告结果
    """
    cfg = OmegaConf.load(cfg_file)
    
    config_dir = Path(cfg_file).resolve().parent
    task_name = cfg.task.name
    task_type = cfg.task.type
    single_hand_side = cfg.task.get("single_hand_side", "left")
    
    # Resolve paths / 解析路径
    intrinsics = cfg.calculate_width.cam_intrinsic_json_path
    aruco_dict_config = cfg.calculate_width.aruco_dict
    marker_size_map_config = cfg.calculate_width.marker_size_map
    
    # Convert relative path to absolute / 将相对路径转换为绝对路径
    if not Path(intrinsics).is_absolute():
        intrinsics = str((config_dir / intrinsics).resolve())
    
    # Parse aruco config from OmegaConf to dict / 从OmegaConf解析aruco配置为字典
    aruco_config_dict = {
        'aruco_dict': OmegaConf.to_container(aruco_dict_config, resolve=True),
        'marker_size_map': OmegaConf.to_container(marker_size_map_config, resolve=True)
    }
    aruco_config = parse_aruco_config(aruco_config_dict)
    
    # Find demos and create tasks / 查找demos并创建任务
    demos_dir = DATA_DIR / task_name / "demos"
    demo_dirs = find_demos_with_images(demos_dir, task_type, single_hand_side)
    print(demos_dir)
    print(f"[{task_type}] Found {len(demo_dirs)} demos")
    
    tasks = create_detection_tasks(demo_dirs, task_type, single_hand_side, intrinsics, aruco_config)
    print(f"Created {len(tasks)} detection tasks")
    
    max_workers = cfg.calculate_width.get("max_workers", 4)
    
    # Execute tasks in parallel with progress bar / 使用进度条并行执行任务
    results = []
    with tqdm(total=len(tasks), desc="ArUco Detection") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            # Submit all tasks / 提交所有任务
            for task in tasks:
                future = executor.submit(process_video_detection, task, max_workers)
                futures[future] = task
            
            # Collect results as they complete / 任务完成时收集结果
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                success, error = future.result()
                results.append((task, success, error))
                pbar.update(1)
                
                if not success:
                    print(f"\n[ERROR] {task['demo']} ({task['hand']}): {error}")
    
    # Print summary / 打印摘要
    success_count = sum(1 for _, s, _ in results if s)
    print(f"\n[DONE] {success_count}/{len(tasks)} successful")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    run_detection(args.cfg)