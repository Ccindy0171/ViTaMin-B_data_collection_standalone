#!/usr/bin/env python3
"""
Image Cropping Script / 图像裁剪脚本
==================================
Splits raw 3840x800 images into three 1280x800 parts:
- left_tactile: Left tactile sensor view
- visual: Main camera view
- right_tactile: Right tactile sensor view

将原始3840x800图像分割为三个1280x800部分:
- left_tactile: 左触觉传感器视图
- visual: 主相机视图
- right_tactile: 右触觉传感器视图

Process flow / 处理流程:
1. Find all demo directories with raw images / 查找所有包含原始图像的demo目录
2. Parallel process each hand's images / 并行处理每只手的图像
3. Apply rotation corrections based on hand side / 根据手的方向应用旋转校正
4. Save cropped images to separate folders / 保存裁剪图像到独立文件夹
"""

import sys
import os
import argparse
from pathlib import Path
import re
from omegaconf import OmegaConf
from tqdm import tqdm
import cv2
from multiprocessing import Pool, cpu_count

# Project path setup / 项目路径设置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT))


def find_demos_with_images(demos_dir: Path, task_type: str, single_hand_side: str):
    """
    Find all demo directories that have image folders.
    查找所有包含图像文件夹的demo目录
    
    Args:
        demos_dir: Directory containing demo folders / 包含demo文件夹的目录
        task_type: "single" or "bimanual" / "单手"或"双手"
        single_hand_side: "left" or "right" (for single-hand tasks) / "左"或"右"(用于单手任务)
    
    Returns:
        List of demo directories with images / 包含图像的demo目录列表
    """
    demo_dirs = []
    for demo_dir in demos_dir.glob('demo_*'):
        if not demo_dir.is_dir():
            continue
        
        if task_type == "single":
            # Single-hand mode: check for specified hand / 单手模式:检查指定的手
            if (demo_dir / f'{single_hand_side}_hand_img').exists():
                demo_dirs.append(demo_dir)
        else:
            # Bimanual mode: check for either hand / 双手模式:检查任一只手
            if (demo_dir / 'left_hand_img').exists() or (demo_dir / 'right_hand_img').exists():
                demo_dirs.append(demo_dir)
    
    return sorted(demo_dirs)


def _crop_images_wrapper(args):
    """
    Wrapper function for multiprocessing.
    多进程处理的包装函数
    
    Args:
        args: Tuple of (demo_dir, hand) / (demo目录, 手的方向)的元组
    
    Returns:
        Result from crop_images_for_hand / crop_images_for_hand的结果
    """
    demo_dir, hand = args
    return crop_images_for_hand(Path(demo_dir), hand)


def crop_images_for_hand(demo_dir: Path, hand: str):
    """
    Crop images for a single hand.
    为单只手裁剪图像
    
    Process / 处理过程:
    1. Read raw 3840x800 images / 读取原始3840x800图像
    2. Split into 3 parts: left_tactile (0-1280), visual (1280-2560), right_tactile (2560-3840)
       分割为3部分: 左触觉(0-1280), 视觉(1280-2560), 右触觉(2560-3840)
    3. Apply rotation correction for left hand / 为左手应用旋转校正
    4. Save to separate folders / 保存到独立文件夹
    
    Args:
        demo_dir: Demo directory path / Demo目录路径
        hand: 'left' or 'right' / '左'或'右'
    
    Returns:
        tuple: (success: bool, demo_name: str, hand: str, total: int, success_count: int, message: str)
               (成功标志, demo名称, 手的方向, 总数, 成功数, 消息)
    """
    demo_dir = Path(demo_dir)  # Ensure it's a Path object / 确保是Path对象
    raw_dir = demo_dir / f'{hand}_hand_img'
    if not raw_dir.exists():
        return (False, demo_dir.name, hand, 0, 0, f"{hand}_hand_img folder not found")
    
    # Create output directories / 创建输出目录
    visual_dir = demo_dir / f'{hand}_hand_visual_img'
    left_tactile_dir = demo_dir / f'{hand}_hand_left_tactile_img'
    right_tactile_dir = demo_dir / f'{hand}_hand_right_tactile_img'
    
    # Create directories if they don't exist / 如果不存在则创建目录
    visual_dir.mkdir(parents=True, exist_ok=True)
    left_tactile_dir.mkdir(parents=True, exist_ok=True)
    right_tactile_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort image files by numeric ID in filename if possible
    # 如果可能，按文件名中的数字ID排序
    raw_files = sorted(
        raw_dir.glob('*.jpg'),
        key=lambda p: int(re.search(r'(\d+)(?=\.jpg$)', p.name).group(1))
        if re.search(r'(\d+)(?=\.jpg$)', p.name) else p.name
    )
    
    if not raw_files:
        return (False, demo_dir.name, hand, 0, 0, "No JPG images found")
    
    # Image dimensions / 图像尺寸
    CROP_WIDTH = 1280  # Width of each cropped section / 每个裁剪部分的宽度
    TOTAL_WIDTH = 3840  # Total width of raw image / 原始图像的总宽度
    
    success_count = 0
    error_count = 0
    for img_path in raw_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            error_count += 1
            continue
        
        h, w = img.shape[:2]
        
        # Verify image dimensions / 验证图像尺寸
        if w != TOTAL_WIDTH or h != 800:
            # Continue anyway, but use actual dimensions / 继续处理,但使用实际尺寸
            if w < CROP_WIDTH * 3:
                error_count += 1
                continue
        
        # Crop into three parts: left_tactile, visual, right_tactile
        # 裁剪为三部分: 左触觉, 视觉, 右触觉
        left_tactile = img[:, 0:CROP_WIDTH]
        visual = img[:, CROP_WIDTH:2*CROP_WIDTH]
        right_tactile = img[:, 2*CROP_WIDTH:3*CROP_WIDTH]
        
        # Rotation correction based on hand side
        # 基于手的方向进行旋转校正
        # Note: Currently only left tactile is rotated 180 degrees
        # 注意: 目前仅左触觉旋转180度
        # Original logic had different rotations for left/right hands
        # 原始逻辑对左右手有不同的旋转
        # if hand == 'right':
        #     left_tactile_final = left_tactile
        #     visual_final = cv2.rotate(visual, cv2.ROTATE_180)
        #     right_tactile_final = cv2.rotate(right_tactile, cv2.ROTATE_180)
        # else:
        #     left_tactile_final = cv2.rotate(left_tactile, cv2.ROTATE_180)
        #     visual_final = visual
        #     right_tactile_final = right_tactile

        left_tactile_final = cv2.rotate(left_tactile, cv2.ROTATE_180)
        visual_final = visual
        right_tactile_final = right_tactile
    
        # Save cropped images with the same filename / 保存裁剪图像,保持相同文件名
        left_tactile_path = left_tactile_dir / img_path.name
        visual_path = visual_dir / img_path.name
        right_tactile_path = right_tactile_dir / img_path.name
        
        cv2.imwrite(str(left_tactile_path), left_tactile_final)
        cv2.imwrite(str(visual_path), visual_final)
        cv2.imwrite(str(right_tactile_path), right_tactile_final)
        
        success_count += 1
    
    # Generate summary message / 生成摘要消息
    message = f"{success_count}/{len(raw_files)} images processed"
    if error_count > 0:
        message += f", {error_count} errors"
    
    return (True, demo_dir.name, hand, len(raw_files), success_count, message)


def main(cfg_file: str, num_workers: int = None):
    """
    Main function to crop images for all demos.
    主函数,为所有demo裁剪图像
    
    Args:
        cfg_file: Path to configuration file / 配置文件路径
        num_workers: Number of parallel workers (default: auto) / 并行worker数量(默认:自动)
    """
    cfg = OmegaConf.load(cfg_file)
    
    task_name = cfg.task.name
    task_type = cfg.task.type
    single_hand_side = cfg.task.get("single_hand_side", "left")
    
    demos_dir = DATA_DIR / task_name / "demos"
    
    if not demos_dir.exists():
        print(f"[ERROR] Demos directory not found: {demos_dir}")
        return
    
    demo_dirs = find_demos_with_images(demos_dir, task_type, single_hand_side)
    
    if not demo_dirs:
        print(f"[WARN] No demos found with image folders in {demos_dir}")
        return
    
    # Generate task list: (demo_dir, hand) pairs
    # 生成任务列表: (demo_dir, hand) 对
    tasks = []
    for demo_dir in demo_dirs:
        if task_type == "single":
            tasks.append((str(demo_dir), single_hand_side))
        else:
            # Check which hands exist / 检查哪只手存在
            if (demo_dir / 'left_hand_img').exists():
                tasks.append((str(demo_dir), 'left'))
            if (demo_dir / 'right_hand_img').exists():
                tasks.append((str(demo_dir), 'right'))
    
    print(f"[INFO] Found {len(demo_dirs)} demos to process")
    print(f"[INFO] Task type: {task_type}")
    if task_type == "single":
        print(f"[INFO] Processing {single_hand_side} hand only")
    else:
        print(f"[INFO] Processing both left and right hands")
    print(f"[INFO] Total tasks: {len(tasks)}")
    
    # Set default number of workers / 设置默认worker数量
    if num_workers is None:
        num_workers = min(cpu_count(), len(tasks))
    print(f"[INFO] Using {num_workers} worker processes")
    
    # Process tasks in parallel / 并行处理任务
    if num_workers == 1:
        # Single process mode (for debugging) / 单进程模式(用于调试)
        results = []
        for demo_dir_str, hand in tqdm(tasks, desc="Processing tasks"):
            result = crop_images_for_hand(Path(demo_dir_str), hand)
            results.append(result)
    else:
        # Multi-process mode / 多进程模式
        print(f"[INFO] Starting parallel processing with {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            # Use imap for real-time progress tracking / 使用imap实现实时进度追踪
            results = []
            with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                for result in pool.imap(_crop_images_wrapper, tasks):
                    results.append(result)
                    pbar.update(1)
    
    # Print results summary / 打印结果摘要
    print("\n" + "="*60)
    print("Processing Summary:")
    print("="*60)
    success_count = 0
    total_images = 0
    processed_images = 0
    
    for success, demo_name, hand, total, success_count_task, message in results:
        status = "✓" if success else "✗"
        print(f"{status} {demo_name}/{hand}_hand: {message}")
        if success:
            success_count += 1
            total_images += total
            processed_images += success_count_task
    
    print("="*60)
    print(f"[SUCCESS] Completed {success_count}/{len(results)} tasks")
    print(f"[SUCCESS] Processed {processed_images}/{total_images} images")
    print(f"[SUCCESS] Image cropping completed for {len(demo_dirs)} demos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop 3840x800 images into three 1280x800 images (left_tactile, visual, right_tactile)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--cfg', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--num_workers', type=int, default=None,
                        help=f'Number of worker processes (default: min(CPU count, number of tasks), max: {cpu_count()})')
    args = parser.parse_args()
    
    main(args.cfg, args.num_workers)

