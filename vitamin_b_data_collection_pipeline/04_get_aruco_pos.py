#!/usr/bin/env python3
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
    demo_dirs = []
    for demo_dir in demos_dir.glob('demo_*'):
        if not demo_dir.is_dir():
            continue
        
        if task_type == "single":
            if (demo_dir / f'{single_hand_side}_hand_visual_img').exists():
                demo_dirs.append(demo_dir)
        else:
            if (demo_dir / 'left_hand_visual_img').exists() or (demo_dir / 'right_hand_visual_img').exists():
                demo_dirs.append(demo_dir)
    
    return sorted(demo_dirs)

def create_detection_tasks(demo_dirs, task_type, single_hand_side, intrinsics, aruco_config):
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
    cv2.setNumThreads(num_workers)
    
    aruco_dict = task['aruco_config']['aruco_dict']
    marker_size_map = task['aruco_config']['marker_size_map']
    
    with open(task['intrinsics'], 'r') as f:
        raw_fisheye_intr = parse_fisheye_intrinsics(json.load(f))
    
    results = []
    input_path = Path(os.path.expanduser(task['input']))
    
    try:
        if not input_path.is_dir():
            return False, f"Input path is not a directory: {input_path}"
        # print(input_path)
        # Sort image files by numeric ID in filename if possible (e.g. left_hand_12.jpg)
        # 按文件名中的数字序号排序，确保与录制顺序一致
        img_files = sorted(
            input_path.glob('*.jpg'),
            key=lambda p: int(re.search(r'(\d+)(?=\.jpg$)', p.name).group(1))
            if re.search(r'(\d+)(?=\.jpg$)', p.name) else p.name
        )
        if not img_files:
            return False, f"No jpg images found in {input_path}"
        
        first_img = cv2.imread(str(img_files[0]))
        if first_img is None:
            return False, f"Failed to read first image: {img_files[0]}"
        
        h, w = first_img.shape[:2]
        in_res = np.array([h, w])[::-1]
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)
        
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
        
        for i, img_file in enumerate(img_files):
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            tag_dict = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True
            )
            
            time_val = timestamps[i] if timestamps and i < len(timestamps) else float(i) / 30.0
            
            result = {
                'frame_idx': i,
                'time': time_val,
                'tag_dict': tag_dict
            }
            results.append(result)
        
        output_path = os.path.expanduser(task['output'])
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        return True, None
    except Exception as e:
        return False, str(e)

def run_detection(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    
    config_dir = Path(cfg_file).resolve().parent
    task_name = cfg.task.name
    task_type = cfg.task.type
    single_hand_side = cfg.task.get("single_hand_side", "left")
    
    intrinsics = cfg.calculate_width.cam_intrinsic_json_path
    aruco_dict_config = cfg.calculate_width.aruco_dict
    marker_size_map_config = cfg.calculate_width.marker_size_map
    
    if not Path(intrinsics).is_absolute():
        intrinsics = str((config_dir / intrinsics).resolve())
    
    # Parse aruco config from OmegaConf to dict
    aruco_config_dict = {
        'aruco_dict': OmegaConf.to_container(aruco_dict_config, resolve=True),
        'marker_size_map': OmegaConf.to_container(marker_size_map_config, resolve=True)
    }
    aruco_config = parse_aruco_config(aruco_config_dict)
    
    demos_dir = DATA_DIR / task_name / "demos"
    demo_dirs = find_demos_with_images(demos_dir, task_type, single_hand_side)
    print(demos_dir)
    print(f"[{task_type}] Found {len(demo_dirs)} demos")
    
    tasks = create_detection_tasks(demo_dirs, task_type, single_hand_side, intrinsics, aruco_config)
    print(f"Created {len(tasks)} detection tasks")
    
    max_workers = cfg.calculate_width.get("max_workers", 4)
    
    results = []
    with tqdm(total=len(tasks), desc="ArUco Detection") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for task in tasks:
                future = executor.submit(process_video_detection, task, max_workers)
                futures[future] = task
            
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                success, error = future.result()
                results.append((task, success, error))
                pbar.update(1)
                
                if not success:
                    print(f"\n[ERROR] {task['demo']} ({task['hand']}): {error}")
    
    success_count = sum(1 for _, s, _ in results if s)
    print(f"\n[DONE] {success_count}/{len(tasks)} successful")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    run_detection(args.cfg)