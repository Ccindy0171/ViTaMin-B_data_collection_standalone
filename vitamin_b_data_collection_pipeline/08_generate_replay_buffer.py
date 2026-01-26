import sys
import os
import argparse
from pathlib import Path
import re
import csv

# Get project root (VB-vla/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # VB-vla/
DATA_DIR = PROJECT_ROOT / "data"

sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from utils.config_utils import get_mandatory_config
import pathlib
import click
import zarr
import pickle
import numpy as np
import numcodecs
import cv2
import av
from typing import Union
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from utils.cv_util import get_fisheye_image_transform, get_tactile_image_transform, inpaint_tag, draw_fisheye_mask
from utils.pose_util import mat_to_pose, pose_to_mat
from utils.replay_buffer import ReplayBuffer
from utils.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

def load_tactile_points(demo_dir, usage_name, total_frames):
    tactile_points_dir = demo_dir / 'tactile_points'
    points_file = tactile_points_dir / f'{usage_name}_points.npy'
    
    if not points_file.exists():
        raise FileNotFoundError(f"Tactile points file not found: {points_file}")
    
    points_data = np.load(points_file, allow_pickle=True)
    
    if len(points_data) != total_frames:
        raise ValueError(f"Points data length ({len(points_data)}) does not match video frames ({total_frames}) for {points_file}")
    
    # print(f"      [SUCCESS] Loaded tactile points: {len(points_data)} frames from {points_file.name}")
    return points_data

def main(input_path: str, output_path: str, visual_out_res: Union[tuple, list], tactile_out_res: Union[tuple, list], compression_level: int, num_workers: int, use_mask: bool, use_inpaint_tag: bool, use_tactile_img: bool, use_tactile_pc: bool, tag_scale: float, use_ee_pose: bool, tx_quest_2_ee_left_path: str, tx_quest_2_ee_right_path: str, fps_num_points: int, fisheye_mask_params: dict = None):
    if os.path.isfile(output_path):
        if sys.stdin.isatty():
            if click.confirm(f'Output file {output_path} exists! Overwrite?', abort=True):
                pass
        else:
            print(f'[WARNING] Output file {output_path} exists! Automatically overwriting in pipeline mode...')
            os.remove(output_path)
        
    visual_out_res = tuple(int(x) for x in visual_out_res)
    tactile_out_res = tuple(int(x) for x in tactile_out_res)

    cv2.setNumThreads(1)
            
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    n_grippers: Union[int, None] = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    for ipath in input_path:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))
        
        print(f"Processing {len(plan)} episodes from {plan_path}")
        
        videos_dict = defaultdict(list)
        for episode_idx, plan_episode in enumerate(plan):
            grippers = plan_episode['grippers']
            try:
                demo_mode = plan_episode['demo_mode']
            except KeyError:
                demo_mode = 'single'
            
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
                
            cameras = plan_episode['cameras']
            
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
                
            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):
                robot_id = gripper_id    
                quest_pose = gripper['quest_pose']
                
                # Debug: Check quest_pose before transformation
                if gripper_id == 0 and episode_idx == 0:
                    print(f"  [DEBUG] Episode {episode_idx} ({plan_episode.get('demo_name', 'unknown')}):")
                    print(f"    quest_pose shape: {quest_pose.shape}")
                    print(f"    quest_pose[0]: {quest_pose[0]}")
                    print(f"    quest_pose min/max: {quest_pose.min():.6f}/{quest_pose.max():.6f}")
                
                if use_ee_pose:
                    if n_grippers == 1:
                        tx_path = tx_quest_2_ee_left_path
                    else:
                        if gripper_id == 0:
                            tx_path = tx_quest_2_ee_left_path
                        else:
                            tx_path = tx_quest_2_ee_right_path
                    
                    if not os.path.exists(tx_path):
                        raise FileNotFoundError(f"Tx file not found: {tx_path}")
                    else:
                        tx_quest_2_ee = np.load(tx_path)
                        # Use old formula: pose_to_mat(quest_pose) @ np.linalg.inv(tx_quest_2_ee)
                        eef_pose = mat_to_pose(pose_to_mat(quest_pose) @ np.linalg.inv(tx_quest_2_ee))
                else:
                    eef_pose = quest_pose
                
                # Debug: Check eef_pose after transformation
                if gripper_id == 0 and episode_idx == 0:
                    print(f"    eef_pose shape: {eef_pose.shape}")
                    print(f"    eef_pose[0]: {eef_pose[0]}")
                    print(f"    eef_pose min/max: {eef_pose.min():.6f}/{eef_pose.max():.6f}")
                    
                eef_pos = eef_pose[...,:3]
                eef_rot = eef_pose[...,3:]
                gripper_widths = gripper['gripper_width']
                
                # Broadcast demo_start_pose and demo_end_pose to match eef_pose shape
                # Use NumPy's automatic broadcasting - simple and correct
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']
                
                robot_name = f'robot{robot_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose
                
                # Debug: Check data before adding to buffer
                if gripper_id == 0 and episode_idx == 0:
                    print(f"    episode_data['{robot_name}_eef_pos'][0]: {episode_data[robot_name + '_eef_pos'][0]}")
                    print(f"    episode_data['{robot_name}_eef_pos'] shape: {episode_data[robot_name + '_eef_pos'].shape}")
            
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # Debug: Check data after adding to buffer (only for first episode)
            if episode_idx == 0:
                print(f"    After add_episode, robot0_eef_pos[:3]:")
                print(f"      {out_replay_buffer['robot0_eef_pos'][:3]}")
                print(f"    robot0_eef_pos shape: {out_replay_buffer['robot0_eef_pos'].shape}")
                print(f"    robot0_eef_pos[0]: {out_replay_buffer['robot0_eef_pos'][0]}")
            
            n_frames: Union[int, None] = None
            for cam_id, camera in enumerate(cameras):
                # Image-folder-only pipeline: each camera points to a folder of JPGs
                video_path_rel = camera['image_folder']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_dir(), f"Image folder not found: {video_path}"
                
                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)
                
                try:
                    usage_name = camera['usage_name']
                except KeyError:
                    usage_name = f'camera{cam_id}'
                
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'usage_name': usage_name,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })
            assert n_frames is not None
            buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")
    
    if len(vid_args) == 0:
        print("No valid videos found. Please run script 09 to generate dataset plan first.")
        exit(1)
    
    # Check resolution of different video types
    visual_input_res = None
    tactile_input_res = None
    
    for mp4_path, tasks in vid_args:
        # Here mp4_path is actually an image folder path
        usage_name = tasks[0]['usage_name']

        img_files = sorted(Path(mp4_path).glob("*.jpg"))
        if not img_files:
            continue
        img = cv2.imread(str(img_files[0]))
        if img is None:
            continue
        h, w = img.shape[:2]
        current_res = (w, h)

        if 'visual' in usage_name and visual_input_res is None:
            visual_input_res = current_res
        elif 'tactile' in usage_name and tactile_input_res is None:
            tactile_input_res = current_res
        
        # Stop checking if both are found
        if visual_input_res and tactile_input_res:
            break
    
    # If no visual folder found, use first folder's resolution
    if visual_input_res is None:
        first_path = vid_args[0][0]
        print(f"Loading visual input metadata from folder: {first_path}")
        img_files = sorted(Path(first_path).glob("*.jpg"))
        if not img_files:
            raise FileNotFoundError(f"No images found in {first_path}")
        img = cv2.imread(str(img_files[0]))
        if img is None:
            raise RuntimeError(f"Failed to read first image in {first_path}")
        h, w = img.shape[:2]
        visual_input_res = (w, h)
    
    # Set global variables for visual image transformation
    iw, ih = visual_input_res
    
    print(f"Visual input resolution: {visual_input_res[0]}x{visual_input_res[1]}")
    print(f"Visual output resolution: {visual_out_res[0]}x{visual_out_res[1]}")
    
    if use_tactile_img or use_tactile_pc:
        if tactile_input_res:
            print(f"Tactile input resolution: {tactile_input_res[0]}x{tactile_input_res[1]}")
        print(f"Tactile output resolution: {tactile_out_res[0]}x{tactile_out_res[1]}")
        print(f"Tactile image processing: {'ENABLED' if use_tactile_img else 'DISABLED'}")
        print(f"Tactile point cloud processing: {'ENABLED' if use_tactile_pc else 'DISABLED'}")
    else:
        print("Tactile processing is DISABLED")
    
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    
    hand_position_to_usage = {}
    for mp4_path, tasks in vid_args:
        for task in tasks:
            usage_name = task['usage_name']
            
            if 'left_hand' in usage_name:
                hand_idx = 0
            elif 'right_hand' in usage_name:
                hand_idx = 1
            elif 'visual' in usage_name:
                if usage_name == 'visual' or usage_name == 'left_visual':
                    hand_idx = 0
                elif usage_name == 'right_hand_visual' or usage_name == 'right_visual':
                    hand_idx = 1
                else:
                    try:
                        hand_idx = task['hand_position_idx']
                    except KeyError:
                        hand_idx = 0
            else:
                try:
                    hand_idx = task['hand_position_idx']
                except KeyError:
                    hand_idx = 0
            
            if hand_idx not in hand_position_to_usage:
                hand_position_to_usage[hand_idx] = set()
            hand_position_to_usage[hand_idx].add(usage_name)
    
    print(f"Hand positions found: {sorted(hand_position_to_usage.keys())}")
    for hand_idx, usage_names in hand_position_to_usage.items():
        print(f"  Hand {hand_idx}: {sorted(usage_names)}")
    
    for hand_idx in sorted(hand_position_to_usage.keys()):
        usage_names = hand_position_to_usage[hand_idx]
        
        visual_usage_names = [name for name in usage_names if 'visual' in name]
        if visual_usage_names:
            rgb_name = f'camera{hand_idx}_rgb'
            _ = out_replay_buffer.data.require_dataset(
                name=rgb_name,
                shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + visual_out_res + (3,),
                chunks=(1,) + visual_out_res + (3,),
                compressor=img_compressor,
                dtype=np.uint8
            )
            print(f"Created visual dataset: {rgb_name} with shape {out_replay_buffer.data[rgb_name].shape} (from {visual_usage_names})")
        
        if use_tactile_img or use_tactile_pc:
            tactile_names = [name for name in usage_names if 'tactile' in name]
            for usage_name in tactile_names:
                parts = usage_name.split('_')
                if len(parts) >= 4 and parts[1] == 'hand' and parts[3] == 'tactile':
                    sensor_side = parts[2]
                    
                    # Create tactile image dataset
                    if use_tactile_img:
                        tactile_dataset_name = f'camera{hand_idx}_{sensor_side}_tactile'
                        _ = out_replay_buffer.data.require_dataset(
                            name=tactile_dataset_name,
                            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + tactile_out_res + (3,),
                            chunks=(1,) + tactile_out_res + (3,),
                            compressor=img_compressor,
                            dtype=np.uint8
                        )
                        print(f"Created tactile image dataset: {tactile_dataset_name} with shape {out_replay_buffer.data[tactile_dataset_name].shape} (from {usage_name})")
                    
                    # Create tactile point cloud dataset
                    if use_tactile_pc:
                        tactile_points_dataset_name = f'camera{hand_idx}_{sensor_side}_tactile_points'
                        # Use fixed-shape array instead of object array, using fps_num_points parameter from config
                        _ = out_replay_buffer.data.require_dataset(
                            name=tactile_points_dataset_name,
                            shape=(out_replay_buffer['robot0_eef_pos'].shape[0], fps_num_points, 3),
                            chunks=(1, fps_num_points, 3),
                            compressor=None,
                            dtype=np.float32
                        )
                        print(f"Created tactile points dataset: {tactile_points_dataset_name} with shape {out_replay_buffer.data[tactile_points_dataset_name].shape} (from {usage_name}), fps_num_points={fps_num_points}")

    def video_to_zarr(replay_buffer, mp4_path, tasks, tag_scale, fisheye_mask_params=None, use_tactile_img=True, use_tactile_pc=True, fps_num_points=256):
        """
        Image-folder-only version: mp4_path is a directory containing JPG frames.
        """
        mp4_path = str(mp4_path)
        video_path = pathlib.Path(mp4_path)
        # video_path is like: demos/demo_xxx/left_hand_visual_img
        # demo_dir should be: demos/demo_xxx (where CSV files are located)
        demo_dir = video_path.parent
        
        usage_name = tasks[0]['usage_name']
        
        # Determine which hand this camera belongs to for CSV lookup
        # For tactile: usage_name format is like "left_hand_left_tactile" or "left_hand_right_tactile"
        # For visual: usage_name format is like "left_visual" or "right_hand_visual"
        if usage_name.startswith('left_hand') or (usage_name.startswith('left_') and 'visual' in usage_name):
            hand = 'left'
        elif usage_name.startswith('right_hand') or (usage_name.startswith('right_') and 'visual' in usage_name):
            hand = 'right'
        elif 'left' in usage_name.split('_')[:2]:  # Check first two parts for "left"
            hand = 'left'
        elif 'right' in usage_name.split('_')[:2]:  # Check first two parts for "right"
            hand = 'right'
        else:
            # Fallback: try to get from task
            hand = tasks[0].get('position', 'left')
        
        # Load image files in the order specified by CSV timestamps file
        csv_file = demo_dir / f'{hand}_hand_timestamps.csv'
        img_files = []
        csv_has_filename = False
        
        if csv_file.exists():
            # Read CSV to get the correct order of image files
            with csv_file.open("r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                csv_has_filename = "filename" in fieldnames
                
                if csv_has_filename:
                    # Use filename from CSV to maintain correct order
                    for row in reader:
                        filename = row.get("filename", "")
                        if filename:
                            img_path = video_path / filename
                            if img_path.exists():
                                img_files.append(img_path)
                            else:
                                # Try without path if filename already includes path info
                                img_path_alt = Path(mp4_path) / Path(filename).name
                                if img_path_alt.exists():
                                    img_files.append(img_path_alt)
        
        # Fallback: if CSV doesn't exist or doesn't have filename field, sort by numeric ID
        if not img_files:
            if csv_file.exists() and not csv_has_filename:
                print(f"      [WARN] CSV file {csv_file.name} exists but has no 'filename' field, falling back to filename sorting")
            elif not csv_file.exists():
                print(f"      [WARN] CSV file not found: {csv_file}, falling back to filename sorting")
            
            # Sort by numeric ID in filename (e.g. left_hand_12.jpg -> 12)
            all_img_files = list(Path(mp4_path).glob("*.jpg"))
            img_files = sorted(
                all_img_files,
                key=lambda p: int(re.search(r'(\d+)(?=\.jpg$)', p.name).group(1))
                if re.search(r'(\d+)(?=\.jpg$)', p.name) else p.name
            )
        
        if not img_files:
            print(f"      [SKIP] No images found in folder: {mp4_path}")
            return
        img0 = cv2.imread(str(img_files[0]))
        if img0 is None:
            print(f"      [SKIP] Failed to read first image in folder: {mp4_path}")
            return
        ih0, iw0 = img0.shape[:2]
        actual_iw, actual_ih = iw0, ih0
        total_frames = len(img_files)
        
        visual_resize_tf = get_fisheye_image_transform(
            in_res=(iw, ih),
            out_res=visual_out_res
        )
        
        tactile_resize_tf = None
        if use_tactile_img:
            tactile_resize_tf = get_tactile_image_transform(
                in_res=(actual_iw, actual_ih),
                out_res=tactile_out_res
            )
        
        tag_detection_results = None
        video_filename = video_path.name
        
        if 'right_hand_visual' in video_filename or usage_name == 'right_visual':
            aruco_pkl_path = demo_dir / 'tag_detection_right.pkl'
        else:
            aruco_pkl_path = demo_dir / 'tag_detection_left.pkl'
        
        if aruco_pkl_path.exists():
            try:
                tag_detection_results = pickle.load(open(aruco_pkl_path, 'rb'))
            except Exception as e:
                print(f"Warning: Could not load ArUco detection from {aruco_pkl_path}: {e}")
        else:
            print(f"Warning: ArUco detection file not found: {aruco_pkl_path.name}")
        
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        
        usage_name = tasks[0]['usage_name']
        
        if 'left_hand' in usage_name:
            hand_position_idx = 0
        elif 'right_hand' in usage_name:
            hand_position_idx = 1
        elif 'visual' in usage_name:
            if usage_name == 'visual' or usage_name == 'left_visual':
                hand_position_idx = 0
            elif usage_name == 'right_hand_visual' or usage_name == 'right_visual':
                hand_position_idx = 1
            else:
                try:
                    hand_position_idx = tasks[0]['hand_position_idx']
                except KeyError:
                    hand_position_idx = 0
        else:
            hand_position_idx = tasks[0].get('hand_position_idx', 0)
        
        if 'visual' in usage_name:
            dataset_name = f'camera{hand_position_idx}_rgb'
        elif 'tactile' in usage_name:
            if not use_tactile_img:
                # If not processing tactile images, skip tactile videos
                print(f"      [SKIP] Skipping tactile image processing for {usage_name} (use_tactile_img=False)")
                return
            parts = usage_name.split('_')
            if len(parts) >= 4 and parts[1] == 'hand' and parts[3] == 'tactile':
                sensor_side = parts[2]
                dataset_name = f'camera{hand_position_idx}_{sensor_side}_tactile'
            else:
                raise ValueError(f"Invalid tactile usage_name format: {usage_name} for video {mp4_path}")
        else:
            raise ValueError(f"Unknown usage_name: {usage_name} for video {mp4_path}")
        
        if dataset_name not in replay_buffer.data:
            raise KeyError(f"Dataset {dataset_name} not found in replay buffer for video {mp4_path}")
            
        img_array = replay_buffer.data[dataset_name]
        
        points_array = None
        if 'tactile' in usage_name and use_tactile_pc:
            parts = usage_name.split('_')
            if len(parts) >= 4 and parts[1] == 'hand' and parts[3] == 'tactile':
                sensor_side = parts[2]
                points_dataset_name = f'camera{hand_position_idx}_{sensor_side}_tactile_points'
                if points_dataset_name in replay_buffer.data:
                    points_array = replay_buffer.data[points_dataset_name]
                    # print(f"      [INFO] Will process tactile points for dataset: {points_dataset_name}")
        
        if 'visual' in usage_name:
            resize_tf = visual_resize_tf
        else:
            resize_tf = tactile_resize_tf if tactile_resize_tf else visual_resize_tf
        
        curr_task_idx = 0
        
        last_detected_corners = {}
        
        tactile_points_data = None
        if 'tactile' in usage_name and use_tactile_pc and points_array is not None:
            tactile_points_data = load_tactile_points(demo_dir, usage_name, total_frames)
        
        buffer_idx = 0

        # Image folder iteration only
        frame_iter = enumerate(img_files)
        total_iter = len(img_files)

        for frame_idx, img_path in tqdm(frame_iter, total=total_iter, leave=False):
                if curr_task_idx >= len(tasks):
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        continue
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    if 'visual' in usage_name:
                        if use_inpaint_tag:
                            if tag_detection_results is not None:
                                if frame_idx < len(tag_detection_results):
                                    this_det = tag_detection_results[frame_idx]
                                    current_frame_corners = []
                                    
                                    if 'tag_dict' in this_det and this_det['tag_dict']:
                                        for tag_id, tag_info in this_det['tag_dict'].items():
                                            if 'corners' in tag_info:
                                                corners = tag_info['corners']
                                                current_frame_corners.append(corners)
                                                last_detected_corners[tag_id] = corners
                                    
                                    for tag_id, cached_corners in last_detected_corners.items():
                                        tag_detected_in_current = False
                                        if 'tag_dict' in this_det and this_det['tag_dict']:
                                            tag_detected_in_current = tag_id in this_det['tag_dict']
                                        
                                        if not tag_detected_in_current:
                                            import random
                                            offset_range = 2
                                            jittered_corners = []
                                            for corner in cached_corners:
                                                jittered_corner = [
                                                    corner[0] + random.uniform(-offset_range, offset_range),
                                                    corner[1] + random.uniform(-offset_range, offset_range)
                                                ]
                                                jittered_corners.append(jittered_corner)
                                            current_frame_corners.append(jittered_corners)
                                    
                                    for corners in current_frame_corners:
                                        img = inpaint_tag(img, tag_scale=tag_scale, corners=corners)
                        
                        if 'visual' in usage_name and use_mask:
                            if fisheye_mask_params is None:
                                fisheye_mask_params = {}
                            
                            radius = get_mandatory_config(fisheye_mask_params, "radius", "10_generate_replay_buffer.py - fisheye mask")
                            center = fisheye_mask_params.get('center', None)
                            fill_color = get_mandatory_config(fisheye_mask_params, "fill_color", "10_generate_replay_buffer.py - fisheye mask")
                            
                            img = draw_fisheye_mask(img, radius=radius, center=center, fill_color=fill_color)

                    img = resize_tf(img)
                    
                    img_array[buffer_idx] = img
                    
                    if tactile_points_data is not None and points_array is not None:
                        frame_points = tactile_points_data[frame_idx]
                        # Check if frame_points is valid (not None and has data)
                        if frame_points is not None and (
                            (isinstance(frame_points, (list, tuple)) and len(frame_points) > 0) or
                            (isinstance(frame_points, np.ndarray) and frame_points.size > 0)
                        ):
                            # Convert to numpy array and ensure shape is (fps_num_points, 3)
                            if isinstance(frame_points, np.ndarray):
                                frame_points_array = frame_points.astype(np.float32)
                            else:
                                frame_points_array = np.array(frame_points, dtype=np.float32)
                            
                            if frame_points_array.shape == (fps_num_points, 3):
                                points_array[buffer_idx] = frame_points_array
                            else:
                                print(f"Warning: Frame {frame_idx} has unexpected point cloud shape {frame_points_array.shape}, expected ({fps_num_points}, 3)")
                                # If points < fps_num_points, pad with zeros; if > fps_num_points, truncate
                                if len(frame_points_array) < fps_num_points:
                                    padded_points = np.zeros((fps_num_points, 3), dtype=np.float32)
                                    padded_points[:len(frame_points_array)] = frame_points_array
                                    points_array[buffer_idx] = padded_points
                                else:
                                    points_array[buffer_idx] = frame_points_array[:fps_num_points]
                        else:
                            # If no point cloud data, fill with zero array
                            points_array[buffer_idx] = np.zeros((fps_num_points, 3), dtype=np.float32)
                    
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        curr_task_idx += 1
                else:
                    assert False

    with tqdm(total=len(vid_args)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks, tag_scale=tag_scale, fisheye_mask_params=fisheye_mask_params, use_tactile_img=use_tactile_img, use_tactile_pc=use_tactile_pc, fps_num_points=fps_num_points))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    try:
        results = [x.result() for x in completed]
        if any([r is not None for r in results]):
            print("Errors occurred during video processing:", [r for r in results if r is not None])
    except Exception as e:
        print(f"Error while processing results: {str(e)}")

    print(f"\nReplay Buffer Statistics:")
    print(f"   Total episodes: {out_replay_buffer.n_episodes}")
    if out_replay_buffer.n_episodes > 0:
        print(f"   Episode length range: {out_replay_buffer['robot0_eef_pos'].shape[1]} frames")
        print(f"   Number of robots: {n_grippers}")
        print(f"   Number of cameras: {n_cameras}")

        if use_tactile_img or use_tactile_pc:
            tactile_img_datasets = [key for key in out_replay_buffer.data.keys() if 'tactile' in key and 'points' not in key]
            tactile_pc_datasets = [key for key in out_replay_buffer.data.keys() if 'tactile_points' in key]
            if use_tactile_img:
                print(f"   Number of tactile image datasets: {len(tactile_img_datasets)}")
            if use_tactile_pc:
                print(f"   Number of tactile point cloud datasets: {len(tactile_pc_datasets)}")
        
        print(f"\n   Dataset details:")
        for key in sorted(out_replay_buffer.data.keys()):
            data = out_replay_buffer.data[key]
            if "rgb" in key:
                dataset_type = "RGB"
            elif "tactile_points" in key:
                dataset_type = "Points"
            elif "tactile" in key:
                dataset_type = "Tactile"
            else:
                dataset_type = "Data"
            print(f"   {dataset_type} {key}: {data.shape} ({data.dtype})")

    print(f"\nSaving ReplayBuffer to {output_path}")
    with zarr.ZipStore(output_path, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"[SUCCESS] Done! Generated replay buffer from {len(all_videos)} videos")
    print(f"   Output file: {output_path}")

    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024**3)
        print(f"   File size: {file_size:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate replay buffer from dataset plan for training (New traj.csv format)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    config_file = args.cfg
    cfg = OmegaConf.load(config_file)
    
    # Resolve relative paths (relative to config file location)
    config_path = Path(config_file).resolve()
    config_dir = config_path.parent
    
    task_name = cfg["task"]["name"]
    visual_out_res = cfg["output_train_data"]["visual_out_res"]
    tactile_out_res = cfg["output_train_data"]["tactile_out_res"]
    use_mask = cfg["output_train_data"]["use_mask"]
    use_inpaint_tag = cfg["output_train_data"]["use_inpaint_tag"]
    use_tactile_img = cfg["output_train_data"]["use_tactile_img"]
    use_tactile_pc = cfg["output_train_data"]["use_tactile_pc"]
    compression_level = cfg["output_train_data"]["compression_level"]
    num_workers = cfg["output_train_data"]["num_workers"]
    use_ee_pose = cfg["output_train_data"]["use_ee_pose"]
    
    # Resolve transformation file paths to absolute
    tx_quest_2_ee_left_path = cfg["output_train_data"]["tx_quest_2_ee_left_path"]
    tx_quest_2_ee_right_path = cfg["output_train_data"]["tx_quest_2_ee_right_path"]
    if not Path(tx_quest_2_ee_left_path).is_absolute():
        tx_quest_2_ee_left_path = str((config_dir / tx_quest_2_ee_left_path).resolve())
    if not Path(tx_quest_2_ee_right_path).is_absolute():
        tx_quest_2_ee_right_path = str((config_dir / tx_quest_2_ee_right_path).resolve())
    
    min_episode_length = cfg["output_train_data"]["min_episode_length"]
    tag_scale = cfg["output_train_data"]["tag_scale"]
    fisheye_mask_params = get_mandatory_config(cfg, ["output_train_data", "fisheye_mask_params"], "10_generate_replay_buffer.py")
    
    # Read FPS point cloud sampling parameters
    fps_num_points = cfg["tactile_point_extraction"]["fps_num_points"]
    print(f"FPS point cloud sampling count: {fps_num_points}")
    
    input_path = DATA_DIR / task_name
    
    output_filename = f"{task_name}.zarr.zip"
    
    output_path = DATA_DIR / task_name / output_filename
    
    print(f"Task: {task_name}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    print(f"Visual resolution: {visual_out_res}")
    print(f"Tactile resolution: {tactile_out_res}")
    print(f"Use inpaint tag: {use_inpaint_tag}")
    print(f"Use tactile images: {use_tactile_img}")
    print(f"Use tactile point clouds: {use_tactile_pc}")
    print(f"Compression level: {compression_level}")
    print(f"Workers: {num_workers}")
    print(f"Use ee pose: {use_ee_pose}")

    print(f"Use mask (For visual): {use_mask}")
    if use_mask and fisheye_mask_params:
        print(f"Fisheye mask params: {fisheye_mask_params}")

    main([input_path], output_path, visual_out_res, tactile_out_res, 
         compression_level, num_workers, use_mask, use_inpaint_tag, 
         use_tactile_img, use_tactile_pc, tag_scale, use_ee_pose, 
         tx_quest_2_ee_left_path, tx_quest_2_ee_right_path, fps_num_points, fisheye_mask_params)