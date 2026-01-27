"""
Data Acquisition Script for ViTaMin-B VLA Data Collection Pipeline
ViTaMin-B VLA 数据采集管道的数据获取脚本

This script combines camera image capture and Quest VR controller pose capture
into a unified recording system with real-time FPS monitoring and TCP control.
该脚本将相机图像采集和Quest VR控制器姿态采集整合到统一的录制系统中，
具有实时FPS监控和TCP控制功能。

Main Components / 主要组件:
- StatusBoard: Real-time status display / 实时状态显示
- BatchSaver: Quest pose data batch writer / Quest姿态数据批量写入器
- ControlClient: TCP client for Quest control / Quest控制的TCP客户端
- QuestPoseReceiver: Background pose data receiver / 后台姿态数据接收器
- CameraWorker: Multi-threaded camera capture / 多线程相机采集
- DataRecorder: Image recording manager / 图像录制管理器
- CombinedRunner: Main orchestrator / 主协调器
"""

import argparse
import json
import os
import select
import sys
import threading
import time
import tty
import termios
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from queue import Queue
import threading as _threading

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Add project root to Python path / 将项目根目录添加到Python路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from utils.camera_device import V4L2Camera


class StatusBoard:
    """
    Lightweight split-screen status printer for camera/pose.
    轻量级分屏状态打印器，用于显示相机和姿态信息
    
    Displays real-time FPS for cameras and pose receiver in a two-column layout.
    以双列布局显示相机和姿态接收器的实时FPS。
    """

    def __init__(self):
        # Thread lock for safe concurrent updates / 线程锁，用于安全的并发更新
        self.lock = _threading.Lock()
        # Camera statistics: {camera_name: (fps, total_frames)} / 相机统计信息
        self.camera_stats: Dict[str, Tuple[float, int]] = {}
        # Pose statistics: (fps, packet_count) / 姿态统计信息
        self.pose_stat: Tuple[float, int] = (0.0, 0)
        self.last_render = 0.0
        self.min_interval = 0.2  # Minimum render interval in seconds / 最小渲染间隔（秒）
        self.demo_status = "Idle"  # Current recording status / 当前录制状态

    def update_camera(self, name: str, fps: float, total: int):
        """
        Update camera FPS and total frame count.
        更新相机FPS和总帧数
        """
        with self.lock:
            self.camera_stats[name] = (fps, total)
            self._render_locked()

    def update_pose(self, fps: float, packets: int):
        """
        Update pose receiver FPS and packet count.
        更新姿态接收器FPS和数据包数量
        """
        with self.lock:
            self.pose_stat = (fps, packets)
            self._render_locked()

    def set_demo_status(self, text: str):
        """
        Set the current demo status (Idle/Recording/Stopped).
        设置当前演示状态（空闲/录制中/已停止）
        """
        with self.lock:
            self.demo_status = text
            self._render_locked()

    def _render_locked(self):
        """
        Render the status display (throttled to min_interval).
        渲染状态显示（限制到最小间隔）
        
        Creates a two-column layout showing camera and pose statistics.
        创建双列布局，显示相机和姿态统计信息。
        """
        now = time.time()
        # Throttle rendering to avoid excessive terminal updates / 限制渲染以避免过多终端更新
        if now - self.last_render < self.min_interval:
            return
        self.last_render = now

        # Build left column (cameras) / 构建左列（相机信息）
        cam_lines = ["Camera FPS"]
        for name in sorted(self.camera_stats.keys()):
            fps, total = self.camera_stats[name]
            cam_lines.append(f"{name:>10}: {fps:5.1f}  (total {total})")
        if len(cam_lines) == 1:
            cam_lines.append("  waiting...")

        # Build right column (pose) / 构建右列（姿态信息）
        pose_fps, pose_packets = self.pose_stat
        pose_lines = ["Pose", f"fps: {pose_fps:5.1f} Hz", f"packets: {pose_packets}"]

        # Pad to equal height / 填充到相同高度
        height = max(len(cam_lines), len(pose_lines))
        cam_lines += [""] * (height - len(cam_lines))
        pose_lines += [""] * (height - len(pose_lines))

        # Compose two-column layout / 组合双列布局
        combined = ["\033[2J\033[H"]  # ANSI: clear screen & move cursor home / 清屏并移动光标到起始位置
        for cl, pl in zip(cam_lines, pose_lines):
            combined.append(f"{cl:<32} | {pl}")
        combined.append("")
        combined.append(f"State: {self.demo_status}")
        combined.append("[S] start/stop  [Q] quit")

        sys.stdout.write("\n".join(combined) + "\n")
        sys.stdout.flush()

 
class BatchSaver:
    """
    Batch writer for Quest pose data.
    Quest姿态数据的批量写入器
    
    Saves pose data in batches to JSON files to avoid frequent I/O operations.
    将姿态数据批量保存到JSON文件，以避免频繁的I/O操作。
    """

    def __init__(self, save_dir: str, frames_per_file: int = 5000):
        self.save_dir = Path(save_dir)
        self.frames_per_file = frames_per_file  # Number of frames per file / 每个文件的帧数
        # Session timestamp for unique filenames / 会话时间戳，用于唯一的文件名
        self.session_timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
        self.file_counter = 1  # File part counter / 文件部分计数器
        self.current_batch = []  # Current batch buffer / 当前批次缓冲区
        self.total_saved_frames = 0  # Total frames saved in session / 会话中保存的总帧数
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def add(self, pose_data, receive_timestamp: float):
        """
        Add a pose data frame to the current batch.
        将姿态数据帧添加到当前批次
        
        Args:
            pose_data: Dict containing Quest pose information / 包含Quest姿态信息的字典
            receive_timestamp: Unix timestamp when data was received / 接收数据时的Unix时间戳
        """
        # Add timestamps to pose data / 向姿态数据添加时间戳
        pose_data["timestamp_unix"] = receive_timestamp
        pose_data["timestamp_readable"] = datetime.fromtimestamp(receive_timestamp).strftime(
            "%Y.%m.%d_%H.%M.%S.%f"
        )
        self.current_batch.append(pose_data)
        self.total_saved_frames += 1

        # Save batch when reaching target size / 达到目标大小时保存批次
        if len(self.current_batch) >= self.frames_per_file:
            self._save_batch()

    def _save_batch(self):
        """
        Save current batch to a JSON file.
        将当前批次保存到JSON文件
        """
        if not self.current_batch:
            return

        # Generate filename with session timestamp and part number / 使用会话时间戳和部分编号生成文件名
        filename = f"quest_poses_{self.session_timestamp}_part{self.file_counter:03d}.json"
        with open(self.save_dir / filename, "w") as f:
            json.dump(self.current_batch, f, indent=2)

        print(f"\nSaved {filename} ({len(self.current_batch)} frames)")
        self.current_batch = []
        self.file_counter += 1

    def finalize(self):
        """
        Finalize the session by saving remaining frames and creating a summary.
        完成会话，保存剩余帧并创建摘要
        """
        # Save any remaining frames / 保存任何剩余帧
        if self.current_batch:
            self._save_batch()

        # Create session summary / 创建会话摘要
        summary = {
            "session_timestamp": self.session_timestamp,
            "total_frames": self.total_saved_frames,
            "total_files": self.file_counter - 1,
            "frames_per_file": self.frames_per_file,
        }

        with open(self.save_dir / f"session_summary_{self.session_timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2)


class ControlClient:
    """
    Lightweight TCP client to notify Quest of demo start/stop and FPS updates.
    轻量级TCP客户端，用于通知Quest演示的开始/停止和FPS更新
    
    Sends control messages to Quest app for synchronization and monitoring.
    向Quest应用发送控制消息以进行同步和监控。
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.last_fps_send_time = 0.0
        self.fps_send_interval = 0.5  # Send FPS every 0.5 seconds / 每0.5秒发送一次FPS

    def _ensure_connected(self) -> bool:
        """
        Ensure TCP connection to Quest is established.
        确保与Quest的TCP连接已建立
        
        Returns:
            bool: True if connected, False otherwise / 如果已连接返回True，否则返回False
        """
        if self.sock:
            return True
        try:
            # Create TCP connection with 1 second timeout / 创建1秒超时的TCP连接
            self.sock = socket.create_connection((self.host, self.port), timeout=1.0)
            # Disable Nagle's algorithm for low latency / 禁用Nagle算法以实现低延迟
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return True
        except Exception as e:
            print(f"[CTRL][ERROR] connect failed: {e}")
            self.sock = None
            return False

    def send_demo_control(self, demo_id: int, action: str):
        """
        Send demo start/stop control message to Quest.
        向Quest发送演示开始/停止控制消息
        
        Args:
            demo_id: Demo number / 演示编号
            action: "start" or "stop" / "开始"或"停止"
        """
        if action not in ("start", "stop"):
            return
        if not self._ensure_connected():
            return
        
        # Construct control payload / 构建控制负载
        payload = {
            "type": "demo_control",
            "demo_id": demo_id,
            "action": action,
            "timestamp": time.time(),
        }
        try:
            msg = json.dumps(payload) + "\n"
            self.sock.sendall(msg.encode("utf-8"))
            print(f"[CTRL] sent {action} for demo #{demo_id}")
        except Exception as e:
            print(f"[CTRL][ERROR] send failed: {e}")
            if self.sock:
                try:
                    self.sock.close()
                finally:
                    self.sock = None

    def send_fps_update(self, camera_fps: Dict[str, float], pose_fps: float):
        """
        Send FPS update message (throttled to avoid flooding).
        发送FPS更新消息（限制发送频率以避免泛洪）
        
        Args:
            camera_fps: Dict of camera name to FPS / 相机名称到FPS的字典
            pose_fps: Pose receiver FPS / 姿态接收器FPS
        """
        now = time.time()
        # Throttle FPS updates to avoid excessive messages / 限制FPS更新以避免过多消息
        if now - self.last_fps_send_time < self.fps_send_interval:
            return
        
        if not self._ensure_connected():
            return
        
        payload = {
            "type": "fps_update",
            "camera_fps": camera_fps,
            "pose_fps": pose_fps,
            "timestamp": now,
        }
        try:
            msg = json.dumps(payload) + "\n"
            self.sock.sendall(msg.encode("utf-8"))
            self.last_fps_send_time = now
        except Exception as e:
            # Silently fail for FPS updates to avoid spam / 静默失败以避免垃圾消息
            if self.sock:
                try:
                    self.sock.close()
                finally:
                    self.sock = None

    def close(self):
        """
        Close the TCP connection.
        关闭TCP连接
        """
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None


class QuestPoseReceiver(threading.Thread):
    """
    Background receiver for Quest pose data with on/off recording.
    Quest姿态数据的后台接收器，支持开关录制
    
    Runs as a daemon thread to continuously receive pose data from Quest via TCP.
    作为守护线程运行，通过TCP持续从Quest接收姿态数据。
    Receives JSON packets containing head pose, left/right wrist poses.
    接收包含头部姿态、左右手腕姿态的JSON数据包。
    """

    def __init__(
        self,
        host: str,
        port: int,
        base_output_dir: Path,
        frames_per_file: int = 5000,
        stop_event: Optional[threading.Event] = None,
        status_board: Optional[StatusBoard] = None,
    ):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.base_output_dir = base_output_dir
        self.frames_per_file = frames_per_file
        self.stop_event = stop_event or threading.Event()
        self.status_board = status_board

        self.socket = None
        self.buffer = ""  # TCP receive buffer / TCP接收缓冲区
        self.buffer_max_size = 10 * 1024 * 1024  # 10MB buffer limit / 10MB缓冲区限制
        self.recording = False  # Recording state / 录制状态
        self.saver: Optional[BatchSaver] = None
        self.pose_count = 0  # Total pose packets received / 接收的总姿态数据包数
        self.window = []  # Sliding window for FPS calculation / FPS计算的滑动窗口
        self.window_size = 30  # Window size for FPS smoothing / FPS平滑的窗口大小
        self.last_display_time = 0.0
        self.lock = threading.Lock()

    def connect(self) -> bool:
        """
        Establish TCP connection to Quest.
        建立与Quest的TCP连接
        
        Requires ADB port forwarding: adb forward tcp:7777 tcp:7777
        需要ADB端口转发：adb forward tcp:7777 tcp:7777
        
        Returns:
            bool: True if connection successful / 连接成功返回True
        """
        try:
            import socket

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(1.0)  # 1 second timeout for recv() / recv()的1秒超时
            self.socket.connect((self.host, self.port))
            print(f"[POSE] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[POSE][ERROR] Failed to connect: {e}")
            print("        Check: adb forward tcp:7777 tcp:7777")
            return False

    def disconnect(self):
        """
        Close the TCP connection.
        关闭TCP连接
        """
        if self.socket:
            self.socket.close()
            self.socket = None

    def start_recording(self, demo_dir: Optional[Path] = None):
        """
        Start recording pose data to files.
        开始将姿态数据录制到文件
        
        Args:
            demo_dir: Target directory, defaults to base_output_dir / 目标目录，默认为base_output_dir
        """
        with self.lock:
            target_dir = demo_dir if demo_dir else self.base_output_dir
            pose_dir = Path(target_dir)
            pose_dir.mkdir(parents=True, exist_ok=True)
            self.saver = BatchSaver(str(pose_dir), frames_per_file=self.frames_per_file)
            self.recording = True
            print(f"[POSE] Recording to {pose_dir}")

    def stop_recording(self):
        """
        Stop recording and finalize saved files.
        停止录制并完成文件保存
        """
        with self.lock:
            if self.saver:
                self.saver.finalize()
                print(
                    f"[POSE] Saved {self.saver.total_saved_frames} frames "
                    f"in {self.saver.file_counter - 1} file(s)"
                )
            self.saver = None
            self.recording = False

    def parse_pose_data(self, json_str):
        """
        Parse JSON string into pose data dict.
        将JSON字符串解析为姿态数据字典
        
        Expected keys: head_pose, left_wrist, right_wrist, timestamp
        期望的键：head_pose, left_wrist, right_wrist, timestamp
        
        Args:
            json_str: JSON string from Quest / 来自Quest的JSON字符串
            
        Returns:
            dict or None: Parsed data if valid, None otherwise / 有效则返回解析的数据，否则返回None
        """
        try:
            data = json.loads(json_str.strip())
            # Validate required fields / 验证必需字段
            if all(k in data for k in ["head_pose", "left_wrist", "right_wrist", "timestamp"]):
                return data
            return None
        except Exception:
            return None

    def _update_fps_display(self):
        """
        Update FPS display using sliding window average.
        使用滑动窗口平均值更新FPS显示
        """
        now = time.time()
        self.pose_count += 1
        self.window.append(now)
        # Keep only recent timestamps for sliding window / 仅保留最近的时间戳用于滑动窗口
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size :]

        # Calculate FPS from window / 从窗口计算FPS
        if len(self.window) >= 2:
            dur = self.window[-1] - self.window[0]
            window_fps = (len(self.window) - 1) / dur if dur > 0 else 0
        else:
            window_fps = 0

        # Update display at 2Hz / 以2Hz更新显示
        if now - self.last_display_time >= 0.5:
            self.last_display_time = now
            if self.status_board:
                self.status_board.update_pose(window_fps, self.pose_count)
            else:
                print(f"\r[POSE] FPS: {window_fps:.1f}Hz | Packets: {self.pose_count}", end="", flush=True)

    def run(self):
        """
        Main thread loop for receiving pose data.
        接收姿态数据的主线程循环
        
        Automatically starts recording upon connection.
        连接后自动开始录制。
        """
        if not self.connect():
            return

        import socket

        print("[POSE] Receiving Quest data (Ctrl+C handled by main)...")
        # Always record pose as soon as connected (matches original standalone behavior)
        # 连接后立即开始录制姿态（与原始独立行为匹配）
        self.start_recording()
        try:
            while not self.stop_event.is_set():
                try:
                    # Receive data from TCP socket / 从TCP套接字接收数据
                    data = self.socket.recv(1024).decode("utf-8")
                    if not data:
                        print("\n[POSE] No more data, closing...")
                        break

                    self.buffer += data
                    # Clear buffer if too large to prevent memory issues / 如果缓冲区过大，清空以防止内存问题
                    if len(self.buffer) > self.buffer_max_size:
                        print(f"\n[POSE][WARN] Buffer too large ({len(self.buffer)} bytes), clearing...")
                        self.buffer = ""
                        continue

                    # Process complete lines (JSON packets end with \n) / 处理完整行（JSON包以\n结束）
                    while "\n" in self.buffer:
                        line, self.buffer = self.buffer.split("\n", 1)
                        if not line.strip():
                            continue

                        receive_ts = time.time()
                        pose_data = self.parse_pose_data(line)
                        if pose_data:
                            self._update_fps_display()
                            with self.lock:
                                if self.recording and self.saver:
                                    self.saver.add(pose_data, receive_ts)

                except socket.timeout:
                    # Normal timeout, continue loop / 正常超时，继续循环
                    continue
                except UnicodeDecodeError:
                    # Skip invalid UTF-8 data / 跳过无效的UTF-8数据
                    continue
                except (ConnectionResetError, BrokenPipeError):
                    print("\n[POSE] Connection lost")
                    break
        finally:
            print("\n[POSE] Closing connection...")
            self.disconnect()
            with self.lock:
                if self.saver:
                    self.saver.finalize()
                    print(
                        f"[POSE] Saved {self.saver.total_saved_frames} frames "
                        f"in {self.saver.file_counter - 1} file(s)"
                    )
                self.saver = None
                self.recording = False


class CameraConfig:
    """
    Configuration for a single camera device.
    单个相机设备的配置
    
    Stores all V4L2 camera parameters including resolution, exposure, white balance, etc.
    存储所有V4L2相机参数，包括分辨率、曝光、白平衡等。
    """
    
    def __init__(
        self,
        name: str,
        path: str,
        format: str,
        width: int,
        height: int,
        auto_exposure: int,
        brightness: int,
        gain: int,
        gamma: int,
        exposure: Optional[int] = None,
        auto_white_balance: int = 1,
        wb_temperature: Optional[int] = None,
    ):
        self.name = name  # Camera identifier (e.g., "left_hand", "right_hand") / 相机标识符
        self.path = path  # V4L2 device path (e.g., "/dev/video0") / V4L2设备路径
        self.format = format  # Image format (e.g., "MJPG") / 图像格式
        self.width = width  # Image width in pixels / 图像宽度（像素）
        self.height = height  # Image height in pixels / 图像高度（像素）
        self.auto_exposure = auto_exposure  # Auto exposure mode (1=on, 3=off) / 自动曝光模式
        self.brightness = brightness  # Brightness value / 亮度值
        self.gain = gain  # Gain value / 增益值
        self.gamma = gamma  # Gamma value / 伽马值
        self.exposure = exposure  # Manual exposure time / 手动曝光时间
        self.auto_white_balance = auto_white_balance  # Auto white balance (1=on, 0=off) / 自动白平衡
        self.wb_temperature = wb_temperature  # Manual white balance temperature / 手动白平衡色温


class RecordingSession:
    """
    Manages a single recording session for camera images.
    管理单个相机图像录制会话
    
    Creates directories, saves frames, and tracks timestamps per camera.
    创建目录、保存帧，并跟踪每个相机的时间戳。
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.recording = False  # Recording state / 录制状态
        self.start_time: Optional[float] = None  # Session start timestamp / 会话开始时间戳
        self.frame_counts: Dict[str, int] = {}  # Frame counter per camera / 每个相机的帧计数器
        self.timestamps: Dict[str, list] = {}  # Timestamp list per camera / 每个相机的时间戳列表
        self.lock = threading.Lock()  # Thread-safe access / 线程安全访问

    def start(self, demo_name: str, cameras: Dict[str, CameraConfig]):
        """
        Start a new recording session.
        开始新的录制会话
        
        Creates directory structure: demo_dir/{camera_name}_img/
        创建目录结构：demo_dir/{camera_name}_img/
        
        Args:
            demo_name: Name of the demo / 演示名称
            cameras: Dict of camera configs / 相机配置字典
            
        Returns:
            str: Path to demo directory / 演示目录路径
        """
        with self.lock:
            self.recording = True
            self.start_time = time.time()
            demo_dir = os.path.join(self.output_dir, demo_name)
            os.makedirs(demo_dir, exist_ok=True)

            # Create image directories for each camera / 为每个相机创建图像目录
            for cam_name in cameras:
                img_dir = os.path.join(demo_dir, f"{cam_name}_img")
                os.makedirs(img_dir, exist_ok=True)
                self.frame_counts[cam_name] = 0
                self.timestamps[cam_name] = []

            print(f"[START] {demo_name}")
            return demo_dir

    def stop(self, demo_dir: str) -> Dict:
        """
        Stop recording and save timestamp CSVs.
        停止录制并保存时间戳CSV文件
        
        Args:
            demo_dir: Demo directory path / 演示目录路径
            
        Returns:
            Dict: Statistics per camera {camera_name: {frames, fps}} / 每个相机的统计信息
        """
        with self.lock:
            if not self.recording:
                return {}

            self.recording = False
            duration = time.time() - self.start_time
            stats = {}

            # Save timestamps to CSV and calculate FPS / 将时间戳保存到CSV并计算FPS
            for cam_name, count in self.frame_counts.items():
                df = pd.DataFrame(self.timestamps[cam_name])
                csv_path = os.path.join(demo_dir, f"{cam_name}_timestamps.csv")
                df.to_csv(csv_path, index=False)

                fps = count / duration if duration > 0 else 0
                stats[cam_name] = {"frames": count, "fps": round(fps, 2)}
                print(f"[STOP] {cam_name}: {count} frames, {fps:.1f} fps")

            self.frame_counts.clear()
            self.timestamps.clear()
            return stats

    def save_frame(self, demo_dir: str, cam_name: str, frame: np.ndarray, ram_time: str):
        """
        Save a single frame to disk.
        将单帧保存到磁盘
        
        Filename format: {camera_name}_{frame_id}.jpg
        文件名格式：{camera_name}_{frame_id}.jpg
        
        Args:
            demo_dir: Demo directory / 演示目录
            cam_name: Camera name / 相机名称
            frame: Image array (BGR format) / 图像数组（BGR格式）
            ram_time: Timestamp from camera driver / 来自相机驱动的时间戳
        """
        with self.lock:
            if not self.recording:
                return

            frame_id = self.frame_counts[cam_name]
            filename = f"{cam_name}_{frame_id}.jpg"
            filepath = os.path.join(demo_dir, f"{cam_name}_img", filename)
            cv2.imwrite(filepath, frame)  # Save as JPEG / 保存为JPEG

            # Record timestamp metadata / 记录时间戳元数据

            self.timestamps[cam_name].append(
                {"frame_id": frame_id, "ram_time": ram_time, "filename": filename}
            )
            self.frame_counts[cam_name] += 1


class CameraWorker(threading.Thread):
    """
    Background thread worker for camera capture.
    相机采集的后台线程工作器
    
    Continuously captures frames from V4L2 camera and puts them in a queue.
    持续从V4L2相机捕获帧并将其放入队列。
    Uses non-blocking queue to drop frames if consumer is too slow.
    如果消费者太慢，使用非阻塞队列丢弃帧。
    """
    
    def __init__(self, config: CameraConfig, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.queue: Queue[Tuple[np.ndarray, str]] = Queue(maxsize=64)  # Frame buffer queue / 帧缓冲队列
        self.camera = None

    def run(self):
        """
        Main capture loop running in background thread.
        在后台线程中运行的主捕获循环
        """
        try:
            # Initialize V4L2 camera with config / 使用配置初始化V4L2相机
            self.camera = V4L2Camera(
                device_path=self.config.path,
                format=self.config.format,
                width=self.config.width,
                height=self.config.height,
            )
            # Apply camera settings / 应用相机设置
            self.camera.set_white_balance(
                auto=(self.config.auto_white_balance == 1), temperature=self.config.wb_temperature
            )
            self.camera.set_exposure(auto=(self.config.auto_exposure == 1), exposure_time=self.config.exposure)
            self.camera.set_brightness(brightness=self.config.brightness)
            self.camera.set_gain(self.config.gain)
            self.camera.set_gamma(gamma=self.config.gamma)

            print(f"[CAM] {self.config.name} ready: {self.config.width}x{self.config.height}")

            # Capture loop / 捕获循环
            while not self.stop_event.is_set():
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    _, ram_time = ret
                    try:
                        # Non-blocking put, drop frame if queue is full / 非阻塞放入，队列满时丢弃帧
                        self.queue.put((frame, ram_time), block=False)
                    except Exception:
                        # Queue full, drop frame / 队列满，丢弃帧
                        pass
        except Exception as e:
            print(f"[ERROR] {self.config.name}: {e}")
        finally:
            # Clean up camera resources / 清理相机资源
            if self.camera:
                self.camera.release()

    def get_frame(self) -> Optional[Tuple[np.ndarray, str]]:
        """
        Get the most recent frame from queue (non-blocking).
        从队列获取最新帧（非阻塞）
        
        Returns:
            Tuple of (frame_array, timestamp) or None if queue empty
            (帧数组, 时间戳)的元组，或队列为空时返回None
        """
        try:
            return self.queue.get(block=False)
        except Exception:
            return None


class DataRecorder:
    """
    Image recorder coordinating multiple camera workers.
    协调多个相机工作器的图像录制器
    
    Manages camera workers, recording sessions, and FPS monitoring.
    管理相机工作器、录制会话和FPS监控。
    Supports both single-hand and bimanual tasks.
    支持单手和双手任务。
    """

    def __init__(self, config: OmegaConf, status_board: Optional[StatusBoard] = None):
        self.config = config
        self.cameras = self._load_cameras()  # Camera configurations / 相机配置
        self.workers = {}  # Camera worker threads / 相机工作线程
        self.stop_event = threading.Event()  # Stop signal for workers / 工作器的停止信号
        # Recording session manager / 录制会话管理器
        self.session = RecordingSession(
            output_dir=os.path.join(config.recorder.output, config.task.name, "demos")
        )
        self.current_demo_dir = None  # Current demo directory / 当前演示目录
        self.demo_count = 0  # Total number of demos recorded / 录制的演示总数
        self.status_board = status_board
        self.last_totals: Dict[str, int] = {name: 0 for name in self.cameras}  # Frame totals / 帧总数
        self._init_fps_monitoring()

    def _load_cameras(self) -> Dict[str, CameraConfig]:
        """
        Load camera configurations from config file.
        从配置文件加载相机配置
        
        Supports single-hand (one camera) or bimanual (two cameras) tasks.
        支持单手（一个相机）或双手（两个相机）任务。
        
        Returns:
            Dict mapping camera name to CameraConfig / 相机名称到CameraConfig的映射字典
        """
        cam_cfg = self.config.recorder.camera
        paths_cfg = self.config.recorder.camera_paths
        task_type = self.config.task.type
        single_side = getattr(self.config.task, "single_hand_side", "left")

        # Common camera settings / 通用相机设置
        settings = {
            "format": getattr(cam_cfg, "format", "MJPG"),
            "width": cam_cfg.width,
            "height": cam_cfg.height,
            "auto_exposure": getattr(cam_cfg, "auto_exposure", 1),
            "exposure": getattr(cam_cfg, "exposure", None),
            "auto_white_balance": getattr(cam_cfg, "auto_white_balance", 1),
            "wb_temperature": getattr(cam_cfg, "wb_temperature", None),
            "brightness": getattr(cam_cfg, "brightness", 0),
            "gain": getattr(cam_cfg, "gain", 100),
            "gamma": getattr(cam_cfg, "gamma", 100),
        }

        # Select camera paths based on task type / 根据任务类型选择相机路径
        if task_type == "single":
            paths = {f"{single_side}_hand": paths_cfg[f"{single_side}_hand"]}
        else:  # bimanual
            paths = {"left_hand": paths_cfg.left_hand, "right_hand": paths_cfg.right_hand}

        cameras = {}
        for name, path in paths.items():
            cameras[name] = CameraConfig(name=name, path=path, **settings)
            print(f"[CONFIG] {name}: {path}")

        return cameras

    def _init_fps_monitoring(self):
        """
        Initialize FPS monitoring structures.
        初始化FPS监控结构
        """
        self.fps_counters = {name: 0 for name in self.cameras}  # Frame counters / 帧计数器
        self.fps_timers = {name: time.time() for name in self.cameras}  # Last update times / 上次更新时间
        self.last_fps = {name: 0.0 for name in self.cameras}  # Store last calculated FPS / 存储最后计算的FPS

    def start_workers(self):
        """
        Start camera worker threads for all cameras.
        启动所有相机的工作线程
        """
        for name, config in self.cameras.items():
            worker = CameraWorker(config, self.stop_event)
            self.workers[name] = worker
            worker.start()

    def start_recording(self):
        """
        Start a new recording demo.
        开始新的录制演示
        
        Creates a timestamped demo directory.
        创建带时间戳的演示目录。
        """
        if self.session.recording:
            return

        self.demo_count += 1
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")
        task_type = self.config.task.type
        demo_name = f"demo_{task_type}_{timestamp}"

        self.current_demo_dir = self.session.start(demo_name, self.cameras)
        print(f"[REC] Demo #{self.demo_count} started")
        if self.status_board:
            self.status_board.set_demo_status(f"Recording demo #{self.demo_count}")

    def stop_recording(self):
        """
        Stop the current recording demo.
        停止当前录制演示
        
        Returns:
            Dict: Recording statistics per camera / 每个相机的录制统计信息
        """
        if not self.session.recording:
            return

        stats = self.session.stop(self.current_demo_dir)
        print(f"[REC] Demo #{self.demo_count} completed\n")
        if self.status_board:
            # Persist totals for display after stop / 停止后保留总数用于显示
            for cam_name, cam_stats in stats.items():
                self.last_totals[cam_name] = cam_stats["frames"]
            totals_str = ", ".join(
                f"{k}:{v['frames']}" for k, v in stats.items()
            ) if stats else "no frames"
            self.status_board.set_demo_status(
                f"Stopped demo #{self.demo_count} ({totals_str})"
            )
        self.current_demo_dir = None
        return stats

    def process_frames(self):
        """
        Process frames from all camera workers.
        处理所有相机工作器的帧
        
        Polls each camera worker, saves frames if recording, and updates FPS stats.
        轮询每个相机工作器，如果正在录制则保存帧，并更新FPS统计。
        This runs in the main loop continuously.
        这在主循环中连续运行。
        """
        frames_processed = 0
        temp_frame = {}
        # Poll each camera worker for available frames / 轮询每个相机工作器获取可用帧
        for name, worker in self.workers.items():
            frame_data = worker.get_frame()
            if frame_data:
                frame, ram_time = frame_data
        #         temp_frame[name] = {}
        #         temp_frame[name]["frame"] = frame
        #         temp_frame[name]["ram_time"] = ram_time
        #         time.sleep(0.00001)

        # all_read = True
        # for name, worker in self.workers.items():
        #     if name not in temp_frame.keys():
        #         all_read = False
        #         return

        # for name, worker in self.workers.items():
                # Save frame if recording / 如果正在录制则保存帧
                if self.session.recording and self.current_demo_dir:
                    # pass
                    # self.session.save_frame(self.current_demo_dir, name, temp_frame[name]["frame"], temp_frame[name]["ram_time"])
                    self.session.save_frame(self.current_demo_dir, name, frame, ram_time)

                # Update FPS statistics / 更新FPS统计
                self.fps_counters[name] += 1
                now = time.time()
                elapsed = now - self.fps_timers[name]
                # Calculate FPS every second / 每秒计算一次FPS
                if elapsed >= 1.0:
                    fps = self.fps_counters[name] / elapsed
                    self.last_fps[name] = fps  # Store the calculated FPS / 存储计算的FPS
                    if self.session.recording:
                        total = self.session.frame_counts.get(name, 0)
                    else:
                        total = self.last_totals.get(name, 0)
                    if self.status_board:
                        self.status_board.update_camera(name, fps, total)
                    else:
                        if self.session.recording:
                            print(f"[{name}] FPS: {fps:.1f}, Total: {total}")
                    self.fps_counters[name] = 0
                    self.fps_timers[name] = now

                frames_processed += 1

        # Sleep briefly if no frames were processed to avoid busy loop / 如果没有处理帧则短暂休眠以避免忙循环
        if frames_processed == 0:
            time.sleep(0.001)

    def cleanup(self):
        """
        Clean up resources on exit.
        退出时清理资源
        
        Stops recording, signals worker threads to stop, and waits for them to finish.
        停止录制，向工作线程发送停止信号，并等待它们完成。
        """
        # Stop recording if still active / 如果仍在录制则停止
        if self.session.recording:
            self.stop_recording()

        # Signal all workers to stop / 向所有工作器发送停止信号
        self.stop_event.set()
        for worker in self.workers.values():
            worker.join(timeout=1)

        print(f"[EXIT] Recorded {self.demo_count} demos")


class CombinedRunner:
    """
    Main orchestrator combining image and pose capture with unified control.
    结合图像和姿态采集并统一控制的主协调器
    
    Manages:
    - Image recording from cameras
    - Pose data reception from Quest
    - TCP control client for Quest synchronization
    - Terminal input handling for start/stop
    - Status board display
    
    管理：
    - 从相机录制图像
    - 从Quest接收姿态数据
    - Quest同步的TCP控制客户端
    - 启动/停止的终端输入处理
    - 状态板显示
    """

    def __init__(self, config: OmegaConf, enable_pose: bool = True):
        self.config = config
        # Status display / 状态显示
        self.status_board = StatusBoard()
        self.status_board.set_demo_status("Idle")
        # Image recording manager / 图像录制管理器
        self.image_recorder = DataRecorder(config, status_board=self.status_board)
        self.enable_pose = enable_pose
        output_root = Path(config.recorder.output) / config.task.name
        # Pose data receiver (optional) / 姿态数据接收器（可选）
        self.pose_receiver = (
            QuestPoseReceiver(
                host="localhost",
                port=7777,
                base_output_dir=output_root / "all_trajectory",
                frames_per_file=5000,
                status_board=self.status_board,
            )
            if enable_pose
            else None
        )
        # TCP control client for Quest / Quest的TCP控制客户端
        ctrl_host = getattr(config.recorder, "control_host", "localhost")
        ctrl_port = getattr(config.recorder, "control_port", 50010)
        self.control_client = ControlClient(ctrl_host, ctrl_port)
        self.should_exit = False  # Exit flag / 退出标志
        self.tty_enabled = sys.stdin.isatty()  # Check if terminal supports keyboard input / 检查终端是否支持键盘输入
        self.old_settings = None  # Original terminal settings / 原始终端设置
        self._setup_terminal()

    def _setup_terminal(self):
        """
        Set terminal to cbreak mode for single-key input.
        将终端设置为cbreak模式以实现单键输入
        
        Allows reading single characters without Enter key.
        允许不按回车键读取单个字符。
        """
        if self.tty_enabled:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())  # Single-key mode / 单键模式
            except Exception:
                self.tty_enabled = False

    def _restore_terminal(self):
        """
        Restore original terminal settings on exit.
        退出时恢复原始终端设置
        """
        if self.old_settings and self.tty_enabled:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def _handle_input(self):
        """
        Handle keyboard input (non-blocking).
        处理键盘输入（非阻塞）
        
        [S] - Start/Stop recording / 开始/停止录制
        [Q] - Quit / 退出
        
        Returns:
            bool: True to continue, False to exit / 继续返回True，退出返回False
        """
        if not self.tty_enabled:
            return True

        try:
            # Non-blocking check for input / 非阻塞检查输入
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                ch = sys.stdin.read(1)
                if ch in ("q", "Q"):
                    self.should_exit = True
                    return False
                elif ch in ("s", "S"):
                    # Toggle recording / 切换录制状态
                    if self.image_recorder.session.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
        except Exception:
            pass
        return True

    def start_recording(self):
        """
        Start recording both image and pose data.
        开始录制图像和姿态数据
        
        Sends start signal to Quest via TCP control client.
        通过TCP控制客户端向Quest发送开始信号。
        """
        self.image_recorder.start_recording()
        if self.control_client:
            self.control_client.send_demo_control(self.image_recorder.demo_count, "start")

    def stop_recording(self):
        """
        Stop recording both image and pose data.
        停止录制图像和姿态数据
        
        Sends stop signal to Quest via TCP control client.
        通过TCP控制客户端向Quest发送停止信号。
        
        Returns:
            Dict: Recording statistics / 录制统计信息
        """
        stats = self.image_recorder.stop_recording()
        if self.control_client:
            self.control_client.send_demo_control(self.image_recorder.demo_count, "stop")
        return stats
    
    def _send_fps_updates(self):
        """
        Collect and send FPS updates to Quest.
        收集并发送FPS更新到Quest
        
        Sends real-time FPS information for monitoring on Quest device.
        发送实时FPS信息以在Quest设备上监控。
        """
        if not self.control_client:
            return
        
        # Collect camera FPS (use last calculated FPS values) / 收集相机FPS（使用最后计算的FPS值）
        camera_fps = {}
        for name in self.image_recorder.cameras:
            fps = self.image_recorder.last_fps.get(name, 0.0)
            camera_fps[name] = round(fps, 1)
        
        # Collect pose FPS / 收集姿态FPS
        pose_fps = 0.0
        if self.enable_pose and self.pose_receiver:
            if len(self.pose_receiver.window) >= 2:
                dur = self.pose_receiver.window[-1] - self.pose_receiver.window[0]
                pose_fps = round((len(self.pose_receiver.window) - 1) / dur if dur > 0 else 0, 1)
        
        self.control_client.send_fps_update(camera_fps, pose_fps)

    def run(self):
        """
        Main run loop for combined recording system.
        组合录制系统的主运行循环
        
        Starts camera workers and pose receiver, then enters main loop
        processing frames and handling keyboard input.
        启动相机工作器和姿态接收器，然后进入主循环处理帧和处理键盘输入。
        """
        print("\n" + "=" * 60)
        print("  ViTaMIn-B Image + Pose Recorder")
        print("  ViTaMIn-B 图像 + 姿态录制器")
        print("=" * 60)
        print("  [S] - Start/Stop recording both image & pose")
        print("  [S] - 开始/停止录制图像和姿态")
        print("  [Q] - Quit")
        print("  [Q] - 退出")
        print("=" * 60 + "\n")

        # Start all workers / 启动所有工作器
        self.image_recorder.start_workers()
        if self.enable_pose and self.pose_receiver:
            self.pose_receiver.start()

        try:
            # Main processing loop / 主处理循环
            while not self.should_exit:
                self.image_recorder.process_frames()
                self._send_fps_updates()  # Send FPS updates to Quest / 向Quest发送FPS更新
                if not self._handle_input():
                    break
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user / 用户中断")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up all resources before exit.
        退出前清理所有资源
        
        Stops image recorder, pose receiver, control client, and restores terminal.
        停止图像录制器、姿态接收器、控制客户端，并恢复终端。
        """
        # Clean up image recorder / 清理图像录制器
        self.image_recorder.cleanup()
        # Stop pose receiver / 停止姿态接收器
        if self.enable_pose and self.pose_receiver:
            self.pose_receiver.stop_event.set()
            self.pose_receiver.join(timeout=2)
        # Close control client / 关闭控制客户端
        if self.control_client:
            self.control_client.close()
        # Restore terminal settings / 恢复终端设置
        self._restore_terminal()
        print("[EXIT] Shutdown complete / 关闭完成")


def main():
    """
    Main entry point for the data acquisition script.
    数据采集脚本的主入口点
    
    Parses command-line arguments, loads configuration, and runs the combined recorder.
    解析命令行参数，加载配置，并运行组合录制器。
    """
    parser = argparse.ArgumentParser(
        description="Combined image + pose recorder / 组合图像和姿态录制器"
    )
    parser.add_argument("--cfg", type=str, required=True, help="Config file path / 配置文件路径")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose capture / 禁用姿态采集")
    parser.add_argument("--control-host", type=str, default="localhost", 
                       help="Quest control host (default: localhost) / Quest控制主机")
    parser.add_argument("--control-port", type=int, default=50010, 
                       help="Quest control port (default: 50010) / Quest控制端口")
    args = parser.parse_args()

    try:
        config = OmegaConf.load(args.cfg)

        # allow CLI override for control channel
        config.recorder.control_host = args.control_host
        config.recorder.control_port = args.control_port

        task_dir = os.path.join(config.recorder.output, config.task.name)
        os.makedirs(os.path.join(task_dir, "demos"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "all_trajectory"), exist_ok=True)

        runner = CombinedRunner(config, enable_pose=not args.no_pose)
        runner.run()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()