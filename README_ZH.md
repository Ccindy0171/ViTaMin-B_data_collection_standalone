# ViTaMIn-B 数据采集与处理pipeline（独立版）

本目录提供 ViTaMIn-B 操作数据采集与处理pipeline的独立、自包含版本，可直接使用，不依赖当前目录之外的任何代码或资源。

pipeline支持基于 Quest 控制器的演示数据采集与离线处理，适用于双手与单手操作任务。

支持的任务模式：

* 双手操作（Bimanual）：双夹爪 + Quest 双控制器
* 单手操作（Single / Mono-manual）：单夹爪 + Quest 单控制器

## 环境要求

* Python 3.10 或 3.11（推荐）
* 虚拟环境（强烈建议）

## 快速上手

### Step 1：环境配置

1）进入pipeline目录：

```bash
cd /path/to/data_collection
```

2）创建并激活虚拟环境（推荐）：

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

3）安装依赖：

```bash
pip install -r requirements.txt
```

### Step 2：验证环境

运行自检脚本，确认依赖与环境是否就绪：

```bash
python test_standalone.py
```

正常情况下会看到：

```
✓ ALL TESTS PASSED - Pipeline is ready to use!
```

### Step 3：准备原始数据

请将原始演示数据整理为以下结构。

双手模式：

```
data_collection/data/<task_name>/demos/
├── demo_<name>_<timestamp_1>/
│   ├── left_hand_img/          # 左手相机图像（JPEG 序列）
│   ├── right_hand_img/         # 右手相机图像（JPEG 序列）
│   └── all_trajectory/
│       └── quest_poses_<timestamp>.json
├── demo_<name>_<timestamp_2>/
│   └── ...
└── ...
```

单手模式：

```
data_collection/data/<task_name>/demos/
├── demo_<name>_<timestamp_1>/
│   ├── left_hand_img/          # 或 right_hand_img/（由 single_hand_side 决定）
│   └── all_trajectory/
│       └── quest_poses_<timestamp>.json
├── demo_<name>_<timestamp_2>/
│   └── ...
└── ...
```

如果你已有现成数据，也可以用符号链接直接挂到本目录：

```bash
ln -s /path/to/your/existing/data ./data
```

### Step 4：配置任务参数

按任务类型修改 `config/VB_task_config.yaml`。

双手任务示例：

```yaml
task:
  name: "_0118_bi_pick_and_place"
  type: bimanual  # 双手模式

recorder:
  camera_paths:
    left_hand: "/dev/video0"
    right_hand: "/dev/video2"
```

单手任务示例：

```yaml
task:
  name: "_0118_mono_pick"
  type: single            # 单手模式
  single_hand_side: left  # 或 "right"

recorder:
  camera_paths:
    left_hand: "/dev/video0"   # single_hand_side: left 时使用
    right_hand: "/dev/video2"  # single_hand_side: right 时使用
```

更详细的参数说明见下方「配置」章节。

### Step 5：运行pipeline

```bash
python run_data_collection_pipeline.py
```

运行后会自动依次完成所有处理步骤，无需手动逐步执行。

## 输入数据结构

pipeline默认按以下方式组织原始演示数据。

### 使用 00_get_data.py 采集数据

pipeline包含数据记录脚本 `vitamin_b_data_collection_pipeline/00_get_data.py`，用于实时采集同步的相机图像与 Quest 控制器姿态。

采集流程：

1. 在 `config/VB_task_config.yaml` 中配置任务类型（`bimanual` 或 `single`）和相机设备
2. 运行：

   ```bash
   python vitamin_b_data_collection_pipeline/00_get_data.py --cfg config/VB_task_config.yaml
   ```
3. 脚本会自动完成：

   * 从配置的相机录制（单手 1 路，双手 2 路）
   * 捕获 Quest 控制器姿态
   * 以正确的文件夹结构落盘保存

输出结构（双手任务）：

```
data/
└── <task_name>/                          # 例如 _0118_bi_pick_and_place
    └── demos/
        ├── demo_<timestamp_1>/
        │   ├── left_hand_img/            # 左相机 JPEG 序列
        │   │   ├── frame_000000.jpg
        │   │   ├── frame_000001.jpg
        │   │   └── ...
        │   ├── right_hand_img/           # 右相机 JPEG 序列
        │   │   ├── frame_000000.jpg
        │   │   └── ...
        │   └── all_trajectory/
        │       └── quest_poses_<timestamp>.json  # VR 控制器姿态
        ├── demo_<timestamp_2>/
        │   └── ...
        └── ...
```

输出结构（单手任务）：

```
data/
└── <task_name>/
    └── demos/
        └── demo_<timestamp>/
            ├── left_hand_img/            # 或 right_hand_img/（由配置决定）
            │   ├── frame_000000.jpg
            │   ├── frame_000001.jpg
            │   └── ...
            └── all_trajectory/
                └── quest_poses_<timestamp>.json
```

图像格式：

* 格式：JPEG（`.jpg`）
* 分辨率：3840×800（原始，裁剪前）或按配置
* 命名：顺序 `frame_XXXXXX.jpg`
* 内容：原始鱼眼视角（在步骤 01 中拆分为视觉/触觉视图）

Quest 姿态数据格式（`quest_poses_*.json`）：

```json
{
  "timestamp_1": {
    "left_controller": [x, y, z, qx, qy, qz, qw],
    "right_controller": [x, y, z, qx, qy, qz, qw],
    "left_gripper_width": 0.05,
    "right_gripper_width": 0.05
  },
  "timestamp_2": { ... }
}
```

### pipeline如何区分单手/双手

pipeline会根据配置与 demo 的目录结构自动适配：

双手模式（`task.type: bimanual`）：

* 处理 `left_hand_img/` 与 `right_hand_img/`
* 检测两个夹爪上的 ArUco 标记
* 计算两个夹爪的开口宽度
* 生成包含 `robot0` 和 `robot1` 的 zarr

单手模式（`task.type: single`）：

* 只处理 `task.single_hand_side` 指定那一侧的图像目录
* 只检测一个夹爪的 ArUco 标记
* 只计算一个夹爪的开口宽度
* 只生成 `robot0` 相关数组（不会创建 `robot1_*`）

在步骤 07 与 08 中，脚本也会通过检查 demo 中实际存在的图像目录来自动识别模式。

## 使用指南

### 一键执行整条pipeline（推荐）

`run_data_collection_pipeline.py` 会按顺序执行整个工作流，不需要手动介入。

运行：

```bash
# 在 data_collection 目录下
python run_data_collection_pipeline.py
```

运行时会发生什么：

1）启动提示
脚本会先提醒你确认配置里的 `use_gelsight` 是否设置正确：

```
Please make sure the param use_gelsight is True or False in the config file
Press Enter to continue...
```

按 Enter 继续。

2）自动依次执行以下步骤

步骤 1：图像裁剪（`01_crop_img.py`）

```
==================================================
Running step: 01_crop_img.py
==================================================
```

* 将 3840×800 原始图像拆分为视觉（中心）与触觉（左/右）视图
* 双手处理两路，单手只处理一路
* 过程会显示进度（如 “10/10 tasks”）

步骤 2：ArUco 检测（`04_get_aruco_pos.py`）

```
==================================================
Running step: 04_get_aruco_pos.py
==================================================
```

* 检测夹爪上的 ArUco 标记用于姿态估计
* 输出 `tag_detection_*.pkl`
* 会报告检测成功率

步骤 3：夹爪宽度计算（`05_get_width.py`）

```
==================================================
Running step: 05_get_width.py
==================================================
```

* 根据标记对计算夹爪开口宽度
* 输出 `gripper_width_*.csv`
* 会报告检测率与插值情况

步骤 4：生成数据集计划（`07_generate_dataset_plan.py`）

```
==================================================
Running step: 07_generate_dataset_plan.py
==================================================
```

* 自动识别每个 demo 是单手还是双手
* 用时间戳同步图像、Quest 姿态与夹爪宽度
* 应用坐标变换（Unity 左手系 → 右手系）
* 生成 `dataset_plan.pkl`

步骤 5：生成 Replay Buffer（`08_generate_replay_buffer.py`）

```
==================================================
Running step: 08_generate_replay_buffer.py
==================================================
```

* 并行处理所有帧
* 按目标分辨率 resize 图像
* 根据识别到的夹爪数量自动调整 zarr 结构
* 用 JPEG-XL 压缩并输出最终数据集
* 生成 `<task_name>.zarr.zip`

3）完成提示：

```
Pipeline completed successfully!
```

### 预期输出

完成后，你会在 `data/<task_name>/` 下看到：

```
data/<task_name>/
├── dataset_plan.pkl
├── <task_name>.zarr.zip
└── demos/
    ├── demo_<timestamp_1>/
    │   ├── left_hand_visual_img/
    │   ├── left_hand_left_tactile_img/
    │   ├── left_hand_right_tactile_img/
    │   ├── right_hand_visual_img/        # 双手模式才有
    │   ├── tag_detection_left.pkl
    │   ├── tag_detection_right.pkl       # 双手模式才有
    │   ├── gripper_width_left.csv
    │   └── gripper_width_right.csv       # 双手模式才有
    └── ...
```

### 典型耗时（参考）

5 个双手 demo（约 1000 帧）：

* Step 1：3 秒
* Step 2：3 秒
* Step 3：2 秒
* Step 4：1 秒
* Step 5：8 秒
* 总计：约 20 秒

### 成功标志

以下输出通常意味着每一步都成功完成：

* ✓ `Completed step: 01_crop_img.py`
* ✓ `Completed step: 04_get_aruco_pos.py`
* ✓ `Completed step: 05_get_width.py`
* ✓ `Completed step: 07_generate_dataset_plan.py`
* ✓ `Completed step: 08_generate_replay_buffer.py`
* ✓ `Pipeline completed successfully!`

### 手动逐步执行（调试用）

如果你需要单独跑某一步定位问题，可以按下面方式执行：

```bash
cd vitamin_b_data_collection_pipeline

python 01_crop_img.py --cfg ../config/VB_task_config.yaml
python 04_get_aruco_pos.py --cfg ../config/VB_task_config.yaml
python 05_get_width.py --cfg ../config/VB_task_config.yaml
python 07_generate_dataset_plan.py --cfg ../config/VB_task_config.yaml
python 08_generate_replay_buffer.py --cfg ../config/VB_task_config.yaml
```

## 配置

所有参数集中在 `config/VB_task_config.yaml`。

### 基本设置

```yaml
task:
  name: "_0118_bi_pick_and_place"  # 任务名称
  type: bimanual                   # "bimanual" 或 "single"
  single_hand_side: left           # 仅在 type: single 时使用（"left" 或 "right"）
```

### 相机配置

```yaml
recorder:
  camera_paths:
    left_hand: "/dev/video0"
    right_hand: "/dev/video2"
  camera:
    format: "MJPG"
    width: 3840
    height: 800
    fps: 30
  output: "./data"
```

注意：单手模式下只会使用 `task.single_hand_side` 对应的相机。

### ArUco 检测与宽度计算

```yaml
calculate_width:
  cam_intrinsic_json_path: "../assets/intri_result/gopro_intrinsics_2_7k.json"
  aruco_dict:
    predefined: DICT_4X4_50
  marker_size_map:
    0: 0.02
    1: 0.02

  left_hand_aruco_id:
    left_id: 0
    right_id: 1

  right_hand_aruco_id:
    left_id: 2
    right_id: 3
```

注意：单手模式下只会读取活动手侧对应的 ArUco ID 配置。

### 输出设置

```yaml
output_train_data:
  min_episode_length: 10
  visual_out_res: [224, 224]
  tactile_out_res: [224, 224]

  use_tactile_img: True
  use_tactile_pc: False
  use_ee_pose: False

  compression_level: 99
  num_workers: 16

  use_inpaint_tag: True
  use_mask: False
```

### 坐标变换

```yaml
output_train_data:
  tx_quest_2_ee_left_path: '../assets/tf_cali_result/quest_2_ee_left_hand.npy'
  tx_quest_2_ee_right_path: '../assets/tf_cali_result/quest_2_ee_right_hand.npy'
```

注意：单手模式下只会使用活动手的变换矩阵。

重要：配置文件中的路径建议相对 `data_collection/` 目录填写，或直接使用绝对路径。

## pipeline输出

### 最终数据集

主要输出为训练可用的压缩 zarr 归档：

位置：`data/<task_name>/<task_name>.zarr.zip`
示例：`data/_0118_bi_pick_and_place/_0118_bi_pick_and_place.zarr.zip`
典型大小：约 5 个 demo、500 帧时为 50–100 MB（仅供参考）

### 数据集内容

双手任务会包含 `robot0` 与 `robot1`；单手任务只包含 `robot0`。

机器人数据（双手）：

* `robot0_eef_pos`, `robot1_eef_pos`：末端位置 [N, 3]（米）
* `robot0_eef_rot_axis_angle`, `robot1_eef_rot_axis_angle`：旋转 [N, 3]（轴角）
* `robot0_gripper_width`, `robot1_gripper_width`：夹爪开口 [N, 1]（米）
* `robot0_demo_start_pose`, `robot1_demo_start_pose`：初始姿态 [N, 6]
* `robot0_demo_end_pose`, `robot1_demo_end_pose`：目标姿态 [N, 6]

机器人数据（单手）：

* `robot0_eef_pos`
* `robot0_eef_rot_axis_angle`
* `robot0_gripper_width`
* `robot0_demo_start_pose`
* `robot0_demo_end_pose`

相机数据（两种模式一致）：

* `camera0_rgb`：主视觉相机 [N, H, W, 3]（uint8）
* `camera0_left_tactile`：左触觉相机 [N, H, W, 3]（启用时）
* `camera0_right_tactile`：右触觉相机 [N, H, W, 3]（启用时）

元数据：

* `episode_ends`：每段 episode 的结束帧索引 [E]

说明：

* N：所有 episode 的总帧数
* H, W：输出分辨率（如 224×224）
* E：episode 数量

### 中间输出

每个 demo 目录下会生成：

```
demos/demo_<timestamp>/
├── left_hand_visual_img/
├── left_hand_left_tactile_img/
├── left_hand_right_tactile_img/
├── right_hand_visual_img/              # 双手模式才有
├── right_hand_left_tactile_img/        # 双手模式才有
├── right_hand_right_tactile_img/       # 双手模式才有
├── tag_detection_left.pkl
├── tag_detection_right.pkl             # 双手模式才有
├── gripper_width_left.csv
└── gripper_width_right.csv             # 双手模式才有
```

任务根目录下会生成：

```
data/<task_name>/
├── dataset_plan.pkl
└── <task_name>.zarr.zip
```

## 数据加载示例

```python
import zarr

dataset_path = "data/_0118_bi_pick_and_place/_0118_bi_pick_and_place.zarr.zip"
store = zarr.ZipStore(dataset_path, mode='r')
root = zarr.group(store=store)

eef_pos = root['data']['robot0_eef_pos'][:]
visual = root['data']['camera0_rgb'][:]
tactile = root['data']['camera0_left_tactile'][:]
episode_ends = root['meta']['episode_ends'][:]

episode_starts = [0] + episode_ends[:-1].tolist()
episode_ends = episode_ends.tolist()

episode_idx = 0
start_idx = episode_starts[episode_idx]
end_idx = episode_ends[episode_idx]

episode_data = {
    'eef_pos': eef_pos[start_idx:end_idx],
    'visual': visual[start_idx:end_idx],
    'tactile': tactile[start_idx:end_idx],
}

print(f"片段 {episode_idx} 有 {end_idx - start_idx} 帧")
```

## pipeline步骤说明

Step 1：图像裁剪（01_crop_img.py）

* 将 3840×800 原始图像切分成 3 个 1280×800 视图
* 按手侧做必要的旋转校正
* 输出到独立的裁剪结果目录

Step 2：ArUco 检测（04_get_aruco_pos.py）

* 在视觉图像中检测 ArUco 标记
* 根据标记估计姿态
* 输出为 pickle 文件

Step 3：夹爪宽度计算（05_get_width.py）

* 根据标记对距离计算夹爪开口宽度
* 对缺失或无效帧做插值
* 输出逐帧 CSV

Step 4：生成数据集计划（07_generate_dataset_plan.py）

* 自动识别每个 demo 的单手/双手模式
* 用时间戳对齐图像、Quest 姿态与夹爪宽度
* 应用坐标系转换（Unity → 右手系）
* 输出 `dataset_plan.pkl`

Step 5：构建 Replay Buffer（08_generate_replay_buffer.py）

* 根据数据集计划构建 zarr
* 按夹爪数量自动决定 `robot0/robot1` 结构
* 多进程并行处理、resize、可选修复/遮罩
* JPEG-XL 压缩后输出最终 `zarr.zip`

## 故障排除

1）导入错误
错误：`ModuleNotFoundError: No module named 'cv2'` 等
处理：

* 确认虚拟环境已激活
* 重新安装依赖：`pip install -r requirements.txt`
* 确认 Python 版本为 3.10/3.11：`python --version`

2）配置文件找不到
错误：`FileNotFoundError: config/VB_task_config.yaml not found`
处理：

* 确认从 `data_collection/` 目录运行
* 检查文件是否存在：`ls config/VB_task_config.yaml`
* 或直接用绝对路径传参：`--cfg /abs/path/to/config/VB_task_config.yaml`

3）ArUco 检测率低
症状：检测率 50% 或更低，频繁出现 NOT FOUND
处理：

* 光照尽量均匀、充足
* 确认标记无遮挡、清晰可见
* 检查 `marker_size_map` 是否与实物一致
* 确认相机内参文件正确

4）夹爪宽度有效率低
症状：`valid (50.0%)` 之类
处理：

* 先解决 ArUco 可见性问题
* 检查标记是否损坏
* 检查配置里的 ID 是否与实物一致

5）内存占用过高
错误：`MemoryError` 或机器明显变慢
处理：

* 降低 `num_workers`（比如 16 → 8 → 4）
* 降低输出分辨率（比如 `[224,224]` → `[128,128]`）
* 减少一次处理的 demo 数量
* 关闭其他占资源的应用

6）路径相关错误
错误：`FileNotFoundError: [Errno 2] No such file or directory`
处理：

* 检查数据目录结构是否符合约定
* 检查 `task.name` 是否与数据文件夹名称一致
* 确保相对路径是基于 `data_collection/` 目录
* 必要时改成绝对路径

7）pipeline卡住/无输出
症状：几分钟没有新日志
处理：

* 看 CPU/磁盘是否仍在工作
* 降低 `num_workers` 以减少资源争用
* Ctrl+C 终止后用更少 worker 重试

8）Zarr 压缩失败
处理：

* 确认 `zarr==2.16.0` 与 `numcodecs==0.11.0` 已安装
* 检查磁盘空间（建议至少留出输出体积 2 倍）
* 适当降低 `compression_level`（如 99 → 90）

9）识别到的相机/夹爪数量不对
错误：`AssertionError: n_cameras != expected` 或缺夹爪数据
处理：

* 先检查 `task.type` 是否与数据一致（双手/单手）
* 双手必须同时存在 `left_hand_img/` 与 `right_hand_img/`
* 单手只应存在一侧目录，且与 `single_hand_side` 一致
* 如果你混合数据集，确保每个 demo 内部结构自洽

10）单手模式下 ArUco 检测不到
处理：

* 检查 `task.single_hand_side` 是否与采集手一致
* 检查对应手侧的 ArUco ID 是否配置正确：

  * `single_hand_side: left` → 使用 `left_hand_aruco_id`
  * `single_hand_side: right` → 使用 `right_hand_aruco_id`

## 单手/双手切换说明

1）改配置

双手：

```yaml
task:
  type: bimanual

recorder:
  camera_paths:
    left_hand: "/dev/video0"
    right_hand: "/dev/video2"
```

单手：

```yaml
task:
  type: single
  single_hand_side: left  # 或 right

recorder:
  camera_paths:
    left_hand: "/dev/video0"
    right_hand: "/dev/video2"
```

2）确保数据结构正确

* 双手 demo：必须同时有 `left_hand_img/` 与 `right_hand_img/`
* 单手 demo：只保留一侧目录（与 `single_hand_side` 对应）

3）重新采集（可选）

* 双手：两路相机同时录制
* 单手：只录制一路（由 `single_hand_side` 决定）

4）处理已有数据

* 01–03 只会处理实际存在数据的手
* 07–08 会通过目录结构自动识别模式
* 允许同一数据集中混合单手与双手 demo（按 demo 结构分别处理）

5）核对 ArUco ID

```yaml
calculate_width:
  left_hand_aruco_id:
    left_id: 0
    right_id: 1
  right_hand_aruco_id:
    left_id: 2
    right_id: 3
```

## 项目结构

```
data_collection/
├── README.md
├── README_ZH.md
├── requirements.txt
├── test_standalone.py
├── run_data_collection_pipeline.py
├── clean_processed_data.py
├── config/
│   └── VB_task_config.yaml
├── assets/
│   ├── intri_result/
│   ├── tf_cali_result/
│   └── cali_width_result/
├── utils/
│   ├── camera_device.py
│   ├── config_utils.py
│   ├── cv_util.py
│   ├── pose_util.py
│   ├── replay_buffer.py
│   └── imagecodecs_numcodecs.py
├── vitamin_b_data_collection_pipeline/
│   ├── 00_get_data.py
│   ├── 01_crop_img.py
│   ├── 04_get_aruco_pos.py
│   ├── 05_get_width.py
│   ├── 07_generate_dataset_plan.py
│   ├── 08_generate_replay_buffer.py
│   ├── README.md
│   └── utils/
└── data/
    └── <task_name>/
        ├── demos/
        ├── dataset_plan.pkl
        └── <task_name>.zarr.zip
```