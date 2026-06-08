# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
将RobotWin轨迹数据转换为Wan环境所需的npy格式。

## 数据源结构

RobotWin数据使用HDF5格式存储，目录结构如下:

```
/manifold-obs/wzl/vla_robotwin_4k_320/
└── fulldata_40000/
    └── click_bell/
        └── demo_clean/
            ├── scene_info.json          # 场景信息 (每个episode的metadata)
            ├── instructions/            # 任务指令
            │   ├── episode0.json
            │   ├── episode1.json
            │   └── ...
            ├── data/                    # HDF5轨迹数据
            │   ├── episode0.hdf5
            │   ├── episode1.hdf5
            │   └── ...
            └── video/                   # 视频文件 (可选)
```

### scene_info.json 结构

```json
{
    "episode_0": {
        "episode_id": 0,
        "task_name": "click_bell",
        "task_config": "demo_clean",
        "policy_name": "pi0_wm",
        "seed": 60100000,
        "success": true,
        "status": "success",
        "take_action_cnt": 112,
        "instruction": "Direct the right arm to click the palm-sized bell's top.",
        "info": {
            "{A}": "050_bell/base1",
            "{a}": "right"
        }
    }
}
```

### HDF5文件结构 (episode0.hdf5)

```
episode0.hdf5
├── joint_action/
│   └── vector: (T, 14) float64        # 14维action (左臂6+1, 右臂6+1)
└── observation/
    └── head_camera/
        └── rgb: (T,) bytes            # JPEG编码的图像数据
```

**关键信息**:
- **Action维度**: 14维 (左臂6关节+1夹爪, 右臂6关节+1夹爪)
- **图像格式**: JPEG编码的字节流，需要解码
- **原始图像尺寸**: 320x240 (width x height)
- **目标图像尺寸**: 256x256 (需要resize)
- **图像值域**: [0, 255] uint8 (需要归一化到[0, 1])

## 转换目标

将每条HDF5轨迹转换为两个npy文件:

1. **traj{N}.npy**: 标准版本 (仅含初始帧)
2. **traj{N}_kir.npy**: KIR版本 (含前4个关键帧)

### NPy文件结构

```python
{
    'start_items': [
        {
            'image': np.ndarray,              # [3, 256, 256], float32, [0, 1]
            'observation.state': np.ndarray   # [14], float32, 初始关节位置
        }
    ],
    'target_items': [
        {
            'image': np.ndarray,              # [3, 256, 256], float32, [0, 1]
            'action': np.ndarray              # [14], float64, 目标action
        }
        # ... 共4个关键帧
    ],
    'task': str                               # 任务描述
}
```

## 使用方法

### 基本用法

```bash
cd /your/path//RLinf

python rlinf/data/datasets/world_model/convert_robotwin_to_npy.py \\
    --src_dir /manifold-obs/wzl/vla_robotwin_4k_320/fulldata_40000/click_bell/demo_clean \\
    --dst_dir /your/path//RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/dataset \\
    --max_trajs 100 \\
    --enable_kir
```

### 参数说明

- `--src_dir`: RobotWin数据目录 (包含scene_info.json, data/, instructions/)
- `--dst_dir`: 输出npy文件路径
- `--max_trajs`: 最大转换轨迹数 (None表示全部)
- `--enable_kir`: 启用KIR模式 (生成_kir.npy文件)
- `--image_size`: 图像目标尺寸 (默认256x256)
- `--task`: 任务描述 (默认从scene_info.json读取，或手动指定)
- `--filter_success`: 仅转换成功的轨迹 (默认True)

### 高级用法

```bash
# 仅转换成功的轨迹
python rlinf/data/datasets/world_model/convert_robotwin_to_npy.py \\
    --src_dir /manifold-obs/wzl/vla_robotwin_4k_320/fulldata_40000/click_bell/demo_clean \\
    --dst_dir /path/to/output/dataset \\
    --filter_success

# 转换所有轨迹 (包括失败的)
python rlinf/data/datasets/world_model/convert_robotwin_to_npy.py \\
    --src_dir /manifold-obs/wzl/vla_robotwin_4k_320/fulldata_40000/click_bell/demo_clean \\
    --dst_dir /path/to/output/dataset \\
    --no-filter_success

# 自定义任务描述
python rlinf/data/datasets/world_model/convert_robotwin_to_npy.py \\
    --src_dir /manifold-obs/wzl/vla_robotwin_4k_320/fulldata_40000/click_bell/demo_clean \\
    --dst_dir /path/to/output/dataset \\
    --task "click the bell"
```
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


def resize_image(img_tensor, target_size=(256, 256)):
    """
    调整图像大小到目标尺寸。
    
    Args:
        img_tensor: [3, H, W] 或 [H, W, 3] 的tensor或numpy数组
        target_size: (H, W) 目标尺寸
        
    Returns:
        [3, target_H, target_W] 的tensor
    """
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor)
    
    # 确保是 [3, H, W] 格式
    if img_tensor.shape[0] != 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    
    # 确保是float32并在[0, 1]范围
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.float() / 255.0
    
    # 调整大小
    img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
    img_tensor = torch.nn.functional.interpolate(
        img_tensor, size=target_size, mode='bilinear', align_corners=False
    )
    img_tensor = img_tensor.squeeze(0)  # [3, H, W]
    
    return img_tensor


def convert_single_trajectory(
    trajectory_data,
    traj_idx,
    enable_kir=True,
    image_size=(256, 256),
    task=None
):
    """
    转换单条轨迹为npy格式。
    
    Args:
        trajectory_data: 轨迹数据dict (包含'steps', 'instruction'等)
        traj_idx: 轨迹索引
        enable_kir: 是否生成KIR版本
        image_size: 图像目标尺寸
        task: 任务描述 (如果为None，使用trajectory_data['instruction'])
        
    Returns:
        保存的文件路径列表
    """
    saved_files = []
    steps = trajectory_data['steps']
    
    if len(steps) == 0:
        print(f"  ⚠️  轨迹 {traj_idx} 没有数据步，跳过")
        return saved_files
    
    # 提取初始帧
    first_step = steps[0]
    head_camera = first_step['head_camera']  # [H, W, 3], uint8
    
    # 调整图像大小并归一化
    img_tensor = resize_image(head_camera, image_size)
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0
    
    # 提取初始状态 (使用action作为qpos的近似)
    qpos = first_step.get('qpos', first_step['action'])
    
    # 构建start_items
    start_items = [
        {
            'image': img_tensor.numpy(),  # [3, 256, 256]
            'observation.state': qpos if isinstance(qpos, np.ndarray) else np.array(qpos)
        }
    ]
    
    # 构建target_items (前4个关键帧，用于KIR)
    target_items = []
    # 取轨迹中的第1, 2, 3, 4步作为关键帧 (索引1-4)
    for i in range(1, min(5, len(steps))):
        step_data = steps[i]
        head_cam = step_data['head_camera']
        action = step_data['action']
        
        # 调整图像并归一化
        img_t = resize_image(head_cam, image_size)
        if img_t.max() > 1.0:
            img_t = img_t / 255.0
        
        target_items.append({
            'image': img_t.numpy(),
            'action': action if isinstance(action, np.ndarray) else np.array(action)
        })
    
    # 确保有4个关键帧 (如果轨迹不够长，重复最后一个)
    while len(target_items) < 4:
        target_items.append({
            'image': target_items[-1]['image'].copy(),
            'action': target_items[-1]['action'].copy()
        })
    
    # 获取任务描述
    if task is None:
        task = trajectory_data.get('instruction', 'click the bell')
    
    # 构建数据dict
    data = {
        'start_items': start_items,
        'target_items': target_items,
        'task': task
    }
    
    # 保存标准版本
    npy_path = f"traj{traj_idx}.npy"
    np.save(npy_path, data)
    saved_files.append(npy_path)
    
    # 保存KIR版本 (如果启用)
    if enable_kir:
        kir_path = f"traj{traj_idx}_kir.npy"
        np.save(kir_path, data)
        saved_files.append(kir_path)
    
    return saved_files


def load_robotwin_data(src_dir, filter_success=True):
    """
    加载RobotWin HDF5格式数据。
    
    数据结构:
    src_dir/
    ├── scene_info.json          # episode metadata
    ├── instructions/            # task instructions
    │   ├── episode0.json
    │   └── ...
    └── data/                    # HDF5 trajectory files
        ├── episode0.hdf5
        └── ...
    
    Args:
        src_dir: 数据源目录 (包含scene_info.json和data/)
        filter_success: 是否仅加载成功的轨迹
        
    Returns:
        trajectories: 列表，每个元素是dict:
            {
                'episode_id': int,
                'instruction': str,
                'success': bool,
                'steps': [  # 每一步的数据
                    {
                        'head_camera': np.ndarray,  # [H, W, 3], uint8
                        'action': np.ndarray,       # [14], float64
                        'qpos': np.ndarray          # [14], float64 (使用action作为qpos)
                    },
                    ...
                ]
            }
    """
    src_path = Path(src_dir)
    
    # 加载scene_info.json
    scene_info_path = src_path / "scene_info.json"
    if not scene_info_path.exists():
        raise FileNotFoundError(f"未找到scene_info.json: {scene_info_path}")
    
    with open(scene_info_path, 'r') as f:
        scene_info = json.load(f)
    
    print(f"📋 加载scene_info.json: {len(scene_info)} 个episodes")
    
    trajectories = []
    data_dir = src_path / "data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"未找到data目录: {data_dir}")
    
    for episode_key, episode_meta in tqdm(sorted(scene_info.items(), key=lambda x: x[1]['episode_id'])):
        episode_id = episode_meta['episode_id']
        
        # 过滤成功的轨迹
        if filter_success and not episode_meta.get('success', False):
            continue
        
        # 加载HDF5文件
        hdf5_path = data_dir / f"episode{episode_id}.hdf5"
        if not hdf5_path.exists():
            print(f"  ⚠️  跳过episode {episode_id}: 未找到 {hdf5_path.name}")
            continue
        
        try:
            import h5py
            from PIL import Image
            import io
            
            episode_data = []
            
            with h5py.File(hdf5_path, 'r') as f:
                # 读取action (T, 14)
                actions = f['joint_action']['vector'][:]  # (T, 14) float64
                
                # 读取图像 (T,) bytes
                rgb_bytes = f['observation']['head_camera']['rgb'][:]
                
                # 解码每一帧
                for step_idx in range(len(actions)):
                    # 解码JPEG图像
                    img = Image.open(io.BytesIO(rgb_bytes[step_idx]))
                    img_array = np.array(img)  # [H, W, 3], uint8
                    
                    episode_data.append({
                        'head_camera': img_array,
                        'action': actions[step_idx],
                        'qpos': actions[step_idx]  # 使用action作为qpos的近似
                    })
            
            # 获取任务指令
            instruction = episode_meta.get('instruction', 'click the bell')
            
            trajectories.append({
                'episode_id': episode_id,
                'instruction': instruction,
                'success': episode_meta.get('success', False),
                'steps': episode_data
            })
            
        except Exception as e:
            print(f"  ❌ 加载episode {episode_id} 失败: {e}")
            continue
    
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="将RobotWin轨迹转换为Wan环境npy格式")
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="RobotWin原始数据路径"
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        required=True,
        help="输出npy文件路径"
    )
    parser.add_argument(
        "--max_trajs",
        type=int,
        default=None,
        help="最大转换轨迹数 (None表示全部)"
    )
    parser.add_argument(
        "--enable_kir",
        action="store_true",
        default=True,
        help="是否生成KIR版本"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="图像目标尺寸 (H W)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="click the bell",
        help="任务描述"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    dst_path = Path(args.dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"RobotWin → Wan环境 数据转换")
    print(f"=" * 60)
    print(f"源目录: {args.src_dir}")
    print(f"目标目录: {args.dst_dir}")
    print(f"图像尺寸: {args.image_size}")
    print(f"KIR模式: {'启用' if args.enable_kir else '禁用'}")
    print(f"任务描述: {args.task}")
    print(f"=" * 60)
    
    # 加载数据
    print("\n📂 加载RobotWin数据...")
    filter_success = not getattr(args, 'no_filter_success', False)
    trajectories = load_robotwin_data(args.src_dir, filter_success=filter_success)
    
    if not trajectories:
        print("❌ 未找到任何轨迹数据，请检查源目录结构")
        return
    
    print(f"✅ 成功加载 {len(trajectories)} 条轨迹")
    
    # 限制转换数量
    if args.max_trajs is not None:
        trajectories = trajectories[:args.max_trajs]
        print(f"📊 将转换前 {args.max_trajs} 条轨迹")
    
    # 转换轨迹
    print(f"\n🔄 开始转换...")
    all_saved_files = []
    
    for traj_idx, trajectory in enumerate(tqdm(trajectories, desc="转换轨迹")):
        try:
            saved_files = convert_single_trajectory(
                trajectory_data=trajectory,
                traj_idx=traj_idx,
                enable_kir=args.enable_kir,
                image_size=tuple(args.image_size),
                task=args.task
            )
            all_saved_files.extend(saved_files)
        except Exception as e:
            print(f"\n❌ 轨迹 {traj_idx} 转换失败: {e}")
            continue
    
    # 移动文件到目标目录
    print(f"\n📦 移动文件到目标目录...")
    for file_path in all_saved_files:
        if Path(file_path).exists():
            dest_path = dst_path / Path(file_path).name
            Path(file_path).rename(dest_path)
    
    print(f"\n" + "=" * 60)
    print(f"✅ 转换完成!")
    print(f"   - 总轨迹数: {len(trajectories)}")
    print(f"   - 生成文件数: {len(all_saved_files)}")
    print(f"   - 输出目录: {args.dst_dir}")
    print(f"=" * 60)
    
    # 显示示例文件
    sample_files = list(dst_path.glob("*.npy"))[:5]
    if sample_files:
        print(f"\n示例文件:")
        for f in sample_files:
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
