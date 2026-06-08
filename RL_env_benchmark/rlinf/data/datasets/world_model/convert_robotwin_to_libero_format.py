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
将RobotWin HDF5轨迹转换为与LIBERO一致的格式。

## 数据格式对比

### LIBERO格式 (目标格式)
```python
# 每个npy文件存储一个轨迹的帧列表
[
    {
        'image': np.ndarray,           # [H, W, 3], uint8, [0, 255]
        'delta_action': np.ndarray,    # [7], float64
        'instruction': str             # 任务描述
    },
    {
        'image': np.ndarray,
        'delta_action': np.ndarray,
        'instruction': str
    },
    ...  # T帧
]
```

### RobotWin HDF5源格式
```
episode0.hdf5
├── joint_action/vector: (T, 14) float64
└── observation/head_camera/rgb: (T,) bytes (JPEG)

scene_info.json
{
    "episode_0": {
        "instruction": "click the bell",
        "success": true,
        ...
    }
}
```

## 转换策略

1. 从HDF5提取所有帧的图像和action
2. 将14维action转换为7维 (仅使用右臂，或根据任务配置)
3. 从scene_info.json获取instruction
4. 保存为与LIBERO一致的列表格式

## 使用方法

```bash
cd /your/path//RLinf

# 转换前10条轨迹测试
python rlinf/data/datasets/world_model/convert_robotwin_to_liberp_format.py \\
    --src_dir /manifold-obs/wzl/vla_robotwin_4k_320/fulldata_40000/click_bell/demo_clean \\
    --dst_dir /your/path//RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/dataset \\
    --max_trajs 10 \\
    --use_right_arm_only

# 转换全部数据
python rlinf/data/datasets/world_model/convert_robotwin_to_liberp_format.py \\
    --src_dir /manifold-obs/wzl/vla_robotwin_4k_320/fulldata_40000/click_bell/demo_clean \\
    --dst_dir /your/path//RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/dataset \\
    --use_right_arm_only
```
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import io
from tqdm import tqdm


def extract_right_arm_action(full_action_14d):
    """
    从14维action提取右臂7维action。
    
    14维action结构:
    - [0:6]: 左臂6关节
    - [6]: 左臂夹爪
    - [7:13]: 右臂6关节
    - [13]: 右臂夹爪
    
    Args:
        full_action_14d: np.ndarray, shape (14,)
        
    Returns:
        right_action_7d: np.ndarray, shape (7,)
    """
    # 右臂: 索引7-13
    return full_action_14d[7:14].copy()


def convert_hdf5_to_liberp_format(
    hdf5_path,
    instruction,
    use_abs_action=True,
    max_frames=None
):
    """
    将单个HDF5文件转换为LIBERO格式。
    
    Args:
        hdf5_path: HDF5文件路径
        instruction: 任务指令
        use_abs_action: 是否使用绝对action (14维) 或 delta action (7维)
        max_frames: 最大帧数 (None表示全部)
        
    Returns:
        trajectory: list of dict, LIBERO格式
    """
    trajectory = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # 读取action (T, 14)
        actions = f['joint_action']['vector'][:]  # (T, 14) float64
        
        # 读取图像 (T,) bytes
        rgb_bytes = f['observation']['head_camera']['rgb'][:]
        
        # 限制帧数
        num_frames = len(actions)
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
        
        # 转换每一帧
        for step_idx in range(num_frames):
            # 解码JPEG图像
            img = Image.open(io.BytesIO(rgb_bytes[step_idx]))
            img_array = np.array(img)  # [H, W, 3], uint8, [0, 255]
            
            # 提取action
            if use_abs_action:
                # 使用14维绝对action
                action = actions[step_idx].copy()
                action_key = 'abs_action'
            else:
                # 使用右臂7维delta action (不推荐)
                action = actions[step_idx, 7:14].copy()
                action_key = 'delta_action'
            
            trajectory.append({
                'image': img_array,
                action_key: action,
                'instruction': instruction
            })
    
    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="将RobotWin HDF5转换为LIBERO格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换成功轨迹，仅使用右臂
  python %(prog)s --src_dir /path/to/demo_clean --dst_dir /path/to/output --use_right_arm_only
  
  # 转换所有轨迹，使用完整14维action
  python %(prog)s --src_dir /path/to/demo_clean --dst_dir /path/to/output --no-filter_success
        """
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="RobotWin数据目录 (包含scene_info.json和data/)"
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
        "--filter_success",
        action="store_true",
        default=True,
        help="仅转换成功的轨迹 (默认: True)"
    )
    parser.add_argument(
        "--no_filter_success",
        action="store_true",
        default=False,
        help="转换所有轨迹"
    )
    parser.add_argument(
        "--use_abs_action",
        action="store_true",
        default=True,
        help="使用14维绝对action (推荐) (默认: True)"
    )
    parser.add_argument(
        "--use_delta_action",
        action="store_true",
        default=False,
        help="使用7维delta action (右臂) (不推荐)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="每条轨迹最大帧数 (None表示全部)"
    )
    
    args = parser.parse_args()
    
    filter_success = not args.no_filter_success
    use_abs_action = not args.use_delta_action  # 默认使用abs_action
    
    src_path = Path(args.src_dir)
    dst_path = Path(args.dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("RobotWin HDF5 → LIBERO格式 转换")
    print("=" * 60)
    print(f"源目录: {args.src_dir}")
    print(f"目标目录: {args.dst_dir}")
    print(f"过滤成功轨迹: {'是' if filter_success else '否'}")
    print(f"Action类型: {'14维绝对action' if use_abs_action else '7维delta action (右臂)'}")
    if args.max_frames:
        print(f"最大帧数: {args.max_frames}")
    print("=" * 60)
    
    # 加载scene_info.json
    scene_info_path = src_path / "scene_info.json"
    if not scene_info_path.exists():
        raise FileNotFoundError(f"未找到scene_info.json: {scene_info_path}")
    
    with open(scene_info_path, 'r') as f:
        scene_info = json.load(f)
    
    print(f"\n📋 加载scene_info.json: {len(scene_info)} 个episodes")
    
    data_dir = src_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"未找到data目录: {data_dir}")
    
    # 转换轨迹
    converted_count = 0
    skipped_count = 0
    failed_count = 0
    
    print(f"\n🔄 开始转换...")
    for episode_key, episode_meta in tqdm(sorted(scene_info.items(), key=lambda x: x[1]['episode_id']), desc="处理episodes"):
        episode_id = episode_meta['episode_id']
        
        # 过滤
        if filter_success and not episode_meta.get('success', False):
            skipped_count += 1
            continue
        
        if args.max_trajs is not None and converted_count >= args.max_trajs:
            break
        
        # 加载HDF5
        hdf5_path = data_dir / f"episode{episode_id}.hdf5"
        if not hdf5_path.exists():
            print(f"  ⚠️  跳过episode {episode_id}: 未找到 {hdf5_path.name}")
            skipped_count += 1
            continue
        
        try:
            instruction = episode_meta.get('instruction', 'click the bell')
            
            # 转换为LIBERO格式
            trajectory = convert_hdf5_to_liberp_format(
                hdf5_path=str(hdf5_path),
                instruction=instruction,
                use_abs_action=use_abs_action,
                max_frames=args.max_frames
            )
            
            if len(trajectory) == 0:
                print(f"  ⚠️  episode {episode_id} 轨迹为空，跳过")
                failed_count += 1
                continue
            
            # 保存为npy
            output_path = dst_path / f"traj{converted_count}.npy"
            np.save(output_path, trajectory)
            
            converted_count += 1
            
        except Exception as e:
            print(f"  ❌ episode {episode_id} 转换失败: {e}")
            failed_count += 1
            continue
    
    print(f"\n" + "=" * 60)
    print(f"✅ 转换完成!")
    print(f"   - 成功转换: {converted_count} 条轨迹")
    print(f"   - 跳过: {skipped_count} 条")
    print(f"   - 失败: {failed_count} 条")
    print(f"   - 输出目录: {args.dst_dir}")
    print(f"=" * 60)
    
    # 验证第一个文件
    if converted_count > 0:
        print(f"\n🔍 验证第一个文件...")
        first_file = dst_path / "traj0.npy"
        data = np.load(first_file, allow_pickle=True)
        print(f"  文件: traj0.npy")
        print(f"  类型: {type(data)}")
        print(f"  长度: {len(data)} 帧")
        print(f"  第一帧keys: {list(data[0].keys())}")
        print(f"  image shape: {data[0]['image'].shape}, dtype: {data[0]['image'].dtype}")
        
        if use_abs_action:
            print(f"  abs_action shape: {data[0]['abs_action'].shape}, dtype: {data[0]['abs_action'].dtype}")
            print(f"  abs_action[0]: {data[0]['abs_action']}")
        else:
            print(f"  delta_action shape: {data[0]['delta_action'].shape}, dtype: {data[0]['delta_action'].dtype}")
            print(f"  delta_action[0]: {data[0]['delta_action']}")
        
        print(f"  instruction: {data[0]['instruction'][:50]}...")


if __name__ == "__main__":
    main()
