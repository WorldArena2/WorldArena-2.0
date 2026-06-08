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
验证LIBERO格式的数据集。

使用方法:
    python verify_libero_format_dataset.py \
        --dataset_dir /path/to/dataset \
        --expected_action_dim 7
"""

import argparse
from pathlib import Path

import numpy as np


def verify_libero_format(
    dataset_dir,
    expected_action_dim=14,
    action_key='abs_action',
    max_files=None
):
    """
    验证LIBERO格式的数据集。
    
    LIBERO格式:
    [
        {'image': [H,W,3], 'delta_action/abs_action': [A], 'instruction': str},
        {'image': [H,W,3], 'delta_action/abs_action': [A], 'instruction': str},
        ...
    ]
    
    Args:
        dataset_dir: 数据集目录
        expected_action_dim: 期望的action维度
        action_key: action字段名 ('abs_action' 或 'delta_action')
        max_files: 最大验证文件数
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ 目录不存在: {dataset_dir}")
        return False
    
    npy_files = list(dataset_path.glob("*.npy"))
    if not npy_files:
        print(f"❌ 未找到任何npy文件: {dataset_dir}")
        return False
    
    if max_files is not None:
        npy_files = npy_files[:max_files]
    
    print("=" * 60)
    print("LIBERO格式数据集验证")
    print("=" * 60)
    print(f"数据集目录: {dataset_dir}")
    print(f"文件总数: {len(npy_files)}")
    print(f"期望action维度: {expected_action_dim}")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    total_frames = 0
    fail_details = []
    
    print(f"\n🔍 开始验证...")
    for npy_file in sorted(npy_files):
        try:
            # 加载数据
            data = np.load(npy_file, allow_pickle=True)
            
            # 检查是否是列表
            if not isinstance(data, np.ndarray) or not isinstance(data[0], (dict, np.void)):
                fail_count += 1
                fail_details.append((npy_file.name, "数据格式错误: 不是帧列表"))
                print(f"  ✗ {npy_file.name}: 不是帧列表")
                continue
            
            # 转换为列表
            if isinstance(data, np.ndarray):
                data = data.tolist()
            
            if not isinstance(data, list) or len(data) == 0:
                fail_count += 1
                fail_details.append((npy_file.name, "数据格式错误: 空列表"))
                print(f"  ✗ {npy_file.name}: 空列表")
                continue
            
            # 检查第一帧
            first_frame = data[0]
            
            # 检查必需字段
            required_fields = ['image', action_key, 'instruction']
            missing_fields = [f for f in required_fields if f not in first_frame]
            if missing_fields:
                fail_count += 1
                fail_details.append((npy_file.name, f"缺少字段: {missing_fields}"))
                print(f"  ✗ {npy_file.name}: 缺少字段 {missing_fields}")
                continue
            
            # 检查图像格式
            img = first_frame['image']
            if len(img.shape) != 3 or img.shape[2] != 3:
                fail_count += 1
                fail_details.append((npy_file.name, f"图像形状错误: {img.shape}"))
                print(f"  ✗ {npy_file.name}: 图像形状错误 {img.shape}")
                continue
            
            # 检查图像dtype (应该是uint8)
            if img.dtype != np.uint8:
                fail_count += 1
                fail_details.append((npy_file.name, f"图像dtype错误: {img.dtype}, 期望: uint8"))
                print(f"  ✗ {npy_file.name}: 图像dtype错误 {img.dtype}")
                continue
            
            # 检查action维度
            action = first_frame[action_key]
            if action.shape != (expected_action_dim,):
                fail_count += 1
                fail_details.append((npy_file.name, f"action维度错误: {action.shape}, 期望: ({expected_action_dim},)"))
                print(f"  ✗ {npy_file.name}: action维度错误 {action.shape}")
                continue
            
            # 检查instruction
            instruction = first_frame['instruction']
            if not isinstance(instruction, str) or len(instruction) == 0:
                fail_count += 1
                fail_details.append((npy_file.name, "instruction为空"))
                print(f"  ✗ {npy_file.name}: instruction为空")
                continue
            
            success_count += 1
            total_frames += len(data)
            
            if success_count <= 3:
                print(f"  ✓ {npy_file.name}: {len(data)}帧, img={img.shape}, action={action.shape}")
            
        except Exception as e:
            fail_count += 1
            fail_details.append((npy_file.name, str(e)))
            print(f"  ✗ {npy_file.name}: {e}")
    
    # 统计信息
    print(f"\n" + "=" * 60)
    print(f"验证结果")
    print(f"=" * 60)
    print(f"✅ 通过: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📊 总帧数: {total_frames}")
    print(f"📊 平均每轨迹帧数: {total_frames // success_count if success_count > 0 else 0}")
    print(f"=" * 60)
    
    if fail_details:
        print(f"\n失败详情:")
        for filename, error in fail_details[:10]:
            print(f"  - {filename}: {error}")
        if len(fail_details) > 10:
            print(f"  ... 还有 {len(fail_details) - 10} 个失败")
    
    if fail_count == 0:
        print(f"\n🎉 所有文件验证通过! 数据格式与LIBERO一致。")
        return True
    else:
        print(f"\n⚠️  有 {fail_count} 个文件验证失败")
        return False


def main():
    parser = argparse.ArgumentParser(description="验证LIBERO格式数据集")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="数据集目录路径"
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=14,
        help="期望的action维度 (默认: 14)"
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default='abs_action',
        help="action字段名 (默认: abs_action)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="最大验证文件数 (默认: 全部)"
    )
    
    args = parser.parse_args()
    
    verify_libero_format(
        args.dataset_dir,
        expected_action_dim=args.action_dim,
        action_key=args.action_key,
        max_files=args.max_files
    )


if __name__ == "__main__":
    main()
