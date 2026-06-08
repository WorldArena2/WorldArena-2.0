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
验证Wan环境数据集格式是否正确。

使用方法:
    python verify_wan_dataset.py \
        --dataset_dir /your/path//RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/dataset
"""

import argparse
from pathlib import Path

import numpy as np


def verify_single_file(npy_file, expected_action_dim=14, expected_image_size=(256, 256)):
    """
    验证单个npy文件的格式。
    
    Args:
        npy_file: npy文件路径
        expected_action_dim: 期望的action维度
        expected_image_size: 期望的图像尺寸 (H, W)
        
    Returns:
        (success, error_msg) 元组
    """
    try:
        data = np.load(npy_file, allow_pickle=True).item()
    except Exception as e:
        return False, f"加载失败: {e}"
    
    # 检查必需字段
    required_fields = ['start_items', 'target_items', 'task']
    for field in required_fields:
        if field not in data:
            return False, f"缺少必需字段: {field}"
    
    # 检查start_items
    start_items = data['start_items']
    if not isinstance(start_items, list) or len(start_items) == 0:
        return False, "start_items必须是非空列表"
    
    first_frame = start_items[0]
    if 'image' not in first_frame:
        return False, "start_items[0]缺少'image'字段"
    if 'observation.state' not in first_frame:
        return False, "start_items[0]缺少'observation.state'字段"
    
    # 检查图像格式
    img = first_frame['image']
    if img.shape != (3, expected_image_size[0], expected_image_size[1]):
        return False, f"图像形状错误: {img.shape}, 期望: (3, {expected_image_size[0]}, {expected_image_size[1]})"
    
    # 检查图像值域
    if img.min() < 0 or img.max() > 1.0:
        return False, f"图像值域错误: [{img.min()}, {img.max()}], 期望: [0, 1]"
    
    # 检查target_items
    target_items = data['target_items']
    if not isinstance(target_items, list):
        return False, "target_items必须是列表"
    
    if len(target_items) > 0:
        first_target = target_items[0]
        if 'image' not in first_target:
            return False, "target_items[0]缺少'image'字段"
        if 'action' not in first_target:
            return False, "target_items[0]缺少'action'字段"
        
        # 检查目标图像
        target_img = first_target['image']
        if target_img.shape != (3, expected_image_size[0], expected_image_size[1]):
            return False, f"目标图像形状错误: {target_img.shape}, 期望: (3, {expected_image_size[0]}, {expected_image_size[1]})"
        
        # 检查action维度
        action = first_target['action']
        if action.shape != (expected_action_dim,):
            return False, f"Action维度错误: {action.shape}, 期望: ({expected_action_dim},)"
    
    # 检查task
    task = data['task']
    if not isinstance(task, str) or len(task) == 0:
        return False, "task必须是非空字符串"
    
    return True, "验证通过"


def verify_dataset(
    dataset_dir,
    expected_action_dim=14,
    expected_image_size=(256, 256),
    max_files=None
):
    """
    验证整个数据集。
    
    Args:
        dataset_dir: 数据集目录
        expected_action_dim: 期望的action维度
        expected_image_size: 期望的图像尺寸
        max_files: 最大验证文件数 (None表示全部)
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ 目录不存在: {dataset_dir}")
        return
    
    npy_files = list(dataset_path.glob("*.npy"))
    if not npy_files:
        print(f"❌ 未找到任何npy文件: {dataset_dir}")
        return
    
    if max_files is not None:
        npy_files = npy_files[:max_files]
    
    print(f"=" * 60)
    print(f"Wan环境数据集验证")
    print(f"=" * 60)
    print(f"数据集目录: {dataset_dir}")
    print(f"文件总数: {len(npy_files)}")
    print(f"期望action维度: {expected_action_dim}")
    print(f"期望图像尺寸: {expected_image_size}")
    print(f"=" * 60)
    
    success_count = 0
    fail_count = 0
    fail_details = []
    
    print(f"\n🔍 开始验证...")
    for npy_file in sorted(npy_files):
        success, error_msg = verify_single_file(
            npy_file,
            expected_action_dim=expected_action_dim,
            expected_image_size=expected_image_size
        )
        
        if success:
            success_count += 1
            if success_count <= 3:  # 只显示前3个成功的信息
                print(f"  ✓ {npy_file.name}")
        else:
            fail_count += 1
            fail_details.append((npy_file.name, error_msg))
            print(f"  ✗ {npy_file.name}: {error_msg}")
    
    # 统计信息
    kir_count = len(list(dataset_path.glob("*_kir.npy")))
    normal_count = len(npy_files) - kir_count
    
    print(f"\n" + "=" * 60)
    print(f"验证结果")
    print(f"=" * 60)
    print(f"✅ 通过: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📊 标准文件: {normal_count}")
    print(f"📊 KIR文件: {kir_count}")
    print(f"📊 总计: {len(npy_files)}")
    print(f"=" * 60)
    
    if fail_details:
        print(f"\n失败详情:")
        for filename, error in fail_details[:10]:  # 只显示前10个
            print(f"  - {filename}: {error}")
        if len(fail_details) > 10:
            print(f"  ... 还有 {len(fail_details) - 10} 个失败")
    
    if fail_count == 0:
        print(f"\n🎉 所有文件验证通过!")
        return True
    else:
        print(f"\n⚠️  有 {fail_count} 个文件验证失败")
        return False


def inspect_single_file(npy_file):
    """
    检查单个文件的详细信息。
    
    Args:
        npy_file: npy文件路径
    """
    print(f"\n{'=' * 60}")
    print(f"文件检查: {npy_file}")
    print(f"{'=' * 60}")
    
    data = np.load(npy_file, allow_pickle=True).item()
    
    print(f"\n顶层键: {list(data.keys())}")
    print(f"Task: {data['task']}")
    
    print(f"\nStart Items:")
    print(f"  数量: {len(data['start_items'])}")
    if data['start_items']:
        first = data['start_items'][0]
        print(f"  键: {list(first.keys())}")
        if 'image' in first:
            img = first['image']
            print(f"  图像形状: {img.shape}, dtype: {img.dtype}")
            print(f"  图像值域: [{img.min():.4f}, {img.max():.4f}]")
        if 'observation.state' in first:
            state = first['observation.state']
            print(f"  State形状: {state.shape}, dtype: {state.dtype}")
    
    print(f"\nTarget Items:")
    print(f"  数量: {len(data['target_items'])}")
    if data['target_items']:
        first = data['target_items'][0]
        print(f"  键: {list(first.keys())}")
        if 'image' in first:
            img = first['image']
            print(f"  图像形状: {img.shape}, dtype: {img.dtype}")
            print(f"  图像值域: [{img.min():.4f}, {img.max():.4f}]")
        if 'action' in first:
            action = first['action']
            print(f"  Action形状: {action.shape}, dtype: {action.dtype}")
            print(f"  Action值: {action}")


def main():
    parser = argparse.ArgumentParser(description="验证Wan环境数据集格式")
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
        "--image_height",
        type=int,
        default=256,
        help="期望的图像高度 (默认: 256)"
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="期望的图像宽度 (默认: 256)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="最大验证文件数 (默认: 全部)"
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="检查单个文件的详细信息"
    )
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_single_file(args.inspect)
    else:
        verify_dataset(
            args.dataset_dir,
            expected_action_dim=args.action_dim,
            expected_image_size=(args.image_height, args.image_width),
            max_files=args.max_files
        )


if __name__ == "__main__":
    main()
