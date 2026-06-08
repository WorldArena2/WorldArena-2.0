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
可视化3个VLA模型在某个任务上的action在各个维度的分布。

数据结构:
/manifold-obs/wzl/vla_robotwin_4k_320/
├── fulldata_40000/    # VLA模型1: fulldata
│   └── click_bell/    # 任务1
│       └── demo_clean/
├── 10radiodata_10000/ # VLA模型2: 10radiodata
│   └── click_bell/
│       └── demo_clean/
└── ref/               # VLA模型3: ref
    └── click_bell/
        └── demo_clean/

Action维度: 14维 (左臂6关节+1夹爪, 右臂6关节+1夹爪)

使用方法:
```bash
cd /your/path//RLinf

# 可视化click_bell任务的action分布
python rlinf/data/datasets/world_model/visualize_action_distribution.py \\
    --base_dir /manifold-obs/wzl/vla_robotwin_4k_320 \\
    --task click_bell \\
    --output_dir /your/path//RLinf/visualizations/action_distribution

# 可视化adjust_bottle任务
python rlinf/data/datasets/world_model/visualize_action_distribution.py \\
    --base_dir /manifold-obs/wzl/vla_robotwin_4k_320 \\
    --task adjust_bottle \\
    --output_dir /your/path//RLinf/visualizations/action_distribution

# 限制每个VLA分析的轨迹数
python rlinf/data/datasets/world_model/visualize_action_distribution.py \\
    --base_dir /manifold-obs/wzl/vla_robotwin_4k_320 \\
    --task click_bell \\
    --max_trajs 50 \\
    --output_dir /your/path//RLinf/visualizations/action_distribution
```
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Action维度说明
ACTION_DIM_LABELS = [
    "Left Joint 1", "Left Joint 2", "Left Joint 3",
    "Left Joint 4", "Left Joint 5", "Left Joint 6",
    "Left Gripper",
    "Right Joint 1", "Right Joint 2", "Right Joint 3",
    "Right Joint 4", "Right Joint 5", "Right Joint 6",
    "Right Gripper"
]

# VLA模型名称映射
VLA_MODEL_NAMES = {
    "fulldata_40000": "FullData (40K)",
    "10radiodata_10000": "10Radiodata (10K)",
    "ref": "Reference"
}

# 颜色方案
VLA_COLORS = {
    "fulldata_40000": "#2196F3",    # 蓝色
    "10radiodata_10000": "#FF9800", # 橙色
    "ref": "#4CAF50"                # 绿色
}


def load_actions_from_hdf5(hdf5_path: str) -> np.ndarray:
    """
    从HDF5文件加载action数据。
    
    Args:
        hdf5_path: HDF5文件路径
        
    Returns:
        actions: (T, 14) 的action数组
    """
    with h5py.File(hdf5_path, 'r') as f:
        actions = f['joint_action']['vector'][:]  # (T, 14) float64
    return actions


def load_all_actions_for_task(
    base_dir: str,
    vla_name: str,
    task: str,
    max_trajs: int = None,
    filter_success: bool = True
) -> np.ndarray:
    """
    加载某个VLA模型在某个任务上的所有action。
    
    Args:
        base_dir: 基础数据目录
        vla_name: VLA模型名称 (fulldata_40000, 10radiodata_10000, ref)
        task: 任务名称 (click_bell, adjust_bottle)
        max_trajs: 最大轨迹数 (None表示全部)
        filter_success: 是否仅加载成功的轨迹
        
    Returns:
        all_actions: (N, 14) 的所有action拼接
    """
    task_dir = Path(base_dir) / vla_name / task / "demo_clean"
    
    if not task_dir.exists():
        raise FileNotFoundError(f"任务目录不存在: {task_dir}")
    
    # 加载scene_info.json
    scene_info_path = task_dir / "scene_info.json"
    if not scene_info_path.exists():
        raise FileNotFoundError(f"未找到scene_info.json: {scene_info_path}")
    
    with open(scene_info_path, 'r') as f:
        scene_info = json.load(f)
    
    print(f"\n📂 加载 {VLA_MODEL_NAMES[vla_name]} - {task}")
    print(f"  总episodes: {len(scene_info)}")
    
    # 过滤成功的轨迹（仅当scene_info包含success字段时）
    if filter_success:
        # 检查是否有success字段（ref数据可能没有）
        has_success_field = any('success' in v for v in scene_info.values())
        if has_success_field:
            scene_info = {k: v for k, v in scene_info.items() if v.get('success', False)}
            print(f"  成功的episodes: {len(scene_info)}")
        else:
            print(f"  ⚠️  scene_info无success字段，加载所有轨迹")
    
    # 限制轨迹数
    if max_trajs is not None:
        scene_info = dict(list(scene_info.items())[:max_trajs])
        print(f"  将分析前 {len(scene_info)} 条轨迹")
    
    # 加载所有action
    data_dir = task_dir / "data"
    all_actions = []
    
    for episode_key, episode_meta in tqdm(scene_info.items(), desc=f"  加载 {vla_name}"):
        # 兼容不同的scene_info格式：
        # - fulldata/10radiodata: 包含 'episode_id' 字段
        # - ref: 不包含 'episode_id'，需要从episode_key提取
        if 'episode_id' in episode_meta:
            episode_id = episode_meta['episode_id']
        else:
            # 从episode_key提取（如 "episode_0" -> 0）
            try:
                episode_id = int(episode_key.split('_')[-1])
            except:
                print(f"  ⚠️  无法解析episode_id: {episode_key}")
                continue
        
        hdf5_path = data_dir / f"episode{episode_id}.hdf5"
        
        if not hdf5_path.exists():
            continue
        
        try:
            actions = load_actions_from_hdf5(str(hdf5_path))
            all_actions.append(actions)
        except Exception as e:
            print(f"  ⚠️  加载episode {episode_id} 失败: {e}")
            continue
    
    if not all_actions:
        raise ValueError(f"未找到任何有效的action数据: {vla_name} - {task}")
    
    # 拼接所有action
    all_actions = np.concatenate(all_actions, axis=0)  # (N, 14)
    print(f"  ✅ 加载完成: {all_actions.shape[0]} 个action steps")
    
    return all_actions


def compute_action_statistics(all_actions: np.ndarray) -> Dict:
    """
    计算action的统计信息。
    
    Args:
        all_actions: (N, 14) 的action数组
        
    Returns:
        stats: 包含均值、标准差、最小值、最大值、中位数等的字典
    """
    stats = {
        'mean': np.mean(all_actions, axis=0),
        'std': np.std(all_actions, axis=0),
        'min': np.min(all_actions, axis=0),
        'max': np.max(all_actions, axis=0),
        'median': np.median(all_actions, axis=0),
        'q25': np.percentile(all_actions, 25, axis=0),
        'q75': np.percentile(all_actions, 75, axis=0),
    }
    return stats


def plot_action_distribution_comparison(
    vla_actions_dict: Dict[str, np.ndarray],
    task: str,
    output_dir: str
):
    """
    绘制3个VLA模型的action分布对比图。
    
    Args:
        vla_actions_dict: {vla_name: actions_array} 字典
        task: 任务名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_dims = 14
    fig, axes = plt.subplots(7, 2, figsize=(16, 28))
    axes = axes.flatten()
    
    # 为每个维度绘制分布对比
    for dim in range(num_dims):
        ax = axes[dim]
        
        for vla_name, actions in vla_actions_dict.items():
            # 提取该维度的所有action值
            dim_values = actions[:, dim]
            
            # 绘制直方图
            ax.hist(
                dim_values,
                bins=50,
                alpha=0.6,
                color=VLA_COLORS[vla_name],
                label=VLA_MODEL_NAMES[vla_name],
                density=True
            )
            
            # 绘制统计信息
            mean_val = np.mean(dim_values)
            ax.axvline(
                mean_val,
                color=VLA_COLORS[vla_name],
                linestyle='--',
                linewidth=2,
                alpha=0.8
            )
        
        ax.set_xlabel('Action Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{ACTION_DIM_LABELS[dim]} (Dim {dim})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f'Action Distribution Comparison - {task.replace("_", " ").title()}',
        fontsize=16,
        fontweight='bold',
        y=1.01
    )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f"action_distribution_{task}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 分布对比图已保存: {output_path}")
    
    plt.close()


def plot_action_statistics_comparison(
    vla_stats_dict: Dict[str, Dict],
    task: str,
    output_dir: str
):
    """
    绘制3个VLA模型的action统计信息对比图。
    
    Args:
        vla_stats_dict: {vla_name: stats_dict} 字典
        task: 任务名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 统计指标
    stat_metrics = [
        ('mean', 'Mean', '均值'),
        ('std', 'Std', '标准差'),
        ('min', 'Min', '最小值'),
        ('max', 'Max', '最大值')
    ]
    
    x = np.arange(14)
    width = 0.25
    
    for idx, (stat_key, stat_name, stat_cn) in enumerate(stat_metrics):
        ax = axes[idx // 2, idx % 2]
        
        for i, (vla_name, stats) in enumerate(vla_stats_dict.items()):
            values = stats[stat_key]
            offset = (i - 1) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=VLA_MODEL_NAMES[vla_name],
                color=VLA_COLORS[vla_name],
                alpha=0.8
            )
        
        ax.set_xlabel('Action Dimension', fontsize=11)
        ax.set_ylabel(stat_name, fontsize=11)
        ax.set_title(f'{stat_cn} across Action Dimensions', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i}' for i in range(14)], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(
        f'Action Statistics Comparison - {task.replace("_", " ").title()}',
        fontsize=16,
        fontweight='bold',
        y=1.01
    )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f"action_statistics_{task}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 统计对比图已保存: {output_path}")
    
    plt.close()


def print_statistics_summary(
    vla_stats_dict: Dict[str, Dict],
    task: str
):
    """
    打印统计信息摘要。
    
    Args:
        vla_stats_dict: {vla_name: stats_dict} 字典
        task: 任务名称
    """
    print(f"\n{'='*80}")
    print(f"Action Statistics Summary - {task.replace('_', ' ').title()}")
    print(f"{'='*80}")
    
    for vla_name, stats in vla_stats_dict.items():
        print(f"\n📊 {VLA_MODEL_NAMES[vla_name]}:")
        print(f"{'Dim':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
        print("-" * 60)
        
        for dim in range(14):
            print(
                f"{dim:<6} "
                f"{stats['mean'][dim]:>10.4f} "
                f"{stats['std'][dim]:>10.4f} "
                f"{stats['min'][dim]:>10.4f} "
                f"{stats['max'][dim]:>10.4f} "
                f"{stats['median'][dim]:>10.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="可视化3个VLA模型在某个任务上的action在各个维度的分布"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/manifold-obs/wzl/vla_robotwin_4k_320",
        help="基础数据目录"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["click_bell", "adjust_bottle"],
        help="任务名称"
    )
    parser.add_argument(
        "--max_trajs",
        type=int,
        default=None,
        help="每个VLA模型最大分析的轨迹数 (None表示全部)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations/action_distribution",
        help="输出目录"
    )
    parser.add_argument(
        "--no_filter_success",
        action="store_true",
        help="不过滤，分析所有轨迹（包括失败的）"
    )
    
    args = parser.parse_args()
    
    # VLA模型列表
    vla_models = ["fulldata_40000", "10radiodata_10000", "ref"]
    filter_success = not args.no_filter_success
    
    print(f"{'='*80}")
    print(f"VLA Action Distribution Visualization")
    print(f"{'='*80}")
    print(f"Base directory: {args.base_dir}")
    print(f"Task: {args.task}")
    print(f"Max trajectories per VLA: {args.max_trajs if args.max_trajs else 'All'}")
    print(f"Filter success only: {filter_success}")
    print(f"Output directory: {args.output_dir}")
    
    # 加载所有VLA的action数据
    vla_actions_dict = {}
    vla_stats_dict = {}
    
    for vla_name in vla_models:
        try:
            actions = load_all_actions_for_task(
                base_dir=args.base_dir,
                vla_name=vla_name,
                task=args.task,
                max_trajs=args.max_trajs,
                filter_success=filter_success
            )
            
            vla_actions_dict[vla_name] = actions
            
            # 计算统计信息
            stats = compute_action_statistics(actions)
            vla_stats_dict[vla_name] = stats
            
        except Exception as e:
            print(f"❌ 加载 {vla_name} 失败: {e}")
            continue
    
    if not vla_actions_dict:
        print("❌ 未加载到任何VLA数据")
        return
    
    # 打印统计信息
    print_statistics_summary(vla_stats_dict, args.task)
    
    # 绘制分布对比图
    print(f"\n📈 生成分布对比图...")
    plot_action_distribution_comparison(
        vla_actions_dict,
        args.task,
        args.output_dir
    )
    
    # 绘制统计对比图
    print(f"📈 生成统计对比图...")
    plot_action_statistics_comparison(
        vla_stats_dict,
        args.task,
        args.output_dir
    )
    
    print(f"\n{'='*80}")
    print(f"✅ 可视化完成!")
    print(f"   输出目录: {args.output_dir}")
    print(f"   生成的文件:")
    print(f"     - action_distribution_{args.task}.png (分布对比)")
    print(f"     - action_statistics_{args.task}.png (统计对比)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
