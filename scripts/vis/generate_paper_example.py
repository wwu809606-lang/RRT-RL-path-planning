#!/usr/bin/env python3
"""
生成论文用的简单示例图（2D对比）
包含无效扩展，一左一右对比形式
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import random

# 设置论文风格 - 使用 Times 风格字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1


def create_simple_scene():
    """创建一个简单的示例场景"""
    # 场景边界
    bounds = {'x_min': 0, 'x_max': 200, 'y_min': 0, 'y_max': 200}

    # 简单建筑（3个矩形建筑，形成障碍）
    buildings = [
        # 中间的大障碍物
        {'coords': [[70, 70], [130, 70], [130, 130], [70, 130]]},
        # 左上角的障碍物
        {'coords': [[30, 140], [60, 140], [60, 170], [30, 170]]},
        # 右下角的障碍物
        {'coords': [[140, 30], [170, 30], [170, 60], [140, 60]]},
    ]

    # 起点和终点（2D）
    start = [20, 20]
    goal = [180, 180]

    return bounds, buildings, start, goal


def generate_baseline_path(start, goal, buildings):
    """生成 baseline 路径（2D，绕开障碍物，更多节点）"""
    path = [start]
    # 绕开中间障碍物的路径（绕远路，更多节点）
    path.append([35, 35])   # 第1个节点
    path.append([50, 50])   # 第2个节点
    path.append([50, 90])   # 第3个节点，向上
    path.append([50, 120])  # 第4个节点，继续向上
    path.append([50, 140])  # 第5个节点
    path.append([90, 140])  # 第6个节点，向右
    path.append([120, 140]) # 第7个节点，继续向右
    path.append([140, 140]) # 第8个节点
    path.append([160, 160]) # 第9个节点，斜向目标
    path.append(goal)

    # 生成5条短无效扩展，独立不相交
    invalid_edges = [
        {'from': [38, 38], 'to': [65, 65]},     # 从第1节点附近尝试
        {'from': [52, 52], 'to': [75, 75]},     # 从第2节点向右尝试
        {'from': [52, 92], 'to': [75, 105]},    # 从第3节点向右尝试
        {'from': [92, 142], 'to': [115, 155]},  # 从第6节点斜向尝试
        {'from': [142, 142], 'to': [165, 125]}, # 从第8节点向右上尝试
    ]

    return np.array(path), invalid_edges


def generate_rl_path(start, goal, buildings):
    """生成 RL 路径（2D，更优的路径）"""
    path = [start]
    # 更优的路径，找到更好的绕行路线
    path.append([60, 40])   # 从右下角绕
    path.append([60, 130])  # 向上
    path.append([130, 130]) # 向右
    path.append([160, 160]) # 斜向目标
    path.append(goal)

    # 生成2条短无效扩展
    invalid_edges = [
        {'from': [65, 45], 'to': [85, 65]},  # 从第一节点短尝试
        {'from': [135, 135], 'to': [155, 115]}, # 从第三节点短尝试
    ]

    return np.array(path), invalid_edges


def draw_building(ax, coords, **kwargs):
    """绘制2D建筑"""
    color = kwargs.get('color', 'lightgray')
    alpha = kwargs.get('alpha', 0.5)
    edgecolor = kwargs.get('edgecolor', 'black')
    linewidth = kwargs.get('linewidth', 1.0)

    polygon = Polygon(coords, closed=True,
                     facecolor=color, edgecolor=edgecolor,
                     linewidth=linewidth, alpha=alpha)
    ax.add_patch(polygon)


def plot_2d_comparison():
    """绘制2D对比图（一左一右）"""
    bounds, buildings, start, goal = create_simple_scene()

    # 生成路径
    baseline_path, baseline_invalid = generate_baseline_path(start, goal, buildings)
    rl_path, rl_invalid = generate_rl_path(start, goal, buildings)

    # 创建图形，左右两个子图，减少间距
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.05)  # 进一步减少子图之间的间距

    # ========== 左图：Baseline ==========
    # 绘制建筑
    for building in buildings:
        draw_building(ax1, building['coords'], color='lightgray', alpha=0.6, edgecolor='black', linewidth=1.5)

    # 绘制起点和终点（无黑边）
    ax1.plot(start[0], start[1], 'o', color='blue', markersize=15, label='Start', zorder=10)
    ax1.plot(goal[0], goal[1], '*', color='red', markersize=18, label='Goal', zorder=10)

    # 绘制路径
    ax1.plot(baseline_path[:, 0], baseline_path[:, 1], '-o',
            color='#08519c', linewidth=3, markersize=6, label='Baseline (RRT)', zorder=5)

    # 绘制无效扩展
    for edge in baseline_invalid:
        ax1.plot([edge['from'][0], edge['to'][0]],
                [edge['from'][1], edge['to'][1]],
                '--', color='#4292c6', linewidth=2, alpha=0.6)

    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Baseline (RRT)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # ========== 右图：RL (Ours) ==========
    # 绘制建筑
    for building in buildings:
        draw_building(ax2, building['coords'], color='lightgray', alpha=0.6, edgecolor='black', linewidth=1.5)

    # 绘制起点和终点（无黑边）
    ax2.plot(start[0], start[1], 'o', color='blue', markersize=15, label='Start', zorder=10)
    ax2.plot(goal[0], goal[1], '*', color='red', markersize=18, label='Goal', zorder=10)

    # 绘制路径
    ax2.plot(rl_path[:, 0], rl_path[:, 1], '-s',
            color='#d95f0e', linewidth=3, markersize=6, label='RL (Ours)', zorder=5)

    # 绘制无效扩展
    for edge in rl_invalid:
        ax2.plot([edge['from'][0], edge['to'][0]],
                [edge['from'][1], edge['to'][1]],
                '--', color='#e6550d', linewidth=2, alpha=0.6)

    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 200)
    ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) RL guided', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()

    # 保存
    output_path = '/home/wuwenjun/uav_path_planning/results/vis/paper_comparison_2d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 2D对比图已保存: {output_path}")

    plt.close()


if __name__ == '__main__':
    print("生成2D对比论文示例图...")
    print()

    plot_2d_comparison()

    print()
    print("完成！图片保存在: results/vis/")
    print("  - paper_comparison_2d.png (2D对比图)")
