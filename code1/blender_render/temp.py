import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
from dataclasses import dataclass

@dataclass
class MagnetInfo:
    """磁铁信息"""
    position: np.ndarray  # 中心位置
    direction: np.ndarray  # S->N 的方向向量
    length: float  # 长度

@dataclass
class NodeInfo:
    """节点信息"""
    position: np.ndarray  # 节点位置
    field_direction: np.ndarray  # 磁场方向（单位向量）
    angle: float  # 与x轴的角度（弧度）
    angle_degrees: float  # 与x轴的角度（角度）

def calculate_node_magnetic_info(
    magnet_center: Tuple[float, float],  # 磁铁中心位置
    magnet_direction: Union[float, Tuple[float, float]],  # 可以是角度(弧度)或方向向量
    magnet_length: float,  # 磁铁长度
    node_position: Tuple[float, float],  # 待计算节点位置
    strength: float = 1.0,  # 磁场强度系数
    visualize: bool = False,  # 是否显示可视化
    ax: Optional[plt.Axes] = None  # 可选的matplotlib axes对象
) -> NodeInfo:
    """
    计算特定节点在磁场中的信息并可选择性地可视化
    
    参数:
        magnet_center: 磁铁中心位置 (x, y)
        magnet_direction: 磁铁方向，可以是角度(弧度)或方向向量 (dx, dy)
        magnet_length: 磁铁长度
        node_position: 待计算节点位置 (x, y)
        strength: 磁场强度系数
        visualize: 是否显示可视化
        ax: matplotlib axes对象（可选）
    
    返回:
        NodeInfo 对象，包含节点位置和磁场方向信息
    """
    # 转换输入为numpy数组
    center = np.array(magnet_center)
    node = np.array(node_position)
    
    # 处理方向输入
    if isinstance(magnet_direction, (int, float)):
        # 如果输入是角度，转换为方向向量
        direction = np.array([np.cos(magnet_direction), np.sin(magnet_direction)])
    else:
        # 如果输入是向量，归一化
        direction = np.array(magnet_direction)
        print("direction:",direction)
        direction = direction / np.linalg.norm(direction)
        print(direction)
    
    # 计算磁铁的N极和S极位置
    half_length = magnet_length / 2
    north = center + direction * half_length  # N极
    south = center - direction * half_length  # S极
    print(center)
    print(direction)
    print(half_length)
    print(center + direction * half_length)
    print(north, south) # 打印N极和S极位置
    
    # 计算点到N极和S极的矢量
    r_n = node - north
    r_s = node - south
    
    # 计算距离
    dist_n = np.linalg.norm(r_n)
    dist_s = np.linalg.norm(r_s)
    
    # 避免除零错误
    if dist_n < 1e-10 or dist_s < 1e-10:
        field_direction = np.array([0.0, 0.0])
        angle = 0.0
    else:
        # 计算磁场向量
        field = strength * (r_n / (dist_n ** 3) - r_s / (dist_s ** 3))
        
        # 归一化方向向量
        magnitude = np.linalg.norm(field)
        if magnitude < 1e-10:
            field_direction = np.array([0.0, 0.0])
            angle = 0.0
        else:
            field_direction = field / magnitude
            angle = np.arctan2(field_direction[1], field_direction[0])
    print("angle:",angle)
    print('field_direction:',field_direction)

    # 创建结果对象
    result = NodeInfo(
        position=node,
        field_direction=field_direction,
        angle=angle,
        angle_degrees=np.degrees(angle) % 360
    )
    
    # 可视化部分
    if visualize:
        # 如果没有提供ax，创建新的图形
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        # 绘制磁铁
        ax.plot([south[0], north[0]], [south[1], north[1]], 'r-', linewidth=3, label='Magnet')
        ax.plot(north[0], north[1], 'ro', markersize=10, label='N pole')
        ax.plot(south[0], south[1], 'bo', markersize=10, label='S pole')
        
        # 绘制节点位置
        ax.plot(node[0], node[1], 'go', markersize=8, label='Node')
        
        # 绘制磁场方向
        if not np.all(field_direction == 0):
            # 箭头长度设为磁铁长度的1/4
            arrow_length = magnet_length / 4
            ax.quiver(node[0], node[1], 
                     field_direction[0], field_direction[1],
                     angles='xy', scale_units='xy', scale=1/arrow_length,
                     color='g', width=0.005, label='Field Direction')
        
        # 添加文本信息
        text_info = f'Angle: {result.angle_degrees:.1f}°'
        ax.text(node[0], node[1] + magnet_length/8, text_info,
                horizontalalignment='center', verticalalignment='bottom')
        
        # 设置图形属性
        ax.grid(True)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Magnetic Field Analysis')
        
        # 自动调整坐标轴范围
        margin = magnet_length
        ax.set_xlim(min(south[0], north[0], node[0]) - margin,
                   max(south[0], north[0], node[0]) + margin)
        ax.set_ylim(min(south[1], north[1], node[1]) - margin,
                   max(south[1], north[1], node[1]) + margin)
        
        if ax is None:
            plt.show()
    
    return result

def visualize_multiple_nodes(
    magnet_center: Tuple[float, float],
    magnet_direction: Union[float, Tuple[float, float]],
    magnet_length: float,
    nodes,
    strength: float = 1.0
) -> None:
    """
    可视化多个节点的磁场方向
    
    参数:
        magnet_center: 磁铁中心位置
        magnet_direction: 磁铁方向（角度或向量）
        magnet_length: 磁铁长度
        nodes: 节点位置列表
        strength: 磁场强度系数
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 为每个节点计算并显示磁场方向
    for node in nodes:
        calculate_node_magnetic_info(
            magnet_center=[0,0],
            magnet_direction=magnet_direction,
            magnet_length=magnet_length,
            node_position=node,
            strength=strength,
            visualize=True,
            ax=ax
        )
    
    plt.show()

# 使用示例
def demo():
    # # 示例1：单个节点的分析
    # print("示例1 - 单个节点分析：")
    # result = calculate_node_magnetic_info(
    #     magnet_center=(0, 0),
    #     magnet_direction=np.pi/4,  # 45度角
    #     magnet_length=2.0,
    #     node_position=(1, 1),
    #     visualize=True
    # )
    
    # print(f"节点位置: {result.position}")
    # print(f"磁场方向向量: {result.field_direction}")
    # print(f"磁场方向角度: {result.angle_degrees:.2f}°")
    
    # 示例2：多个节点的分析
    print("\n示例2 - 多个节点分析：")
    nodes = [
      (-1.0474815763714926, 1.2910639499738765)
    ]
    
    visualize_multiple_nodes(
        magnet_center=(0, 0),
        magnet_direction=[0.70710678, -0.70710678],  # 水平放置
        magnet_length=2.5,
        nodes=nodes
    )

if __name__ == "__main__":
    demo()