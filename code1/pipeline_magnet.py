import bpy
import os
import sys
sys.path.append("/home/ulab/.local/lib/python3.11/site-packages")

import argparse
import sys
import random
from mathutils import Vector
import math
import numpy as np
from typing import Tuple, Union, Optional
from dataclasses import dataclass
from datetime import datetime
import csv
import matplotlib.pyplot as plt

sys.path.append("/home/ulab/.local/lib/python3.11/site-packages")  # 请根据实际路径确认
from tqdm import tqdm

# 设置颜色管理为标准模式
bpy.context.scene.view_settings.view_transform = 'Standard'
bpy.context.scene.display_settings.display_device = 'sRGB'

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
    visualize: bool = True,  # 是否显示可视化
    png_name: str = None,
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
        direction = np.array([np.cos(math.radians(magnet_direction)), np.sin(math.radians(magnet_direction))])
        # print("the direction of the current magnet bar:", direction)
    else:
        # 如果输入是向量，归一化
        direction = np.array(magnet_direction)
        direction = direction / np.linalg.norm(direction)
    
    # 计算磁铁的N极和S极位置
    half_length = magnet_length / 2
    north = center + direction * half_length  # N极
    south = center - direction * half_length  # S极
    
    # 计算点到N极和S极的矢量
    r_n = node - north
    r_s = node - south
    
    # 计算距离
    dist_n = np.linalg.norm(r_n)
    dist_s = np.linalg.norm(r_s)
    
    
    # print("center: "  + str(center))
    # print("direction of magnet: " + str(direction))
    # print("half_length: " + str(half_length))
    # print("center + direction * half_length = " + str(center + direction * half_length))
    # print(f"north: {north}, south: {south}") # 打印N极和S极位置
    

    
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
    # print(f"the angle in {node_position}: ",angle)
    # print('field_direction:',field_direction)
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
        
        fig, ax = plt.subplots(figsize=(10, 10))
            
        # 绘制磁铁
        ax.plot([south[0], north[0]], [south[1], north[1]], 'r-', linewidth=4, label='Magnet')
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
        
        if ax is not None:
            plt.savefig(png_name)
    
    return result


def clear_scene():
    """删除当前场景中的所有对象。"""
    bpy.ops.object.select_all(action='SELECT')  # 选择所有对象
    bpy.ops.object.delete()  # 删除选中的对象
    # print("清空场景完成。")

def fit_camera_to_objects_with_random_position(camera, object_names, margin=1.2, over = False):
    """
    随机设置相机位置，并确保指定的对象都在视野中。
    
    参数：
    - camera: 要调整的相机对象
    - object_names: 要包含在视野中的对象名称列表
    - margin: 视野的边距比例，默认1.2表示多留一些空间
    """
    # 获取指定名称的对象
    objects = [bpy.data.objects.get(name) for name in object_names if bpy.data.objects.get(name)]
    
    if len(objects) < len(object_names):
        print("有一些对象名称在场景中未找到，请检查名称是否正确。")
        return
    
    # 计算这些对象的总体边界框
    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    for obj in objects:
        for vertex in obj.bound_box:
            world_vertex = obj.matrix_world @ Vector(vertex)
            min_corner = Vector((min(min_corner[i], world_vertex[i]) for i in range(3)))
            max_corner = Vector((max(max_corner[i], world_vertex[i]) for i in range(3)))
    
    # 计算边界框的中心和尺寸
    bbox_center = (min_corner + max_corner) / 2
    bbox_size = max_corner - min_corner
    max_dim = max(bbox_size) * margin

    # 随机设置相机位置
    random_distance = max_dim * 1.5  # 确保相机距离足够远
    random_angle = random.uniform(0, 2 * math.pi)  # 随机角度
    if over:
      camera.location = Vector((bbox_center.x, bbox_center.y, bbox_center.z + max_dim + 10))
    else:
      
      camera.location = bbox_center + Vector((
          random_distance * math.cos(random_angle),
          random_distance * math.sin(random_angle),
          random.uniform(1, max_dim)  # 随机高度
      ))
      
    # 将相机对准边界框中心
    direction = ( camera.location-bbox_center).normalized()  # 确保方向是单位向量
    camera.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()  # 使用相机的Z轴朝向目标

    # 确保所有对象都在视野内
    bpy.context.view_layer.objects.active = camera
    bpy.context.scene.camera = camera
    bpy.ops.view3d.camera_to_view_selected()





def load_blend_file(filepath, location=(0, 0, 0), scale=(1, 1, 1), rotation_angle=0):
    """
    导入指定的 .blend 文件中的所有对象，并调整位置、缩放和旋转方向。
    
    参数:
    - filepath: str, .blend 文件的路径
    - location: tuple, 导入模型的位置 (x, y, z)
    - scale: tuple, 导入模型的缩放比例 (x, y, z)
    - rotation_angle: float, 导入模型的旋转角度（以弧度为单位）在Z轴方向
    """
    # 导入指定的 .blend 文件中的所有对象
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects  # 选择导入所有对象
    
    # 将对象链接到当前集合并应用位置、缩放和旋转
    for obj in data_to.objects:
        if obj is not None:
            # 将对象链接到当前集合
            bpy.context.collection.objects.link(obj)
            
            # 设置位置和缩放
            obj.location = location
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            
            # Rotate the object along the Z-axis
            # print(rotation_angle * (3.14159 / 180))
            # obj.rotation_euler.z = rotation_angle * (3.14159 / 180)
            # # 使用 bpy.ops.transform.rotate 在 Z 轴上旋转
    bpy.ops.transform.rotate(
        value=math.radians(rotation_angle),
        orient_axis='Z',
        orient_type='GLOBAL',
        constraint_axis=(False, False, True)
      )
    
    # print("场景已导入成功！")
    
def load_blend_file_backgournd(filepath):
    """导入指定的 .blend 文件中的所有对象。"""
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects  # 选择导入所有对象
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
    # print("场景已导入成功！")

# def set_render_parameters(resolution=(1920, 1080), file_format='PNG', output_path="../database/rendered_image.png"):
#     """设置渲染参数，包括分辨率、格式和输出路径。"""
#     bpy.context.scene.render.resolution_x = resolution[0]
#     bpy.context.scene.render.resolution_y = resolution[1]
#     bpy.context.scene.render.resolution_percentage = 100
#     bpy.context.scene.render.filepath = output_path
#     bpy.context.scene.render.image_settings.file_format = file_format
#     print("渲染参数已设置。")

def set_render_parameters(resolution=(1920, 1080), file_format='PNG', output_path="../database/rendered_image.png", samples=500, use_denoising=True, use_transparent_bg=False):
    """设置渲染参数，包括分辨率、格式、输出路径和高质量渲染设置。"""
    # 设置分辨率和输出路径
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    
    # 使用 Cycles 渲染引擎（更高质量）
    #bpy.context.scene.render.engine = 'CYCLES'
    
    # 设置渲染采样（越高质量越好，但时间更长）
    #bpy.context.scene.cycles.samples = samples
    
    # 启用去噪（推荐用于高质量渲染）
    #bpy.context.scene.cycles.use_denoising = use_denoising
    
    # 设置设备为 GPU（如果系统有 GPU，推荐使用 GPU 渲染）
    bpy.context.scene.cycles.device = 'GPU'
    
    # 设置透明背景（如果需要）
    #bpy.context.scene.render.film_transparent = use_transparent_bg
    
    # 设置光线跟踪反弹次数
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.diffuse_bounces = 2
    bpy.context.scene.cycles.glossy_bounces = 2
    bpy.context.scene.cycles.transmission_bounces = 6
    bpy.context.scene.cycles.volume_bounces = 2
    
    # 设置光路径采样（可选）
    #bpy.context.scene.cycles.use_adaptive_sampling = True
    #bpy.context.scene.cycles.adaptive_threshold = 0.01
    

def calculate_rotated_magnetic_moment(rotation_axis, rotation_angle, magnitude=1.0):
    """
    根据旋转轴和旋转角度计算旋转后的磁矩向量。
    
    参数:
    rotation_axis : np.array
        旋转轴的单位向量 (3D)。
    rotation_angle : float
        旋转角度（弧度）。
    magnitude : float, optional
        磁矩的强度，默认为 1.0。
        
    返回:
    rotated_m : np.array
        旋转后的磁矩向量 (3D)。
    """
    # 初始磁矩向量，假设初始方向沿 x 轴
    initial_m = np.array([1.0, 0.0, 0.0]) * magnitude

    # 计算旋转矩阵（使用 Rodrigues' 旋转公式）
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    ux, uy, uz = rotation_axis

    rotation_matrix = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])

    # 计算旋转后的磁矩向量
    rotated_m = np.dot(rotation_matrix, initial_m)
    return rotated_m

def magnetic_field_at_position(m, r_dipole_to_observation, mu_0=4 * np.pi * 1e-7):
    """
    计算磁场 B 在观测点的位置。
    
    参数:
    m : np.array
        条形磁铁的磁矩向量 (3D)。
    r_dipole_to_observation : np.array
        从条形磁铁到观测点（磁针）的位移向量 (3D)。
    mu_0 : float, optional
        真空磁导率，默认为 4π × 10⁻⁷ H/m。
        
    返回:
    B : np.array
        磁场向量 (3D)。
    """
    r_magnitude = np.linalg.norm(r_dipole_to_observation)  # 位置向量的模
    r_unit = r_dipole_to_observation / r_magnitude  # 位置向量的单位向量
    
    # 计算磁场 B 的各部分
    term1 = 3 * r_unit * np.dot(m, r_unit) / r_magnitude**3
    term2 = m / r_magnitude**3
    B = mu_0 / (4 * np.pi) * (term1 - term2)
    
    return B

def magnetic_needle_direction_xy_plane(m, magnet_position, needle_position):
    """
    计算磁针在 x-y 平面上的方向。
    
    参数:
    m : np.array
        条形磁铁的磁矩向量 (3D)。
    magnet_position : np.array
        条形磁铁的位置向量 (3D)。
    needle_position : np.array
        磁针的位置向量 (3D)。
        
    返回:
    needle_direction_xy : np.array
        磁针在 x-y 平面上的方向单位向量 (3D)，z 分量恒等于 0。
    """
    # 从磁铁位置到磁针位置的向量
    r_dipole_to_observation = needle_position - magnet_position
    
    # 计算磁场向量
    B = magnetic_field_at_position(m, r_dipole_to_observation)
    
    # 将磁场向量的 z 分量设为 0，使其仅在 x-y 平面上
    B[2] = 0
    
    # 重新归一化，使其成为 x-y 平面的单位向量
    needle_direction_xy = B / np.linalg.norm(B)
    
    return needle_direction_xy
    
    
def random_point_on_ring(inner_radius, outer_radius):
    """
    inner_radius : float
    outer_radius : float

    point : tuple
        随机生成的点 (x, y, z)。
    """
    # 随机生成一个半径，位于内外半径之间
    radius = random.uniform(inner_radius, outer_radius)
    # 随机生成一个角度
    angle = random.uniform(0, 2 * np.pi)
    
    # 计算点的 x 和 y 坐标
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    
    # z 坐标保持为 0
    return (x, y, -0.03)


def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    csv_file = None,
    iter = 0
  ):
    clear_scene()
    current_time = datetime.now()
    file_name = current_time.strftime("%Y%m%d_%H%M%S")  # 格式化为 YYYYMMDD_HHMMSS
    twoD_output_path = os.path.join(render_output_path, file_name+"_2D.png")
    if 'blank' in background.lower():
      background = "./database/blank_background.blend"
      load_blend_file_backgournd(background)


    blender = "./database/magnet.blend"
    needle = "./database/compass.blend"
    random_rotation_angle = random.uniform(0, 360)
    
    # print(f"rotation_angle of magnet: {random_rotation_angle:.2f}")
    
    load_blend_file(blender, location=(0, 0, 0), scale=(1, 1, 1), rotation_angle=-random_rotation_angle)

    inner_radius = 2.5  # 圆环的内半径
    outer_radius = 5  # 圆环的外半径
    needle_location = random_point_on_ring(inner_radius, outer_radius)
    result = calculate_node_magnetic_info(
        magnet_center=(0, 0),
        magnet_direction=-random_rotation_angle,
        magnet_length=3.9,
        node_position=needle_location[:2],
        visualize=True,
        png_name = twoD_output_path
    )

    rotation_angle = result.angle_degrees
    load_blend_file(filepath = needle, location = needle_location, scale=(1, 1, 1),rotation_angle=result.angle_degrees)


    render_3D_output_path = os.path.join(render_output_path, file_name+"_3D.png")
    set_render_parameters(output_path=render_3D_output_path)
  
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    fit_camera_to_objects_with_random_position(camera, ["Cylinder.003_CompassNeedleHolder_mat_0", "Object_2"]) 
    render_scene()
    
    render_3D_over_output_path = os.path.join(render_output_path, file_name+"_Over_3D.png")
    fit_camera_to_objects_with_random_position(camera, ["Cylinder.003_CompassNeedleHolder_mat_0", "Object_2"], over=True) 
    set_render_parameters(output_path=render_3D_over_output_path)
    render_scene()
    # 将结果写入 CSV 文件
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iter, render_3D_output_path, render_3D_over_output_path,twoD_output_path, -random_rotation_angle, needle_location, result.field_direction])


def save_blend_file(filepath):
    """保存当前场景为指定的 .blend 文件，直接覆盖原有文件。"""
    if os.path.exists(filepath):
        print('remove the existing file')
        os.remove(filepath)  # 删除已有文件
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    # print(f"修改后的场景已保存到：{filepath}")

def render_scene():
    """执行渲染并保存图像。"""
    bpy.ops.render.render(write_still=True)
    # print(f"渲染完成，图像已保存到：{bpy.context.scene.render.filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])

    iteration_time = 45  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = "magnet_scene.csv"
    
    # 检查 CSV 文件是否存在，如果不存在则写入文件头
    try:
        with open(csv_file, mode="r") as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，写入 CSV 文件头
        if not file_exists:
            writer.writerow(["iter", "3D", "3D_over", "2D", "magnet_direction", "needle_location", "needle_direction"])

        # 设置背景、场景和渲染输出路径
        background = "./database/blank_background.blend"
        scene = "Magnetic"
        render_output_path = "./database/rendered_images/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in tqdm(range(arguments.iter, arguments.iter + iteration_time), desc="Rendering"):
            main(
                background=background,
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iter=i
            )
