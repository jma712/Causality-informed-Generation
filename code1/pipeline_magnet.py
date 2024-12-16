import bpy
import os
import sys
import subprocess
res = subprocess.run(["which", "python"],capture_output=True, text=True).stdout


import argparse
import sys
import random
from mathutils import Vector
import math

# sys.path.append(res.split("bin")[0] + "lib/python3.12/site-packages/")
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.12/site-packages/")
# sys.path.append("/usr/lib/python3.12/site-packages/")
import numpy as np
from typing import Tuple, Union, Optional
from dataclasses import dataclass
from datetime import datetime
import csv
# import matplotlib.pyplot as plt

# graph: https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/magnet_graph.png)

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
    noised_field_direction: np.ndarray  # 加噪后的磁场方向（单位向量）
    noised_angle: float  # 加噪后的角度（弧度）
    noised_angle_degrees: float  # 加噪后的角度（角度）
    noise: float  # 噪声大小

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
    noise = "NAN"
    # 避免除零错误
    if dist_n < 1e-10 or dist_s < 1e-10:
        field_direction = np.array([0.0, 0.0])
        angle = 0.0
        # raise ValueError("The node is too close to the magnet.")
    else:
        # 计算磁场向量
        field = strength * (r_n / (dist_n ** 3) - r_s / (dist_s ** 3))
        
        # 归一化方向向量
        magnitude = np.linalg.norm(field)
        if magnitude < 1e-10:
            field_direction = np.array([0.0, 0.0])
            angle = 0.0
            raise ValueError("The magnetic field is too weak to be detected.")
        else:
            field_direction = field / magnitude
            angle = np.arctan2(field_direction[1], field_direction[0])
            
            
            noise = np.random.normal(0, 0.1 * np.abs(field), size=field.shape)
            # 在原始磁场上加噪声
            noisy_field = field + noise
            # 重新计算加噪后的方向
            noisy_magnitude = np.linalg.norm(noisy_field)
            noisy_field_direction = noisy_field / noisy_magnitude
            noisy_angle = np.arctan2(noisy_field_direction[1], noisy_field_direction[0])


    result = NodeInfo(
        position=node,
        field_direction=field_direction,
        angle=angle,
        angle_degrees=np.degrees(angle) % 360,
        noised_field_direction = noisy_field_direction,
        noised_angle = noisy_angle,
        noised_angle_degrees = np.degrees(noisy_angle) % 360,
        noise = noise
    )
    
    # 可视化部分
    # if visualize:
    #     # 如果没有提供ax，创建新的图形
        
    #     fig, ax = plt.subplots(figsize=(10, 10))
            
    #     # 绘制磁铁
    #     ax.plot([south[0], north[0]], [south[1], north[1]], 'r-', linewidth=4, label='Magnet')
    #     ax.plot(north[0], north[1], 'ro', markersize=10, label='N pole')
    #     ax.plot(south[0], south[1], 'bo', markersize=10, label='S pole')
        
    #     # 绘制节点位置
    #     ax.plot(node[0], node[1], 'go', markersize=8, label='Node')
        
    #     # 绘制磁场方向
    #     if not np.all(field_direction == 0):
    #         # 箭头长度设为磁铁长度的1/4
    #         arrow_length = magnet_length / 4
    #         ax.quiver(node[0], node[1], 
    #                  field_direction[0], field_direction[1],
    #                  angles='xy', scale_units='xy', scale=1/arrow_length,
    #                  color='g', width=0.005, label='Field Direction')
        
    #     # 添加文本信息
    #     text_info = f'Angle: {result.angle_degrees:.1f}°'
    #     ax.text(node[0], node[1] + magnet_length/8, text_info,
    #             horizontalalignment='center', verticalalignment='bottom')
        
    #     # 设置图形属性
    #     ax.grid(True)
    #     ax.set_aspect('equal')
    #     ax.legend()
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_title('Magnetic Field Analysis')
        
    #     # 自动调整坐标轴范围
    #     margin = magnet_length
    #     ax.set_xlim(min(south[0], north[0], node[0]) - margin,
    #                max(south[0], north[0], node[0]) + margin)
    #     ax.set_ylim(min(south[1], north[1], node[1]) - margin,
    #                max(south[1], north[1], node[1]) + margin)
        
    #     if ax is not None:
    #         plt.savefig(png_name)
    
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

def set_render_parameters(resolution=(1920, 1080), file_format='PNG', output_path="../database/rendered_image.png", samples=500, use_denoising=True, use_transparent_bg=False):
    """设置渲染参数，包括分辨率、格式、输出路径和高质量渲染设置。"""
    # 设置分辨率和输出路径
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    bpy.context.scene.cycles.device = 'GPU'
    # bpy.context.scene.render.engine = 'CYCLES'

    # # 设置设备为 GPU
    # preferences = bpy.context.preferences.addons['cycles'].preferences
    # preferences.compute_device_type = 'CUDA'  # 可选：'OPTIX', 'HIP', 'METAL'

    # # 启用所有可用的 GPU 设备
    # for device in preferences.get_devices():
    #     if device.type == 'CUDA' or device.type == 'OPTIX' or device.type == 'METAL':
    #         device.use = True
    #         print(f"启用设备：{device.name}")
    #     else:
    #         print(f"跳过设备：{device.name}")

    # # 设置场景使用 GPU
    # bpy.context.scene.cycles.device = 'GPU'
    # print("GPU 渲染已启用。")
    
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
    return (x, y, -0.4)

def random_point_outside_rotated_rectangle(image_size, rect_size = (2.2411911487579346,11.398683547973633), angle = None):
    """
    在图像范围内但不在中心旋转矩形区域随机生成点。

    参数:
        image_size : tuple
            图像的尺寸 (image_width, image_height)。
        rect_size : tuple
            中心矩形的尺寸 (rect_width, rect_height)。
        angle : float
            矩形围绕原点的旋转角度（以弧度为单位）。

    返回:
        tuple
            随机生成的点 (x, y, z)。
    """
    image_width, image_height = image_size
    rect_width, rect_height = rect_size

    # 旋转矩形的顶点坐标相对于原点
    half_width, half_height = rect_width / 2, rect_height / 2
    rectangle_vertices = [
        (-half_width, -half_height),
        (-half_width, half_height),
        (half_width, half_height),
        (half_width, -half_height),
    ]

    # 计算旋转后的矩形顶点坐标
    rotated_vertices = []
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    for x, y in rectangle_vertices:
        rotated_x = x * cos_angle - y * sin_angle
        rotated_y = x * sin_angle + y * cos_angle
        rotated_vertices.append((rotated_x, rotated_y))

    # 获取旋转矩形的边界范围
    x_coords, y_coords = zip(*rotated_vertices)
    rect_x_min, rect_x_max = min(x_coords), max(x_coords)
    rect_y_min, rect_y_max = min(y_coords), max(y_coords)

    while True:
        # 随机生成点的 x 和 y 坐标在图像范围内
        x = random.uniform(-image_width / 2, image_width / 2)
        y = random.uniform(-image_height / 2, image_height / 2)

        # 检查点是否在旋转矩形内
        point_in_rect = point_in_rotated_rectangle(x, y, rect_width, rect_height, angle)
        if not point_in_rect:
            return (x, y, -0.03)

def point_in_rotated_rectangle(x, y, rect_width, rect_height, angle):
    """
    判断一个点是否在旋转后的矩形内。
    矩形的中心默认在原点。

    参数:
        x, y : float
            点的坐标。
        rect_width, rect_height : float
            矩形的宽度和高度。
        angle : float
            矩形的旋转角度（以弧度为单位）。

    返回:
        bool
            如果点在矩形内返回 True，否则返回 False。
    """
    # 旋转点的坐标到矩形的对齐坐标系
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    rotated_x = x * cos_angle - y * sin_angle
    rotated_y = x * sin_angle + y * cos_angle

    # 检查点是否在对齐的矩形内
    half_width, half_height = rect_width / 2, rect_height / 2
    return (-half_width <= rotated_x <= half_width) and (-half_height <= rotated_y <= half_height)
   
def check_objects_intersection(object_name1, object_name2):
    """
    检查两个对象是否相交（是否穿模）。
    
    参数:
        object_name1 (str): 第一个对象的名称。
        object_name2 (str): 第二个对象的名称。
    
    返回:
        bool: 如果对象相交返回 True，否则返回 False。
    """
    obj1 = bpy.data.objects.get(object_name1)
    obj2 = bpy.data.objects.get(object_name2)
    
    if not obj1:
        print(f"Object '{object_name1}' not found.")
        return False
    if not obj2:
        print(f"Object '{object_name2}' not found.")
        return False
    
    # 保存原始几何体数据
    original_mesh = obj1.data.copy()

    # 创建布尔修改器进行检测
    boolean_mod = obj1.modifiers.new(name="IntersectionCheck", type='BOOLEAN')
    boolean_mod.operation = 'INTERSECT'
    boolean_mod.object = obj2
    
    try:
        # 临时执行布尔操作
        bpy.context.view_layer.objects.active = obj1
        bpy.ops.object.modifier_apply(modifier="IntersectionCheck")
        
        # 如果布尔操作成功且生成了几何体，说明对象相交
        if len(obj1.data.polygons) > 0:
            print(f"Objects '{object_name1}' and '{object_name2}' intersect.")
            return True
        else:
            print(f"Objects '{object_name1}' and '{object_name2}' do not intersect.")
            return False
    
    except Exception as e:
        print(f"Error while checking intersection: {e}")
        return False
    
    finally:
        # 恢复原始几何体数据
        obj1.data = original_mesh

def random_point_outside_rotated_inner_ellipse_within_camera(inner_axes, angle, camera_frame):
    """
    随机生成一个点，该点位于旋转内椭圆外部，同时位于相机取景框范围内。

    参数:
        inner_axes (tuple): 内椭圆的长轴和短轴 (inner_a, inner_b)。
        angle (float): 内椭圆的旋转角度（以弧度为单位，逆时针方向）。
        camera_frame (tuple): 相机取景框的范围 (x_min, x_max, y_min, y_max)。

    返回:
        tuple: 点的位置 (x, y)。
    """
    inner_a, inner_b = inner_axes
    x_min, x_max, y_min, y_max = camera_frame

    while True:
        # 随机生成一个点在相机取景框范围内
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        # 将点的坐标旋转到内椭圆的局部坐标系
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        local_x = cos_angle * x + sin_angle * y
        local_y = -sin_angle * x + cos_angle * y

        # 检查点是否在内椭圆外部
        if (local_x**2 / inner_a**2 + local_y**2 / inner_b**2) >= 1:
            return (x, y)

def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    csv_file = None,
    iter = 0,
    resolution = 1920,
    without_2D = False,
    overlook_only = False 
  ):
    clear_scene()
    current_time = datetime.now()
    file_name = current_time.strftime("%Y%m%d_%H%M%S")  # 格式化为 YYYYMMDD_HHMMSS
    twoD_output_path = os.path.join(render_output_path, file_name+"_2D.png")
    if 'blank' in background.lower():
      background = "./database/background_magnet_white.blend"
      load_blend_file_backgournd(background)

    blender = "./database/magnet/magnet.blend"
    needle = "./database/compass/compass_b.blend"
    random_rotation_angle = random.uniform(0, 360)
    
    # print(f"rotation_angle of magnet: {random_rotation_angle:.2f}")
    
    load_blend_file(blender, location=(0, 0, 0), scale=(1, 1, 1), rotation_angle=-random_rotation_angle)

    inner_radius = 14/2 # 圆环的内半径
    outer_radius = 13.2/2 + 1  # 圆环的外半径
    # needle_location = random_point_on_ring(inner_radius, outer_radius)
    needle_location = random_point_outside_rotated_inner_ellipse_within_camera((14.5, 3.5), angle=random_rotation_angle, camera_frame=(-7.5,7.5, -7.5, 7.5))
    print(needle_location)
    # needle_location = random_point_outside_rotated_rectangle((11.5, 11.5), angle=random_rotation_angle)
    
    # needle_location = random_point_outside_rotated_rectangle()
    visualize = False
    if not without_2D:
      visualize = True
      
    result = calculate_node_magnetic_info(
        magnet_center=(0, 0),
        magnet_direction=-random_rotation_angle,
        magnet_length=11.4,
        node_position=needle_location[:2],
        visualize=visualize,
        png_name = twoD_output_path
    )
   
    rotation_angle = result.angle_degrees
    noise_rotation_angle = result.noised_angle_degrees
    # print(f"rotation_angle of needle: {rotation_angle:.2f}")
    load_blend_file(filepath = needle, location = needle_location, 
                    scale=(1, 1, 1), rotation_angle = noise_rotation_angle)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    
    if not overlook_only:
      render_3D_output_path = os.path.join(render_output_path, file_name+f"_3D_{resolution}.png")
      set_render_parameters(output_path=render_3D_output_path, resolution=(resolution, resolution))
      fit_camera_to_objects_with_random_position(camera, ["needle", "Object_2"]) 
      render_scene()
      
    render_3D_over_output_path = os.path.join(render_output_path, file_name+f"_Over_3D_{resolution}p.png")
    # fit_camera_to_objects_with_random_position(camera, ["needle", "Object_2"], over=True) 
    fit_camera_to_objects_with_random_position(camera, [ "Object_2"], over=True) 
    object_name = "Object_2"
    needle_name = "needle"
    set_render_parameters(output_path=render_3D_over_output_path, resolution=(resolution, resolution))
    if check_objects_intersection(object_name, needle_name):
      pass
    else:
      # save_blend_file("./debug.blend")
      render_scene()
      
      # 将结果写入 CSV 文件
      with open(csv_file, mode="a", newline="") as file:
          writer = csv.writer(file)
          if without_2D and overlook_only:
            writer.writerow([iter, render_3D_over_output_path, -random_rotation_angle + 360, needle_location[0], needle_location[1], 
                             result.angle_degrees, result.noised_angle_degrees, result.noise])
          # elif without_2D:
          else:
            writer.writerow([iter, render_3D_output_path, render_3D_over_output_path, twoD_output_path, -random_rotation_angle, 
                             needle_location, result.field_direction, result.noisy_field_direction, result.noise])


def random_point_outside_rotated_inner_ellipse_within_camera(inner_axes, angle, camera_frame):
    """
    随机生成一个点，该点位于旋转内椭圆外部，同时位于相机取景框范围内。

    参数:
        inner_axes (tuple): 内椭圆的长轴和短轴 (inner_a, inner_b)。
        angle (float): 内椭圆的旋转角度（以角度为单位，逆时针方向）。
        camera_frame (tuple): 相机取景框的范围 (x_min, x_max, y_min, y_max)。

    返回:
        tuple: 点的位置 (x, y)。
    """
    inner_a, inner_b = inner_axes
    x_min, x_max, y_min, y_max = camera_frame

    # 将角度转换为弧度
    angle_rad = np.radians(angle)

    while True:
        # 随机生成一个点在相机取景框范围内
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        # 将点的坐标旋转到内椭圆的局部坐标系
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        local_x = cos_angle * x + sin_angle * y
        local_y = -sin_angle * x + cos_angle * y

        # 检查点是否在内椭圆外部
        if (local_x**2 / inner_a**2 + local_y**2 / inner_b**2) >= 1:
            return (x, y, -0.3)

def save_blend_file(filepath):
    """保存当前场景为指定的 .blend 文件，直接覆盖原有文件。"""
    if os.path.exists(filepath):
        print('remove the existing file')
        os.remove(filepath)  # 删除已有文件
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    # print(f"修改后的场景已保存到：{filepath}")

def render_scene():
    """执行渲染并保存图像。"""
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'  # 可选 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.ops.render.render(write_still=True)
    # print(f"渲染完成，图像已保存到：{bpy.context.scene.render.filepath}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--resolution", type=int, help="resolution of the image")
    parser.add_argument("--overlook_only", action="store_true", help="only render the overlook image")
    parser.add_argument("--without_2D", action="store_true", help="whether use 2D figure")
    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])

    iteration_time = 25  # 每次渲染的批次数量
    resolution = arguments.resolution

    # CSV 文件路径
    scene = "Magnetic"
    csv_file = f"./database/rendered_{scene.lower()}_{resolution}P/magnet_scene_{resolution}P.csv"
    
    try:
        with open(csv_file, mode="r") as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，写入 CSV 文件头
        if not file_exists and arguments.overlook_only:
            writer.writerow(["iter", "3D_over", "magnet_direction(degree)", "needle_location_x", "needle_location_y", 
                             "needle_direction(degree)", "noisy_needle_direction(degree)", "noise"])

        # 设置背景、场景和渲染输出路径
        background = "blank"

        render_output_path = f"./database/rendered_{scene.lower()}_{resolution}P/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in range(arguments.iter, arguments.iter + iteration_time):
            main(
                background=background,
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iter=i,
                resolution=resolution,
                without_2D = arguments.without_2D,
                overlook_only= arguments.overlook_only
            )
            object_name = "Object_2"
            obj = bpy.data.objects.get(object_name)
            needle_name = "needle"
            needle_object = bpy.data.objects.get(needle_name)

            if obj:
                # 确保对象有尺寸数据
                dimensions = obj.dimensions
                print(f"The size of {object_name} is:")
                print(f"X: {dimensions.x}, Y: {dimensions.y}, Z: {dimensions.z}")
            else:
                print(f"Object '{object_name}' not found in the scene.")
            if needle_object:
                # 确保对象有尺寸数据
                dimensions = needle_object.dimensions
                print(f"The size of {needle_name} is:")
                print(f"X: {dimensions.x}, Y: {dimensions.y}, Z: {dimensions.z}")
            else:
                print(f"Object '{needle_name}' not found in the scene.")
                

            if check_objects_intersection(object_name, needle_name):
                print(f"'{object_name}' 和 '{needle_name}' 相交了！")
                raise Exception(f"'{object_name}' 和 '{needle_name}' 相交了！")
            else:
                # print(f"'{object1}' 和 '{object2}' 没有相交。")
                pass
                                        
                
