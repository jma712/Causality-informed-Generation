import bpy
import os
import sys
import subprocess
res = subprocess.run(["which", "python"],capture_output=True, text=True).stdout
import uuid

import argparse
import sys
import random
from mathutils import Vector
import math
import multiprocessing as mp
from functools import partial
# sys.path.append(res.split("bin")[0] + "lib/python3.12/site-packages/")
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.9/site-packages/")
# sys.path.append("/usr/lib/python3.12/site-packages/")
import numpy as np
from typing import Tuple, Union, Optional
from dataclasses import dataclass
from datetime import datetime
import csv

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

def calculate_node_magnetic_info(
    magnet_center: Tuple[float, float],  # 磁铁中心位置
    magnet_direction: Union[float, Tuple[float, float]],  # 可以是角度(弧度)或方向向量
    magnet_length: float,  # 磁铁长度
    node_position: Tuple[float, float],  # 待计算节点位置
    strength: float = 1.0,  # 磁场强度系数
    visualize: bool = True,  # 是否显示可视化
    iteration = None
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
    np.random.seed(iteration)
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
            
            
            # 重新计算加噪后的方向
            magnitude = np.linalg.norm(field)
            field_direction = field / magnitude
            angle = np.arctan2(field_direction[1], field_direction[0])


    result = NodeInfo(
        position=node,
        field_direction=field_direction,
        angle=angle,
        angle_degrees=np.degrees(angle) % 360  + np.random.randn() * 3.6,
        # noised_field_direction = noisy_field_direction,
        # noised_angle = noisy_angle,
        # noised_angle_degrees = np.degrees(noisy_angle) % 360,
        # noise = noise
    )
    
    return result

def clear_scene():
    """删除当前场景中的所有对象。"""
    bpy.ops.object.select_all(action='SELECT')  # 选择所有对象
    bpy.ops.object.delete()  # 删除选中的对象

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
    """导入指定的 .blend 文件中的所有对象，并为名为 'wooden_floor' 的对象添加材质贴图。"""
    # 导入所有对象
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects  # 导入所有对象
    
    # 链接对象到当前场景
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

    # 为名为 wooden_floor 的对象添加材质贴图
    wooden_floor_obj = bpy.data.objects.get("wooden_floor")
    if wooden_floor_obj:
        # 创建一个新的材质
        material = bpy.data.materials.new(name="Wooden_Floor_Material")
        material.use_nodes = True
        
        # 获取材质节点
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf_node = nodes.get("Principled BSDF")

        # 添加 Base Color 贴图
        base_color_path = "/home/lds/github/Causality-informed-Generation/code1/Wood066_1K-JPG/Wood066_1K-JPG_Color.jpg"  # 替换为实际路径
        if base_color_path:
            tex_image_color = nodes.new("ShaderNodeTexImage")
            tex_image_color.image = bpy.data.images.load(base_color_path)
            links.new(bsdf_node.inputs["Base Color"], tex_image_color.outputs["Color"])
        
        # 添加 Roughness 贴图
        roughness_path = "/home/lds/github/Causality-informed-Generation/code1/Wood066_1K-JPG/Wood066_1K-JPG_Roughness.jpg"  # 替换为实际路径
        if roughness_path:
            tex_image_roughness = nodes.new("ShaderNodeTexImage")
            tex_image_roughness.image = bpy.data.images.load(roughness_path)
            links.new(bsdf_node.inputs["Roughness"], tex_image_roughness.outputs["Color"])
        
        # 添加 Normal Map 贴图
        normal_map_path = "/home/lds/github/Causality-informed-Generation/code1/Wood066_1K-JPG/Wood066_1K-JPG_NormalGL.jpg"  # 替换为实际路径
        if normal_map_path:
            tex_image_normal = nodes.new("ShaderNodeTexImage")
            tex_image_normal.image = bpy.data.images.load(normal_map_path)
            normal_map_node = nodes.new("ShaderNodeNormalMap")
            links.new(normal_map_node.inputs["Color"], tex_image_normal.outputs["Color"])
            links.new(bsdf_node.inputs["Normal"], normal_map_node.outputs["Normal"])

        # 添加 Displacement 贴图
        displacement_path = "/home/lds/github/Causality-informed-Generation/code1/Wood066_1K-JPG/Wood066_1K-JPG_Displacement.jpg"  # 替换为实际路径
        if displacement_path:
            tex_image_displacement = nodes.new("ShaderNodeTexImage")
            tex_image_displacement.image = bpy.data.images.load(displacement_path)
            displacement_node = nodes.new("ShaderNodeDisplacement")
            links.new(displacement_node.inputs["Height"], tex_image_displacement.outputs["Color"])
            material_output_node = nodes.get("Material Output")
            links.new(material_output_node.inputs["Displacement"], displacement_node.outputs["Displacement"])

        # 将材质赋予 wooden_floor 对象
        if wooden_floor_obj.data.materials:
            wooden_floor_obj.data.materials[0] = material
        else:
            wooden_floor_obj.data.materials.append(material)

        print(f"Material successfully applied to '{wooden_floor_obj.name}'.")
    else:
        print("Object 'wooden_floor' not found in the imported scene.")

def set_render_parameters(resolution=(1920, 1080), file_format='PNG', output_path="../database/rendered_image.png", samples=500, use_denoising=True, use_transparent_bg=False):
    """设置渲染参数，包括分辨率、格式、输出路径和高质量渲染设置。"""
    # 设置分辨率和输出路径
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    
    bpy.context.scene.render.engine = 'CYCLES'
    # bpy.context.scene.render.resolution_percentage = 60
    bpy.context.scene.cycles.samples = 2800  #渲染时的采样数

    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
    
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
    radius = np.random.uniform(inner_radius, outer_radius)
    # 随机生成一个角度
    angle = np.random.uniform(0, 2 * np.pi)
    
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
        x = np.random.uniform(-image_width / 2, image_width / 2)
        y = np.random.uniform(-image_height / 2, image_height / 2)

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

def xx(location, target_location, camera_name="Camera"):
    """
    添加一个相机到指定位置，并将其指向目标位置，同时设置相机名称。

    :param location: tuple(float, float, float), 相机的位置 (x, y, z)。
    :param target_location: tuple(float, float, float), 相机目标的位置 (x, y, z)。
    :param camera_name: str, 相机的名称。
    """
    # 创建相机对象
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object  # 获取刚添加的相机对象
    camera.name = camera_name
    camera.data.lens = 20

    # 创建一个空对象作为目标
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=target_location)
    target = bpy.context.object
    target.name = f"{camera_name}_Target"

    # 添加跟踪约束，让相机指向目标
    track_constraint = camera.constraints.new(type='TRACK_TO')
    track_constraint.target = target
    track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    track_constraint.up_axis = 'UP_Y'

    print(f"Camera '{camera_name}' added at {location}, pointing to {target_location}.")

def delete_object_by_name(object_name):
    """
    删除指定名称的对象。
    
    :param object_name: str, 要删除的对象名称。
    """
    obj = bpy.data.objects.get(object_name)  # 获取指定名称的对象
    if obj:
        # 将对象从场景中取消链接
        bpy.data.objects.remove(obj, do_unlink=True)
        print(f"Object '{object_name}' has been deleted.")
    else:
        print(f"Object '{object_name}' not found.")


def random_point_outside_rotated_inner_ellipse_within_camera(inner_axes, 
                                                             angle, 
                                                             camera_frame,
                                                             iteration):
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
    np.random.seed(iteration)
    while True:
        # 随机生成一个点在相机取景框范围内
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        # 将点的坐标旋转到内椭圆的局部坐标系
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        local_x = cos_angle * x + sin_angle * y
        local_y = -sin_angle * x + cos_angle * y

        # 检查点是否在内椭圆外部
        if (local_x**2 / inner_a**2 + local_y**2 / inner_b**2) >= 1:
            return (x, y)

def add_hdr_environment(hdr_path, strength=1.0, rotation_z=0.0):
    """
    在 Blender 场景中添加 HDR 环境贴图。
    
    参数:
        hdr_path (str): HDR 图像文件的路径。
        strength (float): 环境光的强度 (默认值: 1.0)。
        rotation_z (float): HDR 贴图在 Z 轴上的旋转角度（弧度，默认值: 0.0）。
    """
    # 获取当前场景的 World
    world = bpy.context.scene.world

    # 如果场景没有 World，则创建一个
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    # 启用节点
    world.use_nodes = True
    nodes = world.node_tree.nodes

    # 清除现有节点
    for node in nodes:
        nodes.remove(node)

    # 添加背景节点
    background_node = nodes.new(type="ShaderNodeBackground")
    background_node.location = (0, 0)

    # 添加环境纹理节点
    env_texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    env_texture_node.location = (-300, 0)
    try:
        env_texture_node.image = bpy.data.images.load(hdr_path)  # 加载 HDR 图像
    except:
        print(f"无法加载 HDR 文件: {hdr_path}")
        return

    # 添加输出节点
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    output_node.location = (200, 0)

    # 连接节点
    links = world.node_tree.links
    links.new(env_texture_node.outputs["Color"], background_node.inputs["Color"])
    links.new(background_node.outputs["Background"], output_node.inputs["Surface"])

    # 设置 HDRI 的强度
    background_node.inputs["Strength"].default_value = strength

    # 添加贴图坐标节点和映射节点（用于旋转）
    texture_coord_node = nodes.new(type="ShaderNodeTexCoord")
    texture_coord_node.location = (-600, 0)

    mapping_node = nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-450, 0)

    # 连接贴图坐标节点和映射节点
    links.new(texture_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], env_texture_node.inputs["Vector"])

    # 设置旋转值（仅旋转 Z 轴）
    mapping_node.inputs["Rotation"].default_value[2] = rotation_z

    print(f"HDR 环境贴图已成功添加: {hdr_path}")



def main(
    render_output_path = "../database/rendered_image.png",
    csv_file = None,
    iteration = 0,
    resolution = 1920,
    without_2D = False,
    overlook_only = False,
    realistic = False,
    black_floor = False
  ):
    clear_scene()
    np.random.seed(iteration)
    file_name = f'{iteration}'
    if realistic:
      load_blend_file_backgournd("/home/lds/github/Causality-informed-Generation/code1/database/3d_scenes/background_magnet_realiscit.blend")
      # raise ValueError("Not implemented")
      add_hdr_environment("/home/lds/github/Causality-informed-Generation/code1/database/3d_scenes/environment/machine_shop_02_2k.hdr")
    elif black_floor:
      load_blend_file_backgournd("./database/background_magnet_blank_floor.blend")
    else:
      load_blend_file_backgournd("./database/background_magnet_wooden_floor.blend")

    blender = "./database/magnet/magnet.blend"
    needle = "./database/compass/compass_b.blend"
    random_rotation_angle = np.random.uniform(0, 360)
    
    load_blend_file(blender, location=(0, 0, 0), scale=(1, 1, 1), rotation_angle=-random_rotation_angle)

    inner_radius = 14/2 # 圆环的内半径
    outer_radius = 13.2/2 + 1  # 圆环的外半径
    # needle_location = random_point_on_ring(inner_radius, outer_radius)
    needle_location = random_point_outside_rotated_inner_ellipse_within_camera(iteration = iteration, 
                                                                               inner_axes = (14.5, 3.5), 
                                                                               angle=random_rotation_angle, 
                                                                               camera_frame=(-7.5,7.5, -7.5, 7.5))
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
        iteration = iteration
    )

    rotation_angle = result.angle_degrees
    load_blend_file(filepath = needle, location = needle_location, 
                    scale=(1, 1, 1), rotation_angle = rotation_angle)
   

    def set_active_camera(camera_name):
      camera = bpy.data.objects.get(camera_name)
      if camera and camera.type == 'CAMERA':
          bpy.context.scene.camera = camera
          bpy.context.view_layer.update()  # 强制更新
          print(f"Camera '{camera_name}' is now set as the active render camera.")
      else:
          print(f"Camera '{camera_name}' not found or is not a valid camera.")

    # 示例：设置名为 "MyCamera" 的相机为渲染相机


    # if not overlook_only:
    #   render_3D_output_path = os.path.join(render_output_path, file_name+f"_3D_{resolution}.png")
    #   set_render_parameters(output_path=render_3D_output_path, resolution=(resolution, resolution))
    #   fit_camera_to_objects_with_random_position(camera, ["needle", "Object_2"]) 
    #   render_scene()
      
    render_3D_over_output_path = os.path.join(render_output_path, file_name+".png")
    object_name = "Object_2"
    needle_name = "needle"
    
    if check_objects_intersection(object_name, needle_name):
      pass
    else:
      for i in range(9):
        if i == 0:
          bpy.ops.object.camera_add()
          camera = bpy.context.object
          camera.name = "Camera"
          
          fit_camera_to_objects_with_random_position(camera, ["needle", "Object_2"], over=True) 
          fit_camera_to_objects_with_random_position(camera, [ "Object_2"], over=True) 
          camere_location = camera.location
        else:
         
          if i == 1:
            camera_position = (0, -8, 10)  # 相机位于 x=0, y=-5, z=5
          elif i == 2:
            camera_position = (0, 8, 10)
          elif i == 3:
            camera_position = (8, 0, 10)
          elif i == 4:
            camera_position = (-8, 0, 10)
          elif i == 5:
            camera_position = (8, 8, 10)
          elif i == 6:
            camera_position = (-8, -8, 10)
          elif i == 7:
            camera_position = (8, -10, 5)
          else:
            camera_position = (-8, 8, 10)
          target_position = (0, 0, 0)   # 目标位于原点

          # 添加相机并设置
          add_camera(location=camera_position, target_location=target_position, camera_name="MyCamera") 

          set_active_camera("MyCamera")
          camere_location = camera_position
        render_3D_over_output_path = os.path.join(render_output_path, file_name+f"_{i}.png")
        set_render_parameters(output_path=render_3D_over_output_path, 
                              resolution=(resolution, resolution))
        render_scene()
        
        if i == 0:
          delete_object_by_name("Camera")
        else:
          delete_object_by_name("MyCamera")

        
    # save_blend_file("./magent_3D_wooden_floor_DEBUG.blend")

        # 将结果写入 CSV 文件
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if without_2D and overlook_only:
              writer.writerow([iteration, os.path.basename(render_3D_over_output_path),
                                camere_location,
                                -random_rotation_angle + 360, 
                                needle_location[0], needle_location[1], 
                              result.angle_degrees])
            else:
              writer.writerow([iteration, render_3D_output_path, render_3D_over_output_path, twoD_output_path, -random_rotation_angle, 
                              needle_location, result.field_direction, result.noisy_field_direction, result.noise])

def random_point_outside_rotated_inner_ellipse_within_camera(iteration, inner_axes, angle, camera_frame):
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
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

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


def process_task(i, render_output_path, csv_file, resolution, without_2D, overlook_only):
    """
    Function to process a single iteration.
    """
    np.random.seed(i)
    main(
        render_output_path=render_output_path,
        csv_file=csv_file,
        iteration=i,
        resolution=resolution,
        without_2D=without_2D,
        overlook_only=overlook_only
    )

# Multi-threaded execution
def run_in_parallel(arguments, render_output_path, csv_file, resolution):
    # Define the range of iterations
    start = 0# arguments.iter
    end = 50# arguments.iter + iteration_time
    print(f"Processing iterations from {start} to {end}")

    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust `max_workers` as needed
        futures = [
            executor.submit(
                process_task, i, render_output_path, csv_file, resolution, 
                arguments.without_2D, arguments.overlook_only
            )
            for i in range(start, end)
        ]
        for future in futures:
            try:
                future.result()  # Wait for each thread to complete
            except Exception as e:
                print(f"An error occurred: {e}")


import multiprocessing as mp
from functools import partial
import numpy as np

def process_task(i, render_output_path, csv_file, resolution, without_2D, overlook_only, realistic=False, black_floor = False):
    """单个进程要执行的任务"""
    np.random.seed(i)
    main(
        render_output_path=render_output_path,
        csv_file=csv_file,
        iteration=i,
        resolution=resolution,
        without_2D=without_2D,
        overlook_only=overlook_only,
        realistic=realistic,
        black_floor=black_floor
    )

def run_multiprocess(arguments, iteration_time, render_output_path, csv_file, resolution):
    """多进程执行主函数"""
    # print(arguments.iter, arguments.iter + iteration_time)
    
    # 创建进程池，使用CPU核心数量的进程
    num_processes = 5# mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    realistic = arguments.realistic
    
    # 准备任务参数
    task_range = range(arguments.iter, arguments.iter + iteration_time)
    
    # 使用偏函数固定其他参数
    process_func = partial(
        process_task,
        render_output_path=render_output_path,
        csv_file=csv_file,
        resolution=resolution,
        without_2D=arguments.without_2D,
        overlook_only=arguments.overlook_only,
        realistic=realistic,
        black_floor = arguments.black_floor
    )
    
    # 使用进程池执行任务
    try:
        pool.map(process_func, task_range)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")
    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--resolution", type=int, help="resolution of the image")
    parser.add_argument("--overlook_only", action="store_true", help="only render the overlook image")
    parser.add_argument("--without_2D", action="store_true", help="whether use 2D figure")
    parser.add_argument("--realistic", action="store_true", help="using realistic scene")
    parser.add_argument("--black_floor", action="store_true", help="using black floor")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    

    iteration_time = 10  # 每次渲染的批次数量
    resolution = arguments.resolution
    realisitc = arguments.realistic

    # CSV 文件路径
    scene = "Magnetic"

    csv_file = f"./database/Real_magnet_v3/tabular.csv"
    if realisitc:
        csv_file = f"./database/Real_magnet_v3_realistic_back/tabular.csv"

    elif arguments.black_floor:
      csv_file = f"./database/Real_magnet_v3_blank_floor/tabular.csv"
      # raise ValueError("Not implemented")
    try:
        with open(csv_file, mode="r") as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Ensure the directory exists
    directory = os.path.dirname(csv_file)
    os.makedirs(directory, exist_ok=True)

    # Open the CSV file in append mode and write headers if needed
    if not file_exists and arguments.overlook_only:
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            # Write headers only if the file does not exist
            writer.writerow([
                "iter", "3D_over", 'camera_location',  "magnet_direction(degree)", 
                "needle_location_x", "needle_location_y", 
                "needle_direction(degree)"
            ])


    render_output_path = directory

    # 使用起始帧数循环渲染 iteration_time 个批次
    print(f"Rendering {iteration_time} batches of {resolution}P images...")
    print(arguments.iter, arguments.iter + iteration_time)
    


    run_multiprocess(arguments, iteration_time, render_output_path, csv_file, resolution)
    # for i in range(arguments.iter, arguments.iter + iteration_time):
    #     np.random.seed(i)
    #     main(
    #         render_output_path=render_output_path,
    #         csv_file=csv_file,
    #         iter=i,
    #         resolution=resolution,
    #         without_2D = arguments.without_2D,
    #         overlook_only= arguments.overlook_only
    #     )                 
              
