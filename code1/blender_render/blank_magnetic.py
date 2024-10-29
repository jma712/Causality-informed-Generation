import bpy
import os
import argparse
import sys
import random
from mathutils import Vector
import math
import numpy as np


def clear_scene():
    """删除当前场景中的所有对象。"""
    bpy.ops.object.select_all(action='SELECT')  # 选择所有对象
    bpy.ops.object.delete()  # 删除选中的对象
    print("清空场景完成。")


def setting_camera(location, target, scene_bounds=((-30, 30), (-30, 30), (0, 30))):
    """
    This function sets the camera location and target.
    The camera's position should be within the range defined by the scene bounds.
    
    Parameters:
    - location: tuple (x, y, z) representing the desired camera position.
    - target: tuple (x, y, z) representing the target point the camera should point at.
    - scene_bounds: tuple of tuples ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                    defining the allowable range for camera positioning.
    """
    
    # Unpack scene bounds
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = scene_bounds

    # Clamp the camera location within the scene bounds
    clamped_location = (
        max(xmin, min(xmax, location[0])),
        max(ymin, min(ymax, location[1])),
        max(zmin, min(zmax, location[2]))
    )

    # 删除已有的摄像机
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
        bpy.data.objects.remove(camera, do_unlink=True)
        print("Deleted existing camera")

    # 创建新的摄像机
    bpy.ops.object.camera_add(location=clamped_location)
    camera = bpy.context.active_object
    camera.name = "Camera"
    print("Created new camera")

    # 设置摄像机朝向目标位置
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # 将摄像机设置为当前场景的活动摄像机
    bpy.context.scene.camera = camera
    print(f"Camera location set to {camera.location}, pointing towards {target}")
    
    
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
            # obj.scale = scale
            
            # 选择对象并应用旋转
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            
            # 使用 bpy.ops.transform.rotate 在 Z 轴上旋转
            bpy.ops.transform.rotate(
                value=rotation_angle,
                orient_axis='Z',
                orient_type='GLOBAL',
                constraint_axis=(False, False, True)
            )
    
    print("场景已导入成功！")
    
def load_blend_file_backgournd(filepath):
    """导入指定的 .blend 文件中的所有对象。"""
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects  # 选择导入所有对象
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
    print("场景已导入成功！")

def set_render_parameters(resolution=(1920, 1080), file_format='PNG', output_path="../database/rendered_image.png"):
    """设置渲染参数，包括分辨率、格式和输出路径。"""
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    print("渲染参数已设置。")
 

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

def align_needle_direction(needle_mesh_name, needle_direction_xy):
    """
    调整磁针的方向，使其与指定方向对齐。
    
    参数:
    needle_mesh_name : str
        磁针 mesh 的名称。
    needle_direction_xy : np.array
        目标方向的单位向量，限制在 x-y 平面上。
    """
    # 获取磁针对象
    needle = bpy.data.objects[needle_mesh_name]
    
    # 确保目标方向在 x-y 平面内，并将其归一化
    needle_direction_xy[2] = 0  # 保持 z 分量为 0
    needle_direction_xy = needle_direction_xy / np.linalg.norm(needle_direction_xy)  # 归一化
    
    # 计算目标方向与 x 轴之间的夹角
    angle = np.arctan2(needle_direction_xy[1], needle_direction_xy[0])  # 返回弧度
    
    # 设置磁针的旋转，使其指向目标方向
    # 假设磁针初始方向是沿 x 轴
    needle.rotation_euler = (0, 0, angle)  # z 轴上的旋转
    
# 示例用法
def random_point_on_ring(inner_radius, outer_radius):
    """
    在给定的圆环内随机生成一个点。
    
    参数:
    inner_radius : float
        圆环的内半径。
    outer_radius : float
        圆环的外半径。
        
    返回:
    point : tuple
        随机生成的点 (x, y, z)。
    """
    # 随机生成一个半径，位于内外半径之间
    radius = random.uniform(inner_radius, outer_radius)
    print("radius:", radius)
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
    save_path = "../database/modified_scene.blend"
  ):
    clear_scene()
    # 使用模块化的函数执行完整流程
    if 'blank' in background.lower():
      background = "./database/blank_background.blend"
      load_blend_file_backgournd(background)


    blender = "./database/magnet.blend"
    needle = "./database/compass.blend"
    random_rotation_angle = 0#random.uniform(-180, 180)
    load_blend_file(blender, location=(0, 0, 0), scale=(3, 3, 3), rotation_angle=random_rotation_angle)

    inner_radius = 1.3  # 圆环的内半径
    outer_radius = 3  # 圆环的外半径
    needle_location = random_point_on_ring(inner_radius, outer_radius)
    rotation_axis = np.array([0.0, 0.0, 1.0])  # 假设旋转轴为 z 轴
    rotation_angle = random_rotation_angle  # 旋转角度，45度
    magnitude = 1.0  # 磁矩的强度
    # 计算旋转后的磁矩向量
    print("random_rotation_angle:", random_rotation_angle)
    rotated_m = calculate_rotated_magnetic_moment(rotation_axis, rotation_angle, magnitude)
    print("旋转后的磁矩向量：", rotated_m)
    
    magnet_position = np.array([0.0, 0.0, 0.0])  # 条形磁铁的位置
    needle_position = needle_location  # 磁针的位置在 x-y 平面上
    load_blend_file(filepath = needle, location = needle_position)
    print("磁针的位置：", needle_position)
    needle_dir_xy = magnetic_needle_direction_xy_plane(rotated_m, magnet_position, needle_position)
    print("磁针在 x-y 平面的方向：", needle_dir_xy)    
    # align_needle_direction("Object_2", needle_dir_xy)
    
    set_render_parameters(output_path=render_output_path)

    target_location = (0, 0, 1)
    range_min = -15
    range_max = 15
    range_max_z = 10
    range_min_z = 3
    camera_location = (random.uniform(range_min, range_max), random.uniform(range_min, range_max), random.uniform(range_min_z, range_max_z))
    setting_camera(camera_location, target_location)
    render_scene()

    if save_path:
        save_blend_file(save_path)

def save_blend_file(filepath):
    """保存当前场景为指定的 .blend 文件，直接覆盖原有文件。"""
    if os.path.exists(filepath):
        print('remove the existing file')
        os.remove(filepath)  # 删除已有文件
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"修改后的场景已保存到：{filepath}")

def render_scene():
    """执行渲染并保存图像。"""
    bpy.ops.render.render(write_still=True)
    print(f"渲染完成，图像已保存到：{bpy.context.scene.render.filepath}")

if __name__ == "__main__":
  # 创建 ArgumentParser 对象
  parser = argparse.ArgumentParser(description="Blender Rendering Script")

  parser.add_argument("--background", type=str, help="背景文件路径")
  parser.add_argument("--scene", type=str, help="场景类型 (例如: Seesaw, Tennis, Magnetic)")
  parser.add_argument("--render_output_path", type=str, default="../database/rendered_image.png", help="渲染输出文件路径")
  parser.add_argument("--save_path", type=str, default="/Users/liu/Desktop/school_academy/Case/Yin/causal_project/Causality-informed-Generation/code1/database/temp.blend", help="保存场景文件路径")
  # 解析命令行参数
  arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
  # arguments = parser.parse_args()
  # 将解析的参数传递给 main 函数
  # clear_scene()
  main(
      background=arguments.background,
      scene=arguments.scene,
      render_output_path=arguments.render_output_path,
      save_path=arguments.save_path
  ) 