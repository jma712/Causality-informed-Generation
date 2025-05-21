"""
1. load the 3D model
2. load bulb (object with random location)
3. load object
4. record the location of the bulb and the object
5. render the scene
"""

import bpy
import math

import numpy as np
from typing import List, Tuple, Union
import argparse
import sys
import random
from mathutils import Vector
import os

def setting_camera(location, target):
    """
    在场景中添加一个相机，并将其位置和指向设置为目标点。
    
    参数:
    location (tuple or list): 相机的位置坐标 (x, y, z)
    target (tuple or list): 相机指向的目标坐标 (x, y, z)
    """
    # 添加相机到场景
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "SceneCamera"

    # 将相机指向目标点
    direction = Vector(target) - Vector(location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # 设置场景相机为新创建的相机
    bpy.context.scene.camera = camera

# 示例使用
setting_camera(location=(10, -10, 5), target=(0, 0, 0))


def clear_scene():
    """删除当前场景中的所有对象。"""
    bpy.ops.object.select_all(action='SELECT')  # 选择所有对象
    bpy.ops.object.delete()  # 删除选中的对象
    print("清空场景完成。")

def fit_camera_to_objects_with_random_position(camera, object_names, margin=1.2):
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
            # obj.scale = scale
            # print(">>>>>>", rotation_angle)
            # obj.rotation_euler[2] = rotation_angle  # 直接设置 Z 轴的旋转角度
            # 选择对象并应用旋转
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

def generate_random_coordinates():
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(1, 1) 
    return x, y, z


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


# def import_object_from_blend(blend_file, object_name, location):
#     """
#     从指定的 .blend 文件中导入对象并设置位置。
    
#     参数:
#     blend_file (str): .blend 文件的路径。
#     object_name (str): 要导入的对象的名称。
#     location (tuple): 导入对象的位置坐标 (x, y, z)。
#     """
#     # 构建 .blend 文件中的对象路径
#     directory = os.path.join(blend_file, "Object")
#     filepath = os.path.join(directory, object_name)

#     # 导入对象
#     bpy.ops.wm.append(filepath=filepath, directory=directory, filename=object_name)

#     # 获取导入的对象并设置位置
#     imported_object = bpy.context.selected_objects[0]
#     imported_object.location = location


def import_object_from_blend(blend_file, object_name, location, scale=4.0, light_strength=300):
    """
    从指定的 .blend 文件中导入对象，设置位置和大小，并调节光源强度（如果是光源）。
    
    参数:
    blend_file (str): .blend 文件的路径。
    object_name (str): 要导入的对象的名称。
    location (tuple): 导入对象的位置坐标 (x, y, z)。
    scale (float): 调整对象大小的比例因子（默认值为 1.0，即不缩放）。
    light_strength (float): 如果对象是光源，设置其强度（默认为 None，即不调整）。
    """
    # 构建 .blend 文件中的对象路径
    directory = os.path.join(blend_file, "Object")
    filepath = os.path.join(directory, object_name)

    # 导入对象
    bpy.ops.wm.append(filepath=filepath, directory=directory, filename=object_name)

    # 获取导入的对象
    imported_object = bpy.context.selected_objects[0]
    
    # 设置对象的位置和大小
    imported_object.location = location
    imported_object.scale = (scale, scale, scale)

    # 如果对象是光源，调整其强度
    if imported_object.type == 'LIGHT' and light_strength is not None:
        imported_object.data.energy = light_strength
  
  
def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend"
  ):
    clear_scene()

    if 'blank' in background.lower():
      background = "./database/blank_background_1.blend"
      load_blend_file_backgournd(background)
      
    # 示例使用
    blend_file_path = "./database/bulb.blend"  # 替换为您的 .blend 文件路径
    object_name = "Lampada_Low"  # 替换为您要导入的对象名称
    location = (1, 2, -20)  # 指定位置

    import_object_from_blend(blend_file_path, object_name, location)
    # load_blend_file("./database/pendulum_1.blend", location=(0, 0, 10))#), scale=(1, 1, 1), rotation_angle=0)
    load_blend_file("./database/bulb.blend", location=(0, 0, 5008))#, scale=(1, 1, 1), rotation_angle=0)

    set_render_parameters(output_path=render_output_path)
    # load_blend_file_backgournd("./database/reflection_space.blend")


    bpy.ops.object.camera_add()
    camera = bpy.context.object
    fit_camera_to_objects_with_random_position(camera, [ "Lampada_Low"]) 
    # setting_camera(camera_location, target_location)   
    
    # camera_location = (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(0, 10))

    target_location = (0, 0, 1)
    # 4. 设置渲染参数
    # setting_camera(camera_location, target_location)

    render_scene()

    if save_path:
        save_blend_file(save_path)
    return 1
    # return (incident_point, reflection_point, camera_location, target_location)

  
  
if __name__ == "__main__":
  # 创建 ArgumentParser 对象
  parser = argparse.ArgumentParser(description="Blender Rendering Script")

  parser.add_argument("--background", type=str, help="背景文件路径")
  parser.add_argument("--scene", type=str, help="场景类型 (例如: Seesaw, Tennis, Magnetic)")
  parser.add_argument("--render_output_path", type=str, default="../database/rendered_image.png", help="渲染输出文件路径")
  parser.add_argument("--save_path", type=str, default="/Users/liu/Desktop/school_academy/Case/Yin/causal_project/Causality-informed-Generation/code1/database/temp.blend", help="保存场景文件路径")
  arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
  records = main(
      background=arguments.background,
      scene=arguments.scene,
      render_output_path=arguments.render_output_path,
      save_path=arguments.save_path
  ) 
  print(records)