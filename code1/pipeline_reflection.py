import bpy
import math
from datetime import datetime
import numpy as np
from typing import List, Tuple, Union
import argparse
import sys
import random
from mathutils import Vector
import os
import csv
sys.path.append("/home/ulab/.local/lib/python3.11/site-packages")  # 请根据实际路径确认
from tqdm import tqdm


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

def create_laser_beam(laser_length=10, name = "LaserBeam", color = (1.0, 0, 0, 1)):
    """
    创建一个激光场景，其中激光束根据给定的入射角发射并与平面相交。
    
    参数:
    angle (float): 激光的入射角（单位：度），相对于 Z 轴的角度。
    """
    # 设置渲染引擎为 Cycles，以便支持发光效果
    # bpy.context.scene.render.engine = 'CYCLES'

    # 激光束的起始位置和长度
    laser_start_height = 0  # 激光束的起始高度
    laser_length = laser_length  # 激光束的长度，足够长以确保与平面相交


    # 添加一个圆柱体来模拟激光束，使其指向平面
    bpy.ops.mesh.primitive_cylinder_add(radius=0.01, depth=laser_length, location=(0, 0, (laser_length/2)))
    laser_beam = bpy.context.object
    laser_beam.name = name

    # 创建发光材质
    laser_material = bpy.data.materials.new(name="LaserMaterial")
    laser_material.use_nodes = True
    laser_material.node_tree.nodes.clear()  # 清除默认节点

    # 添加发光节点
    emission_node = laser_material.node_tree.nodes.new(type="ShaderNodeEmission")
    emission_node.inputs['Strength'].default_value = 1000  # 调整发光强度，增加亮度
    emission_node.inputs['Color'].default_value = color#(1.0, 0, 1, 1)  # 设置为红色激光

    # 添加 Material Output 节点并连接发光节点
    material_output = laser_material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    laser_material.node_tree.links.new(emission_node.outputs['Emission'], material_output.inputs['Surface'])

    # 将发光材质赋予圆柱体
    laser_beam.data.materials.append(laser_material)

    # 添加光源来照亮场景
    # bpy.ops.object.light_add(type='POINT', location=(5, -5, 5))
    # light = bpy.context.object
    # light.name = "SceneLight"
    # light.data.energy = 50  # 设置光源亮度
    # 添加光源来照亮场景
    # bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))
    # light = bpy.context.object
    # light.name = "SceneLight"
    # light.data.energy = 100  # 设置光源亮度

    # 禁用光源的阴影
    # light.data.use_shadow = False  # 设置光源不产生阴影

    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    return laser_beam


def place_and_align_cylinder(cylinder, target_vector, length=2):
    """
    将圆柱体放置并旋转，使得一端在 (0, 0, 0)，另一端指向目标方向向量。
    
    参数:
    cylinder (Object): 要旋转的垂直圆柱体对象
    target_vector (list or tuple): 圆柱体指向的目标方向向量 [x, y, z]
    length (float): 圆柱体的长度
    """
    # 将目标方向向量转换为 numpy 数组并归一化
    target_vector = np.array(target_vector)
    target_vector = target_vector / np.linalg.norm(target_vector)

    # 默认垂直圆柱体的初始方向向量为 [0, 0, 1]
    initial_direction = np.array([0, 0, 1])

    # 计算旋转轴和旋转角度
    axis = np.cross(initial_direction, target_vector)
    axis_length = np.linalg.norm(axis)

    if axis_length == 0:
        # 如果轴向量长度为 0，说明方向已经一致，无需旋转
        angle = 0
        axis = [1, 0, 0]  # 任意轴都可以，因为角度为 0
    else:
        axis = axis / axis_length  # 归一化轴向量
        angle = math.acos(np.dot(initial_direction, target_vector))  # 旋转角度

    # 设置圆柱体的旋转模式为 Axis-Angle，以便精确指定旋转方向
    cylinder.rotation_mode = 'AXIS_ANGLE'
    cylinder.rotation_axis_angle[0] = angle  # 旋转角度
    cylinder.rotation_axis_angle[1] = axis[0]  # 旋转轴 x 分量
    cylinder.rotation_axis_angle[2] = axis[1]  # 旋转轴 y 分量
    cylinder.rotation_axis_angle[3] = axis[2]  # 旋转轴 z 分量

    # 调整圆柱体的长度
    cylinder.scale[2] = length / 1  # 使用缩放来调整长度

    # 将圆柱体的底部对齐到 (0, 0, 0)
    cylinder.location = target_vector * (length / 2)


def clear_scene():
    """删除当前场景中的所有对象。"""
    bpy.ops.object.select_all(action='SELECT')  # 选择所有对象
    bpy.ops.object.delete()  # 删除选中的对象
    print("清空场景完成。")


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
            
    bpy.ops.transform.rotate(
        value=math.radians(rotation_angle),
        orient_axis='Z',
        orient_type='GLOBAL',
        constraint_axis=(False, False, True)
      )
    
    
def load_blend_file_backgournd(filepath):
    """导入指定的 .blend 文件中的所有对象。"""
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects  # 选择导入所有对象
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

def set_render_parameters(resolution=(1920, 1080), file_format='PNG', 
                          output_path="../database/rendered_image.png",
                          circle = False):
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    if circle:
      bpy.context.scene.render.engine = 'CYCLES'
      bpy.context.scene.render.resolution_percentage = 50
      bpy.context.scene.render.resolution_x = 1920/2
      bpy.context.scene.render.resolution_y = 1080/2
      
      

def calculate_reflection_vector(incident_point):
    """
    根据入射点计算反射光的方向向量，假设反射平面为 x-y 平面。
    
    参数:
    incident_point (tuple): 入射光的起始坐标 (x, y, z)
    
    返回:
    numpy.ndarray: 反射光方向向量
    """
    # 将 incident_point 和 origin 转换为 numpy 数组
    incident_point = np.array(incident_point)
    origin = np.array([0, 0, 0])

    # 计算入射方向向量，并归一化
    incident_vector = origin - incident_point
    incident_vector = incident_vector / np.linalg.norm(incident_vector)

    # 定义 x-y 平面的法向量
    normal_vector = np.array([0, 0, 1])

    # 计算反射方向向量
    reflection_vector = incident_vector - 2 * np.dot(incident_vector, normal_vector) * normal_vector
    
    return reflection_vector
  
def draw_decreasing_probability_sample_once(low=0, high=10, scale=1.5):
    """
    随机抽取一个符合概率递减的数值，值越大，取到的概率越小。
    
    参数:
    - low: 取值范围的最小值
    - high: 取值范围的最大值
    - scale: 控制分布形状的参数
    
    返回:
    - 单个采样值
    """
    value = np.random.exponential(scale)
    value = np.clip(value, low, high)  # 将值截断在 [low, high] 范围内
    return value
  
def generate_random_coordinates():
    x = draw_decreasing_probability_sample_once()
    y = np.random.uniform(0, 0)
    z = np.random.uniform(1, 1) 
    return x, y, z


def render_scene():
    """执行渲染并保存图像。"""
    bpy.ops.render.render(write_still=True)
    print(f"渲染完成，图像已保存到：{bpy.context.scene.render.filepath}")

def save_blend_file(filepath):
    """保存当前场景为指定的 .blend 文件，直接覆盖原有文件。"""
    if os.path.exists(filepath):
        print('remove the existing file')
        os.remove(filepath)  # 删除已有文件
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"修改后的场景已保存到：{filepath}")
  
  
def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend",
    csv_file = None,
    iteration = 0,
    circle = False,
  ):
    clear_scene()
    current_time = datetime.now()
    file_name = current_time.strftime("%Y%m%d_%H%M%S")  # 格式化为 YYYYMMDD_HHMMSS
    file_name = os.path.join(render_output_path, file_name+".png")

    background = "./database/reflection_space.blend"
    load_blend_file_backgournd(background)


    set_render_parameters(output_path=file_name, circle = circle)
    incident_point = generate_random_coordinates()
    reflection_point = calculate_reflection_vector(incident_point)
    random_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1)  # 随机 RGB，A 设置为 1
    incident_beam = create_laser_beam(name = "IncidentBeam", color = random_color)
    reflect_beam = create_laser_beam(name = "ReflectBeam", color = random_color)
    place_and_align_cylinder(incident_beam, incident_point)
    place_and_align_cylinder(reflect_beam, reflection_point)
    camera_location = (random.uniform(-0, 0), random.uniform(10, 10), random.uniform(2, 2))

    target_location = (0, 0, 1)
    setting_camera(camera_location, target_location)

    render_scene()
    if save_path:
        save_blend_file("./temp.blend")
        

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iter, incident_point,  reflection_point, camera_location,
                         random_color, file_name])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument('--circle', action='store_true', help="A boolean flag argument")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])

    iteration_time = 45  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = "reflection_scene.csv"
    if arguments.circle:
      csv_file = "refleciton_scene_circle.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        init = True
        # 文件不存在，创建并写入表头
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", "incident_point", "reflection_point", "camera_location", "color", "images"])
    else:
        init = False

    try:
        with open(csv_file, mode="r") as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)


    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，写入 CSV 文件头
        if not file_exists:
            writer.writerow(["iter", "left_weight", "right_weight", "left_arm", "right_arm", "images"])

        # 设置背景、场景和渲染输出路径
        background = "./database/reflection_space.blend"
        scene = "Reflection"
        render_output_path = "./database/reflection_rendered_images/"
        if arguments.circle:
          render_output_path = './database/reflection_rendered_image_circle/'

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in tqdm(range(arguments.iter, arguments.iter + iteration_time), desc="Rendering"):
            main(
                background=background,
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                circle = arguments.circle
            )
