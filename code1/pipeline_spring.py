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


material_density = {
    "Water": 1.0,                # 水
    "Air": 0.0012,               # 空气
    "Iron": 7.87,                # 铁
    "Gold": 19.32,               # 金
    "Silver": 10.49,             # 银
    "Copper": 8.96,              # 铜
    "Aluminum": 2.70,            # 铝
    "Lead": 11.34,               # 铅
    "Glass": 2.5,                # 玻璃
    "Wood": 0.6,                 # 木材（平均值）
    "Concrete": 2.4,             # 混凝土
    "Oil": 0.92,                 # 油
    "Mercury": 13.6,             # 水银
    "Platinum": 21.45,           # 铂
    "Diamond": 3.51,             # 钻石
    "Ice": 0.92,                 # 冰
    "Rubber": 1.1,               # 橡胶
    "Steel": 7.85,               # 钢
    "Titanium": 4.51,            # 钛
    "Uranium": 18.95             # 铀
}

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
    bpy.context.scene.render.resolution_percentage = 250
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format

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

def create_cube_based_on_weight(weight, density):
    """
    根据输入的 weight 和物质的密度创建一个立方体，并返回其长、宽、高。
    
    参数:
        weight (float): 立方体的重量。
        density (float): 物质的密度（单位：质量/体积）。
    
    返回:
        tuple: (length, width, height) 立方体的长、宽、高。
    """
    if weight <= 0:
        raise ValueError("Weight must be a positive number.")
    if density <= 0:
        raise ValueError("Density must be a positive number.")
    
    # 计算立方体的体积
    volume = weight / density
    
    # 计算立方体的边长（假设长宽高相等）
    side_length = math.pow(volume, 1/3)
    
    # 在场景中创建一个立方体
    bpy.ops.mesh.primitive_cube_add(size=side_length, location=(0, 0, 0))
    cube = bpy.context.object
    
    # 命名立方体
    cube.name = "Weight_Cube"
    
    # 返回立方体的长、宽、高
    dimensions = cube.dimensions
    return dimensions.x, dimensions.y, dimensions.z, cube
  
def move_object_to_location(object_name, location):
    """
    将指定的对象移动到指定的位置。
    
    参数:
        object_name (str): 需要移动的对象的名称。
        location (tuple): 目标位置 (x, y, z)，表示对象要移动到的坐标。
    
    返回:
        None
    """
    # 获取对象
    obj = bpy.data.objects.get(object_name)
    
    if obj is None:
        raise ValueError(f"Object '{object_name}' not found in the scene.")
    
    # 移动对象到指定位置
    obj.location = location
    
    print(f"Object '{object_name}' moved to location {location}.")

def calculate_spring_deformation(weight, spring_constant, max_deformation):
    """
    根据输入的重量计算弹簧的形变量，同时限制形变量不能超过弹簧的最大形变量。
    
    参数:
        weight (float): 施加在弹簧上的重量（单位：N 或 kg*g）。
        spring_constant (float): 弹簧的劲度系数（单位：N/m）。
        max_deformation (float): 弹簧的最大允许形变量（单位：m）。
        
    返回:
        float: 计算后的形变量（单位：m），不会超过最大形变量。
    """
    if spring_constant <= 0:
        raise ValueError("Spring constant must be a positive number.")
    if max_deformation <= 0:
        raise ValueError("Max deformation must be a positive number.")
    
    # 计算形变量
    deformation = weight / spring_constant  # x = F / k
    noise = random.uniform(-0.05*deformation, 0.05*deformation)
    deformation += noise
    
    # 限制形变量在最大允许范围内
    if deformation > max_deformation:
        deformation = max_deformation
        print("Warning: Deformation exceeded maximum limit. Limiting to max deformation.")
    
    return deformation, noise

def resize_object_on_z_axis(object_name, scale_factor):
    """
    根据对象名称选择对象，并对其在 Z 轴上进行缩放。
    
    参数:
        object_name (str): 需要缩放的对象的名称。
        scale_factor (float): Z 轴的缩放因子。
    """
    # 获取对象
    obj = bpy.data.objects.get(object_name)
    
    if obj is None:
        raise ValueError(f"Object '{object_name}' not found in the scene.")
    
    # 确保对象被选中
    bpy.ops.object.select_all(action='DESELECT')  # 取消选择所有对象
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj  # 设置为活动对象

    # 调整对象的 Z 轴缩放
    # obj.scale[2] *= scale_factor
    bpy.ops.transform.resize(value=(1, 1, scale_factor), orient_type='GLOBAL', 
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                             orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), 
                             mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', 
                             proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, 
                             snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', 
                             use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
        
      
    
    print(f"Object '{object_name}' resized on Z axis by a factor of {scale_factor}.")

def disable_shadows_for_render():
    """
    禁用场景中所有对象和灯光的阴影效果，确保渲染出的图像没有影子。
    """
    # 禁用所有灯光的阴影投射
    for light in bpy.data.lights:
        light.use_shadow = False
    
    # 禁用所有对象接收阴影
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.cycles.is_shadow_catcher = False  # 禁用Shadow Catcher
            obj.cycles.cast_shadow = False        # 禁用投射阴影
            obj.cycles.use_receive_shadows = False  # 禁用接收阴影
    
    print("Shadows have been disabled for rendering.")


def apply_pbr_material(obj, texture_dir, texture_files):
    """
    Apply a PBR material to an object using provided texture files.

    Parameters:
        obj (bpy.types.Object): The Blender object to apply the material to.
        texture_dir (str): Directory containing texture files.
        texture_files (dict): Dictionary mapping texture types to filenames.
    """
    bpy.ops.object.shade_smooth()

    # Create a new material
    mat = bpy.data.materials.new(name="PBRMaterial")
    mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    # Create essential nodes
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture_coord_node = nodes.new(type="ShaderNodeTexCoord")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    normal_map_node = nodes.new(type="ShaderNodeNormalMap")

    # Position nodes for clarity
    bsdf.location = (400, 300)
    material_output.location = (700, 300)
    texture_coord_node.location = (-600, 300)
    mapping_node.location = (-400, 300)
    normal_map_node.location = (200, -200)

    # Connect UV mapping
    links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])

    # Function to load texture and handle errors
    def load_texture(texture_type, input_socket, is_data=False):
        if texture_type in texture_files:
            texture_path = os.path.join(texture_dir, texture_files[texture_type])
            if os.path.exists(texture_path):
                texture_node = nodes.new(type="ShaderNodeTexImage")
                texture_node.location = (-200, 300 - len(nodes) * 100)
                texture_node.image = bpy.data.images.load(texture_path)
                texture_node.image.colorspace_settings.is_data = is_data
                links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
                links.new(texture_node.outputs['Color'], input_socket)
                return texture_node
            else:
                print(f"Texture not found: {texture_path}")
        else:
            print(f"Texture type '{texture_type}' not provided.")
        return None

    # Load textures
    load_texture("Base Color", bsdf.inputs['Base Color'], is_data=False)
    load_texture("Metalness", bsdf.inputs['Metallic'], is_data=True)
    load_texture("Roughness", bsdf.inputs['Roughness'], is_data=True)
    ao_node = load_texture("Ambient Occlusion", bsdf.inputs['Base Color'], is_data=True)
    normal_texture_node = load_texture("Normal", normal_map_node.inputs['Color'], is_data=True)
    load_texture("Displacement", None, is_data=True)  # Note: Connect to displacement later if needed

    # Handle normal map connection
    if normal_texture_node:
        links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])

    # Connect BSDF to material output
    links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # Assign material to the object
    obj.data.materials.append(mat)


def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend",
    csv_file = None,
    iter = 0,
    resolution = None,
    circle = False,
  ):
    clear_scene()
    current_time = datetime.now()
    file_name = current_time.strftime("%Y%m%d_%H%M%S")  # 格式化为 YYYYMMDD_HHMMSS
    file_path = os.path.join(render_output_path, file_name+".png")


    background = "./database/blank_background_spring.blend"
    load_blend_file_backgournd(background)

    set_render_parameters(output_path=file_path, resolution=(resolution, resolution))
    camera_location = (random.uniform(-0, 0), random.uniform(15, 15), random.uniform(1, 1))
    load_blend_file("./database/Spring.blend")
    
    
    materials = ["Iron", "Wood"]
    material = random.choice(materials)
    if material == "Iron":
      weight = random.uniform(2, 10)
    elif material == "Wood":
      weight = random.uniform(0.5, 10)
    x,y,z, cube = create_cube_based_on_weight(weight=weight, density=material_density[material])

    spring_constant = 10  # 弹簧劲度系数 (N/m)

    high = 1.3
    max_deformation = high * 0.83
    deformation, noise = calculate_spring_deformation(weight, spring_constant, max_deformation)
    
    spring = bpy.data.objects.get("spring")
    scale_factor = (high - deformation) /  high
    resize_object_on_z_axis("spring", scale_factor)
    move_object_to_location("Weight_Cube", (0, 0, high*scale_factor+z/2))
    

    if material == "Iron":
      apply_pbr_material(
          obj=cube, 
          texture_dir="./database/material/Metal055A_1K-JPG/",  # 替换为实际路径
          texture_files={
              'Base Color': 'Metal055A_1K-JPG_Color.jpg',
              'Metalness': 'Metal055A_1K-JPG_Metalness.jpg',
              'Roughness': 'Metal055A_1K-JPG_Roughness.jpg',
              'Normal': 'Metal055A_1K-JPG_NormalGL.jpg'
          }
      )
    elif material == "Wood":
      apply_pbr_material(
          obj=cube, 
          texture_dir="./database/material/Wood066_1K-JPG/",  # 替换为实际路径
          texture_files={
              'Base Color': 'Wood066_1K-JPG_Color.jpg',
              'Roughness': 'Wood066_1K-JPG_Roughness.jpg',
              'Normal': 'Wood066_1K-JPG_NormalGL.jpg'
          }
      )
    

    target_location = (0, 0, 1.6)
    setting_camera(camera_location, target_location)

    render_scene()
    # if save_path:
    #     save_blend_file("./temp.blend")
        

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iter, weight,  high, deformation, noise, max_deformation, spring_constant, f"{material}'s density:{material_density[material]}", (x,y,z), file_path])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument('--circle', action='store_true', help="A boolean flag argument")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--resolution', type=int, help="resolution of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    iteration_size = arguments.size  # 每次渲染的批次数量
    resolution =  arguments.resolution

    # CSV 文件路径
    csv_file = f"./database/rendered_spring_{resolution}P/spring_scene_{resolution}P.csv"
    if arguments.circle:
      csv_file = f"./database/rendered_spring_circle_{resolution}P/spring_scene_circle_{resolution}P.csv"


    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", "weight", "spring high", "deformation", "noise", "max_deformation", "spring_constant", "matrial", "cube size", "img_path"])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        

        # 设置背景、场景和渲染输出路径
        background = "./database/blank_background_spring.blend"
        render_output_path = f"./database/rendered_spring_{resolution}P/"
        if arguments.circle:
          render_output_path = './database/rendered_spring_circle_{resolution}//'

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in range(arguments.iter, arguments.iter + iteration_size):
            main(
                background=background,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iter=i,
                circle = arguments.circle,
                resolution = resolution
            )
