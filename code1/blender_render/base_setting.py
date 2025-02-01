import bpy
import math
import numpy as np
from typing import List, Tuple, Union
import argparse
import sys
import random
from mathutils import Vector
import os

def rotate_object_y_axis_by_name(object_name, angle):
    """
    Rotates the specified object along the Y-axis by a given angle.

    Parameters:
        object_name (str): The name of the object to rotate.
        angle (float): The rotation angle in degrees.
    """
    # Get the object by name
    obj = bpy.data.objects.get(object_name)
    
    if obj is None:
        print(f"Object '{object_name}' not found in the scene.")
        return

    # Ensure the object is active and selected
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Convert angle to radians and apply the rotation on Y-axis
    radians = math.radians(angle)
    obj.rotation_euler[1] -= radians  # Euler order is (X, Y, Z)

    print(f"Rotated object '{object_name}' by {angle} degrees along Y-axis.")
    
def setting_camera(location, target, scene_bounds=((-30, 30), (-30, 30), (0, 30)), len_ = None):
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
    if len_ is not None:
      camera.data.lens = len_
    print(f"Camera location set to {camera.location}, pointing towards {target}")

def clear_scene():
    """删除当前场景中的所有对象。"""
    bpy.ops.object.select_all(action='SELECT')  # 选择所有对象
    bpy.ops.object.delete()  # 删除选中的对象
    print("清空场景完成。")

def load_blend_file(filepath, location=(0, 0, 0), scale=(1, 1, 1), rotation_angle=0, color=None):
    """
    导入指定的 .blend 文件中的所有对象，并调整位置、缩放和旋转方向。
    如果指定了颜色，则设置导入对象的颜色。
    
    参数:
    - filepath: str, .blend 文件的路径
    - location: tuple, 导入模型的位置 (x, y, z)
    - scale: tuple, 导入模型的缩放比例 (x, y, z)
    - rotation_angle: float, 导入模型的旋转角度（以弧度为单位）在Z轴方向
    - color: tuple, 可选, RGBA (0-1) 格式的颜色值, 例如 (1.0, 0.0, 0.0, 1.0) 表示红色
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
            obj.scale = scale
            
            # 设置旋转角度
            obj.rotation_euler[2] = math.radians(rotation_angle)
            
            # 如果指定了颜色，则设置对象的颜色
            if color:
                # 创建材质并应用颜色
                material_name = f"{obj.name}_Material"
                if not bpy.data.materials.get(material_name):
                    material = bpy.data.materials.new(name=material_name)
                else:
                    material = bpy.data.materials[material_name]
                
                material.use_nodes = True
                bsdf = material.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)  # 设置颜色，使用 RGB，完全不透明
                
                # 将材质应用到对象
                obj.data.materials.clear()  # 清除已有材质
                obj.data.materials.append(material)
                print(f"为对象 '{obj.name}' 设置了颜色 {color}")

    print("场景已导入成功！")
    
def load_blend_file_backgournd(filepath):
    """导入指定的 .blend 文件中的所有对象。"""
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects  # 选择导入所有对象
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
    print("场景已导入成功！")


def set_render_parameters(resolution=(1920, 1080), file_format='PNG', output_path="../database/rendered_image.png", 
                          res=None, circle=False, use_gpu=True):
    """
    设置渲染参数，包括分辨率、格式、输出路径和 GPU 渲染。
    
    Parameters:
        resolution (tuple): 渲染分辨率 (宽度, 高度)。
        file_format (str): 输出文件格式，例如 'PNG' 或 'JPEG'。
        output_path (str): 输出图像路径。
        res (int): 分辨率百分比（1-100）。默认100%。
        circle (bool): 是否使用 Cycles 渲染引擎。
        use_gpu (bool): 是否启用 GPU 渲染。
    """
    # 设置分辨率
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = res if res is not None else 100

    # 设置输出路径和格式
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format

    # 设置渲染引擎
    if circle:
        bpy.context.scene.render.engine = 'CYCLES'
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    
    # 设置 GPU 渲染（仅适用于 Cycles）
    if use_gpu and bpy.context.scene.render.engine == 'CYCLES':
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # 使用CUDA（NVIDIA GPU）
        bpy.context.scene.cycles.device = 'GPU'

        # 设置设备
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True  # 启用所有可用设备

    print("Render parameters set:")
    print(f"Resolution: {resolution[0]}x{resolution[1]} ({bpy.context.scene.render.resolution_percentage}%)")
    print(f"File format: {file_format}")
    print(f"Output path: {output_path}")
    print(f"Engine: {'Cycles' if circle else 'Eevee'}")
    print(f"GPU Enabled: {use_gpu and bpy.context.scene.render.engine == 'CYCLES'}")

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
    
    # 限制形变量在最大允许范围内
    if deformation > max_deformation:
        deformation = max_deformation
        print("Warning: Deformation exceeded maximum limit. Limiting to max deformation.")
    
    return deformation

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
    bpy.ops.object.shade_smooth()

    # 创建胡桃木材质
    mat = bpy.data.materials.new(name="GroundWalnutMaterial")
    mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    # 创建节点
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture_node = nodes.new(type="ShaderNodeTexImage")
    normal_map_node = nodes.new(type="ShaderNodeTexImage")
    normal_map = nodes.new(type="ShaderNodeNormalMap")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    texture_coord_node = nodes.new(type="ShaderNodeTexCoord")
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    
    if "Base Color" in texture_files:
      base_color_path = os.path.join(texture_dir, texture_files["Base Color"])

      if os.path.exists(base_color_path):
          texture_node.image = bpy.data.images.load(base_color_path)
          texture_node.image.colorspace_settings.name = 'sRGB'
      else:
          print(f"未找到颜色贴图：{base_color_path}")
          bsdf.inputs['Base Color'].default_value = (0.6, 0.4, 0.2, 1)  # 默认棕色
    if  "Metalness" in texture_files:
      metalness_map_path = os.path.join(texture_dir, texture_files["Metalness"])
      if os.path.exists(metalness_map_path):
          bsdf.inputs['Metallic'].default_value = 1.0
          texture_node.image = bpy.data.images.load(metalness_map_path)
          texture_node.image.colorspace_settings.is_data = True  # 设置为非颜色数据
      else:
          print(f"未找到金属贴图：{metalness_map_path}")
          bsdf.inputs['Metallic'].default_value = 0.0
    if "Roughness" in texture_files:
      roughness_map_path = os.path.join(texture_dir, texture_files["Roughness"])
      if os.path.exists(roughness_map_path):
          bsdf.inputs['Roughness'].default_value = 0.4
          texture_node.image = bpy.data.images.load(roughness_map_path)
          texture_node.image.colorspace_settings.is_data = True
          
      else:
          print(f"未找到粗糙度贴图：{roughness_map_path}")
          bsdf.inputs['Roughness'].default_value = 0.4
          
    if "Normal" in texture_files:
      normal_map_path = os.path.join(texture_dir, texture_files["Normal"])
      if os.path.exists(normal_map_path):
          normal_map_node.image = bpy.data.images.load(normal_map_path)
          normal_map_node.image.colorspace_settings.is_data = True
      else:
          print(f"未找到法线贴图：{normal_map_path}")
          normal_map_node = None
          

    # 设置材质属性
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.4

    # 连接纹理坐标到映射节点
    links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])

    # 连接映射节点到颜色纹理和法线贴图
    links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
    if normal_map_node:
        links.new(mapping_node.outputs['Vector'], normal_map_node.inputs['Vector'])

    # 连接颜色纹理到 BSDF 基础颜色
    links.new(texture_node.outputs['Color'], bsdf.inputs['Base Color'])

    if normal_map_node:
        links.new(normal_map_node.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    obj.data.materials.append(mat)

# Function to create a rectangular prism
def create_rectangular_prism(location=(0, 0, 0), dimensions=(1, 1.5, 4)):
    """
    Creates a rectangular prism by scaling a cube.
    
    Parameters:
        location (tuple): The (x, y, z) coordinates of the prism's center.
        dimensions (tuple): The (width, depth, height) of the prism.
    """
    # Add a default cube
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    
    # Get the newly created object
    obj = bpy.context.object
    
    # Set its scale to match the desired dimensions
    obj.scale = (dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2)
    return obj

def rotate_object_around_edge(obj, angle, edge="bottom-right"):
    """
    Rotates the given object around the specified edge by a certain angle.
    
    Parameters:
        obj (bpy.types.Object): The Blender object to rotate.
        angle (float): The rotation angle in degrees.
        edge (str): The edge to use as the axis of rotation.
                    Options: "bottom-right", "bottom-left", "top-right", "top-left"
    """
    # Get object dimensions and location in world coordinates
    dimensions = obj.dimensions
    location = obj.location

    # Compute the edge location based on the specified edge
    if edge == "bottom-right":
        edge_location = Vector((
            location.x + dimensions.x / 2,  # x_max
            location.y - dimensions.y / 2,  # y_min
            location.z - dimensions.z / 2   # z_min
        ))
    elif edge == "bottom-left":
        edge_location = Vector((
            location.x - dimensions.x / 2,  # x_min
            location.y - dimensions.y / 2,  # y_min
            location.z - dimensions.z / 2   # z_min
        ))
    elif edge == "top-right":
        edge_location = Vector((
            location.x + dimensions.x / 2,  # x_max
            location.y - dimensions.y / 2,  # y_min
            location.z + dimensions.z / 2   # z_max
        ))
    elif edge == "top-left":
        edge_location = Vector((
            location.x - dimensions.x / 2,  # x_min
            location.y - dimensions.y / 2,  # y_min
            location.z + dimensions.z / 2   # z_max
        ))
    else:
        raise ValueError("Invalid edge specified. Options: bottom-right, bottom-left, top-right, top-left")

    # Set the object's origin to the edge location
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = edge_location

    # Rotate the object around the edge by the specified angle
    obj.rotation_euler = (math.radians(angle), 0, 0)  # Rotate around X-axis as an example



def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend"
  ):
    clear_scene()
    disable_shadows_for_render()
    if 'blank' in background.lower():
      background = "./database/blank_background_spring.blend"
      load_blend_file_backgournd(background)

    set_render_parameters(output_path=render_output_path)
    camera_location = (random.uniform(-0, 0), random.uniform(15, 15), random.uniform(1, 1))
    load_blend_file("./database/Spring.blend")
    
    weight = random.uniform(2, 10)
    materials = ["Iron", "Wood"]
    material = random.choice(materials)
    x,y,z, cube = create_cube_based_on_weight(weight=weight, density=material_density[material])

    spring_constant = 10  # 弹簧劲度系数 (N/m)

    high = 1.3
    max_deformation = high * 0.83
    deformation = calculate_spring_deformation(weight, spring_constant, max_deformation)
    
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
          texture_dir="/Users/liu/Desktop/school_academy/Case/Yin/causal_project/Causality-informed-Generation/code1/database/material/Wood060_1K-JPG",  # 替换为实际路径
          texture_files={
              'Base Color': 'Wood060_1K-JPG_Color.jpg',
              'Metalness': 'Metal055A_1K-JPG_Metalness.jpg',
              'Roughness': 'Wood060_1K-JPG_Roughness',
              'Normal': 'Wood060_1K-JPG__NormalGL.jpg'
          }
      )
    

    target_location = (0, 0, 1.5)
    setting_camera(camera_location, target_location)

    render_scene()

    if save_path:
        save_blend_file(save_path)
        
    # dic = ("weight:", weight, "spring high:",high,  "deformation:",deformation,  "spring_constant: ", spring_constant, "matrial: ", material, "cube size: ", (x,y,z))
    # return dic
  
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