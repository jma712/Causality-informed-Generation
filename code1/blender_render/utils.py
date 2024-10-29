import re
import bpy
import math
from mathutils import Vector
import logging

# 设置日志文件名、日志级别和格式
logging.basicConfig(filename='error.log',    # 日志文件路径
                    level=logging.DEBUG,   # 记录日志的级别，DEBUG 是最低级别，记录所有日志
                    format='%(asctime)s - %(levelname)s - %(message)s')  # 日志格式

def parse_lever_analysis(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    scenario_pattern = re.compile(
        r"\d+\.\s+\w+\s+Scenario:\n"
        r"- Context: Right Force = (?P<right_force>\d+) kg, Right Arm = (?P<right_arm>\d+) cm, "
        r"Left Force = (?P<left_force>\d+) kg, Left Arm = (?P<left_arm>\d+) cm\n"
        r"- Right Force \(F₁\) = \d+ kg\n"
        r"- Right Arm \(d₁\) = \d+ cm\n"
        r"- Left Force \(F₂\) = \d+ kg\n"
        r"- Left Arm \(d₂\) = \d+ cm\n"
        r"Result: The lever will tilt to the (?P<result>\w+)."
    )
    
    matches = scenario_pattern.findall(text)
    
    results = []
    for match in matches:
        scenario = {
            "right_force": int(match[0]),
            "right_arm": int(match[1]),
            "left_force": int(match[2]),
            "left_arm": int(match[3]),
            "result": match[4]
        }
        results.append(scenario)
    
    return results
  
def cylinder_volume(radius, height):
    return math.pi * radius**2 * height

def get_r_h_cylinder(volumn, ratio):
    unit = 1
    unit_r = unit
    unit_h = ratio * unit
    temp = volumn/math.pi
    unit_v = temp/(unit_r**2 * unit_h)
    return unit_v*unit_r, unit_v*unit_h
  
def clear_scene():
    """清空当前场景中的所有对象。"""
    if bpy.context.active_object is not None:
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def get_rotation(height, lever_length, lever_x_offset,result):
    if result == "left":
      x = lever_length/2 - (lever_x_offset)
      if height > x:
        print(f"\033[31mheight: {height}, x: {x}\033[0m")
        # 设置日志级别和输出格式
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.warning(f'height is greater than hypotenuse: {height} > {x}')
        return 0
      angle_radians = math.asin(height/x)
      angle_radians = - angle_radians
    else:
      x = lever_length/2 + (lever_x_offset)
      angle_radians = math.asin(height/x)
      angle_radians = angle_radians
    return angle_radians
  
def create_metal_tabletop(location=(0, 0, 0), size=(2, 1, 0.05)):
    """创建一个金属桌面。"""
    import bpy

    # 创建平面
    bpy.ops.mesh.primitive_cube_add(location=location)
    tabletop = bpy.context.object
    tabletop.scale = (size[0]/2, size[1]/2, size[2]/2)

    # 启用平滑着色
    bpy.ops.object.shade_smooth()

    # 创建金属材质
    mat = bpy.data.materials.new(name="TabletopMetalMaterial")
    mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    # 创建节点
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # 设置材质属性
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)  # 银色
    bsdf.inputs['Metallic'].default_value = 1.0
    bsdf.inputs['Roughness'].default_value = 0.2  # 光滑的金属表面

    # 连接节点
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # 将材质赋予桌面
    tabletop.data.materials.append(mat)

    return tabletop
  
  
def create_marble_tabletop(location=(0, 0, 0), size=(2, 1, 0.05)):
    """创建一个大理石桌面。"""
    import bpy
    import os

    # 创建平面
    bpy.ops.mesh.primitive_cube_add(location=location)
    tabletop = bpy.context.object
    tabletop.scale = (size[0]/2, size[1]/2, size[2]/2)

    # 启用平滑着色
    bpy.ops.object.shade_smooth()

    # 创建大理石材质
    mat = bpy.data.materials.new(name="TabletopMarbleMaterial")
    mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    # 创建节点
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture_node = nodes.new(type="ShaderNodeTexImage")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    texture_coord_node = nodes.new(type="ShaderNodeTexCoord")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # 加载大理石纹理（请替换为您的纹理路径）
    texture_path = '/path/to/your/marble_texture.jpg'  # 替换为实际的纹理路径
    if os.path.exists(texture_path):
        texture_node.image = bpy.data.images.load(texture_path)
    else:
        print("未找到大理石纹理，使用默认颜色。")
        bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)  # 灰白色

    # 设置材质属性
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.3  # 光滑的石材表面

    # 连接节点
    if texture_node.image:
        links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        links.new(texture_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # 将材质赋予桌面
    tabletop.data.materials.append(mat)

    return tabletop


def create_tabletop(location=(0, 0, 0), size=(2, 1, 0.05)):
    """创建一个木质桌面。"""
    import bpy
    import os

    # 创建平面
    bpy.ops.mesh.primitive_cube_add(location=location)
    tabletop = bpy.context.object
    tabletop.scale = (size[0]/2, size[1]/2, size[2]/2)  # 调整尺寸

    # 启用平滑着色
    bpy.ops.object.shade_smooth()

    # 创建木材材质
    mat = bpy.data.materials.new(name="TabletopWoodMaterial")
    mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    # 创建节点
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture_node = nodes.new(type="ShaderNodeTexImage")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    texture_coord_node = nodes.new(type="ShaderNodeTexCoord")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # 加载木纹纹理（请替换为您的纹理路径）
    texture_path = '/path/to/your/wood_texture.jpg'  # 替换为实际的纹理路径
    if os.path.exists(texture_path):
        texture_node.image = bpy.data.images.load(texture_path)
    else:
        print("未找到木纹纹理，使用默认颜色。")
        bsdf.inputs['Base Color'].default_value = (0.6, 0.4, 0.2, 1)  # 棕色

    # 设置材质属性
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.6  # 适度的粗糙度

    # 连接节点
    if texture_node.image:
        links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        links.new(texture_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # 将材质赋予桌面
    tabletop.data.materials.append(mat)

    return tabletop
  
def create_tabletop_walnut(location=(0, 0, 0), size=(2, 1, 0.05)):
    """创建一个胡桃木桌面。"""
    import bpy
    import os

    # 创建平面
    bpy.ops.mesh.primitive_cube_add(location=location)
    tabletop = bpy.context.object
    tabletop.scale = size

    # 应用缩放变换
    # bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 进入编辑模式
    bpy.ops.object.mode_set(mode='EDIT')

    # 选择所有面
    bpy.ops.mesh.select_all(action='SELECT')

    # 进行智能 UV 展开
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)

    # 退出编辑模式
    bpy.ops.object.mode_set(mode='OBJECT')

    # 启用平滑着色
    bpy.ops.object.shade_smooth()

    # 创建胡桃木材质
    mat = bpy.data.materials.new(name="WalnutWoodMaterial")
    mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    # 创建节点
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture_node = nodes.new(type="ShaderNodeTexImage")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    texture_coord_node = nodes.new(type="ShaderNodeTexCoord")
    normal_map_node = nodes.new(type="ShaderNodeTexImage")
    normal_map = nodes.new(type="ShaderNodeNormalMap")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # 加载胡桃木纹理（请确保路径正确）
    base_color_path = '/Users/liu/Downloads/causal_inference/scene_support/texture/plywood_diff_4k.jpg'
    normal_map_path = '/Users/liu/Downloads/causal_inference/scene_support/texture/plywood_nor_gl_4k.exr'

    if os.path.exists(base_color_path):
        print(f"正在加载颜色贴图：{base_color_path}")
        texture_node.image = bpy.data.images.load(base_color_path)
        texture_node.image.colorspace_settings.name = 'sRGB'  # 颜色贴图使用 sRGB
    else:
        print(f"未找到颜色贴图：{base_color_path}")
        bsdf.inputs['Base Color'].default_value = (0.6, 0.4, 0.2, 1)  # 默认棕色

    # 设置材质属性
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.4

    # 连接纹理坐标到映射节点
    links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])

    # 连接映射节点到颜色纹理和法线贴图
    links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], normal_map_node.inputs['Vector'])

    # 连接颜色纹理到 BSDF 基础颜色
    links.new(texture_node.outputs['Color'], bsdf.inputs['Base Color'])

    # 如果有法线贴图，加载并连接
    if os.path.exists(normal_map_path):
        print(f"正在加载法线贴图：{normal_map_path}")
        normal_map_node.image = bpy.data.images.load(normal_map_path)
        normal_map_node.image.colorspace_settings.is_data = True  # 法线贴图使用 Non-Color

        # 连接法线贴图到法线节点，再连接到 BSDF
        links.new(normal_map_node.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    else:
        print(f"未找到法线贴图：{normal_map_path}")

    # 连接 BSDF 到材质输出
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # 将材质赋予桌面
    tabletop.data.materials.append(mat)

    return tabletop

def setting_camera(location, target, scene_bounds = ((-30, 30), (-30, 30), (0, 30))):
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

    # Check if a camera exists, otherwise create one
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    else:
        camera = bpy.data.objects["Camera"]

    # Set the camera location
    camera.location = Vector(clamped_location)

    # Point the camera at the target location
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    print(f"Camera location set to {camera.location}, pointing towards {target}")