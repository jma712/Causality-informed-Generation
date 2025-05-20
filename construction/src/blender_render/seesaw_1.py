import bpy
import math
import sys
import os

# 将目标文件夹（utils）的路径添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils import parse_lever_analysis, cylinder_volume, get_r_h_cylinder, clear_scene,create_metal_tabletop,get_rotation,create_marble_tabletop, create_tabletop
from utils import create_tabletop_walnut
import logging

ground_size = 10
# 设置日志文件名、日志级别和格式
logging.basicConfig(filename='error_seesaw.log',    # 日志文件路径
                    level=logging.DEBUG,   # 记录日志的级别，DEBUG 是最低级别，记录所有日志
                    format='%(asctime)s - %(levelname)s - %(message)s')  # 日志格式

def create_ground(location=(0, 0, 0), size=(10, 10, 0.2), plain_plane = True):
    bpy.ops.mesh.primitive_cube_add(location=location)
    ground = bpy.context.object
    ground.name = "Ground"
    bevel_width=0.1 if plain_plane else 0.01
    bevel_segments=8

    gray_material = bpy.data.materials.new(name="GrayMaterial")
    gray_material.use_nodes = True

    # 检查或添加 "Principled BSDF" 节点
    node_tree = gray_material.node_tree
    bsdf_node = node_tree.nodes.get("Principled BSDF")
    if not bsdf_node:
        bsdf_node = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")

    # 设置材质颜色为灰色
    bsdf_node.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)  # RGB 灰色


    # 缩放平面到所需尺寸
    if not plain_plane:
      ground.scale = (size[0], size[1], size[2] / 2)
    else:
      ground.scale = (size[0]*10, size[1]*10, size[2] / 2)
      
      # 获取地面的尺寸
      size_x = ground.scale[0] * 2  # 地面平面的宽度（scale[0] 为一半）
      size_y = ground.scale[1] * 2  # 地面平面的长度（scale[1] 为一半）
      wall_thickness = 3         # 墙的厚度
      wall_height = 100.0             # 墙的高度

      # 定义四面墙的位置和尺寸
      # 墙体1：左侧
      # 假设已有 size_x, size_y, wall_thickness, wall_height 的定义

      # 假设已有 size_x, size_y, wall_thickness, wall_height 的定义

      bpy.ops.mesh.primitive_cube_add(size=1, location=(-size_x / 2, 0, wall_height / 2))
      left_wall = bpy.context.object
      left_wall.scale = (wall_thickness, size_y / 1, wall_height / 1)

      # 创建一个灰色材质
      gray_material = bpy.data.materials.new(name="GrayMaterial")
      gray_material.use_nodes = True

      # 获取节点树并检查
      node_tree = gray_material.node_tree
      if not node_tree:
          print("Error: Material node tree not found.")
      else:
          # 获取或创建 "Principled BSDF" 节点
          bsdf_node = node_tree.nodes.get("Principled BSDF")
          if not bsdf_node:
              bsdf_node = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")

          # 确保 bsdf_node 不为 None，然后设置颜色
          if bsdf_node:
              bsdf_node.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)  # RGB 灰色
          else:
              print("Error: Unable to create or find 'Principled BSDF' node.")

          # 获取或创建材质输出节点
          material_output = node_tree.nodes.get("Material Output")
          if not material_output:
              material_output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")

          # 确保材质输出节点存在，并连接 BSDF 节点
          if material_output and bsdf_node:
              node_tree.links.new(bsdf_node.outputs["BSDF"], material_output.inputs["Surface"])
          else:
              print("Error: Unable to link nodes.")

      # 将材质应用到 left_wall 对象
      left_wall.data.materials.append(gray_material)
      # 墙体2：右侧
      bpy.ops.mesh.primitive_cube_add(size=1, location=(size_x / 2, 0, wall_height / 2))
      right_wall = bpy.context.object
      right_wall.scale = (wall_thickness, size_y / 1, wall_height / 1)
      right_wall.data.materials.append(gray_material)  # 添加灰色材质
      
      # 墙体3：前方
      bpy.ops.mesh.primitive_cube_add(size=1, location=(0, size_y / 2, wall_height / 2))
      front_wall = bpy.context.object
      front_wall.scale = (size_x / 1, wall_thickness, wall_height / 1)
      front_wall.data.materials.append(gray_material)  # 添加灰色材质

      # 墙体4：后方
      bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -size_y / 2, wall_height / 2))
      back_wall = bpy.context.object
      back_wall.scale = (size_x / 1, wall_thickness, wall_height / 1)
      back_wall.data.materials.append(gray_material)  # 添加灰色材质
      
        
      # 定义一个简单的灰色材质
      gray_material = bpy.data.materials.new(name="GrayMaterial")
      gray_material.use_nodes = True

      # 获取节点树并创建 "Principled BSDF" 节点
      node_tree = gray_material.node_tree
      bsdf_node = node_tree.nodes.get("Principled BSDF")
      if not bsdf_node:
          bsdf_node = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")

      # 设置材质颜色为灰色
      bsdf_node.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)

      # 将 BSDF 连接到材质输出
      material_output = node_tree.nodes.get("Material Output")
      node_tree.links.new(bsdf_node.outputs["BSDF"], material_output.inputs["Surface"])
      

    # 应用缩放变换
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # 移动立方体，使其上平面位于z=0
    ground.location.z -= size[2] / 2
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.bevel(affect='VERTICES')
    bpy.ops.object.mode_set(mode='OBJECT')

    bevel = ground.modifiers.new(name="Bevel", type='BEVEL')

    bevel.width = bevel_width  # 圆角的宽度
    bevel.segments = bevel_segments  # 圆角的细分段数，越高越平滑
    bevel.profile = 0.5  # 圆角的形状，0.5是圆形
    
    # 应用倒角修饰符
    bpy.context.view_layer.objects.active = ground
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # 应用平滑着色
    bpy.ops.object.shade_smooth()

    if not plain_plane:
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

      # 加载胡桃木纹理（请替换为您的纹理路径）
      base_color_path = '/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture/Wood060_1K-JPG/Wood060_1K-JPG_Color.jpg'  # 替换为实际的颜色贴图路径
      normal_map_path = '/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture//scene_support/texture/Wood060_1K-JPG/Wood060_1K-JPG_NormalDX.jpg'     # 替换为实际的法线贴图路径
      displacement_path = "/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture//scene_support/texture/Wood060_1K-JPG_Displacement.jpg"
      roughness_path = '/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture//scene_support/texture/Wood060_1K-JPG_Roughness.jpg"'
      ao_path = '/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture//scene_support/texture/Wood060_1K-JPG/Wood060_1K-JPG_AmbientOcclusion.jpg"'


      if os.path.exists(base_color_path):
          texture_node.image = bpy.data.images.load(base_color_path)
          texture_node.image.colorspace_settings.name = 'sRGB'
      else:
          print(f"未找到颜色贴图：{base_color_path}")
          bsdf.inputs['Base Color'].default_value = (0.6, 0.4, 0.2, 1)  # 默认棕色

      if os.path.exists(normal_map_path):
          normal_map_node.image = bpy.data.images.load(normal_map_path)
          normal_map_node.image.colorspace_settings.is_data = True  # 设置为非颜色数据
      else:
          print(f"未找到法线贴图：{normal_map_path}")
          normal_map_node = None  # 如果没有法线贴图

      # 3. Roughness Map 粗糙度贴图
      if os.path.exists(roughness_path):
          roughness_node = nodes.new(type="ShaderNodeTexImage")
          roughness_node.image = bpy.data.images.load(roughness_path)
          roughness_node.image.colorspace_settings.is_data = True

          links.new(mapping_node.outputs['Vector'], roughness_node.inputs['Vector'])
          links.new(roughness_node.outputs['Color'], bsdf.inputs['Roughness'])
      else:
          print(f"未找到粗糙度贴图：{roughness_path}")

      # 4. Ambient Occlusion Map 环境遮挡贴图
      if os.path.exists(ao_path):
          ao_node = nodes.new(type="ShaderNodeTexImage")
          ao_node.image = bpy.data.images.load(ao_path)
          ao_node.image.colorspace_settings.is_data = True

          # 混合环境遮挡与 Base Color
          mix_rgb_node = nodes.new(type="ShaderNodeMixRGB")
          mix_rgb_node.blend_type = 'MULTIPLY'
          mix_rgb_node.inputs['Fac'].default_value = 1.0  # 完全混合

          links.new(mapping_node.outputs['Vector'], ao_node.inputs['Vector'])
          links.new(ao_node.outputs['Color'], mix_rgb_node.inputs['Color2'])
          links.new(base_color_node.outputs['Color'], mix_rgb_node.inputs['Color1'])
          links.new(mix_rgb_node.outputs['Color'], bsdf.inputs['Base Color'])
      else:
          print(f"未找到环境遮挡贴图：{ao_path}")

      # 5. Displacement Map 位移贴图
      if os.path.exists(displacement_path):
          displacement_node = nodes.new(type="ShaderNodeTexImage")
          displacement_node.image = bpy.data.images.load(displacement_path)
          displacement_node.image.colorspace_settings.is_data = True

          displacement = nodes.new(type="ShaderNodeDisplacement")
          links.new(mapping_node.outputs['Vector'], displacement_node.inputs['Vector'])
          links.new(displacement_node.outputs['Color'], displacement.inputs['Height'])

          # 将位移节点连接到材质输出的位移端口
          links.new(displacement.outputs['Displacement'], material_output.inputs['Displacement'])
      else:
          print(f"未找到位移贴图：{displacement_path}")

      # 连接纹理坐标到映射节点

      links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])

      # 连接映射节点到颜色纹理和法线贴图
      links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
      if normal_map_node:
          links.new(mapping_node.outputs['Vector'], normal_map_node.inputs['Vector'])

      # 连接颜色纹理到 BSDF 基础颜色
      links.new(texture_node.outputs['Color'], bsdf.inputs['Base Color'])

      # 如果有法线贴图，连接法线节点
      if normal_map_node:
          links.new(normal_map_node.outputs['Color'], normal_map.inputs['Color'])
          links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

      # 连接 BSDF 到材质输出
      links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

      # 将材质赋予地面
      ground.data.materials.append(mat)

    # 添加刚体模拟，确保地面是被动物体
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'  # 地面作为被动刚体

    return ground

def create_plastic_tabletop(location=(0, 0, 0), size=(2, 1, 0.05), color=(0.2, 0.2, 0.8, 1)):
    """创建一个塑料材质的桌面。"""
    import bpy

    # 创建平面
    bpy.ops.mesh.primitive_cube_add(location=location)
    tabletop = bpy.context.object
    tabletop.scale = (size[0]/2, size[1]/2, size[2]/2)

    # 启用平滑着色
    bpy.ops.object.shade_smooth()

    # 创建塑料材质
    mat = bpy.data.materials.new(name="TabletopPlasticMaterial")
    mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    # 创建节点
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # 设置材质属性
    bsdf.inputs['Base Color'].default_value = color  # 使用传入的颜色
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.5  # 适度的粗糙度

    # 连接节点
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # 将材质赋予桌面
    tabletop.data.materials.append(mat)

    return tabletop

def create_sun(location=(0, 0, 10), strength=5.0):
    """创建太阳光源。"""
    bpy.ops.object.light_add(type='SUN', location=location)
    sun = bpy.context.object
    sun.name = "Sun"
    sun.data.energy = strength
    sun.data.color = (1.0, 1.0, 0.9)
    sun.data.use_shadow = True
    sun.data.shadow_soft_size = 0.5
    return sun
  
def create_lever_1(length=5, width=1, height=0.2, location=(0, 0, 0)):
    """创建更为真实的木质杠杆（木板）。"""
    # 创建木板
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    lever = bpy.context.object
    lever.name = "Lever"
    lever.scale[0] = length  # 调整X轴方向长度
    lever.scale[1] = width   # 调整Y轴方向宽度
    lever.scale[2] = height  # 调整Z轴方向厚度
    
    # 添加倒角修饰器
    bevel_segments = 3
    bevel_width = 0.08
    bevel = lever.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = bevel_width
    bevel.segments = bevel_segments
    bevel.profile = 0.7

    # 应用倒角修饰器
    bpy.context.view_layer.objects.active = lever
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # 创建木材材质
    mat = bpy.data.materials.new(name="LeverMaterial")
    mat.use_nodes = True  # 启用节点系统

    # 获取节点树
    node_tree = mat.node_tree

    # 清除默认节点（可选，但推荐）
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # 创建新的 Principled BSDF 节点和 Material Output 节点
    bsdf = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    material_output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")

    # 设置木材的颜色和纹理参数
    bsdf.inputs['Base Color'].default_value = (0.768533, 0.800023, 0.0236259, 1)
    bsdf.inputs['Roughness'].default_value = 0.9  # 木材表面的粗糙度

    # 使用噪声纹理来模拟木纹效果
    noise_texture = node_tree.nodes.new(type="ShaderNodeTexNoise")
    noise_texture.inputs['Scale'].default_value = 15  # 调整噪声纹理的细节级别

    # 添加“颜色渐变”节点来控制木纹的颜色分布
    color_ramp = node_tree.nodes.new(type="ShaderNodeValToRGB")
    color_ramp.color_ramp.interpolation = 'LINEAR'

    # 配置木纹颜色
    color_ramp.color_ramp.elements[0].color = (0.4, 0.2, 0.05, 1)  # 深色木纹
    color_ramp.color_ramp.elements[1].color = (0.8, 0.5, 0.3, 1)  # 浅色木纹

    # 连接节点
    node_tree.links.new(noise_texture.outputs['Fac'], color_ramp.inputs['Fac'])
    node_tree.links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])

    # 添加凹凸节点（Bump）来增强木材纹理的立体感
    bump_node = node_tree.nodes.new(type="ShaderNodeBump")
    bump_node.inputs['Strength'].default_value = 0.1  # 适中的凹凸强度
    node_tree.links.new(noise_texture.outputs['Fac'], bump_node.inputs['Height'])
    node_tree.links.new(bump_node.outputs['Normal'], bsdf.inputs['Normal'])

    # **关键步骤：将 BSDF 节点连接到 Material Output 节点**
    node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # 将材质赋予木板
    lever.data.materials.append(mat)

    # 添加刚体模拟
    bpy.ops.rigidbody.object_add()
    lever.rigid_body.type = 'PASSIVE'  # 杠杆作为被动刚体

    return lever

def create_lever(length=5, width=1, height=0.2, location=(0, 0, 0)):
    """创建更为真实的木质杠杆（木板）。"""
    import bpy
    # 创建木板
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    lever = bpy.context.object
    lever.name = "Lever"
    lever.scale[0] = length  # 调整X轴方向长度
    lever.scale[1] = width   # 调整Y轴方向宽度
    lever.scale[2] = height  # 调整Z轴方向厚度
    
    # 添加倒角修饰器
    bevel_segments = 3
    bevel_width = 0.08
    bevel = lever.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = bevel_width
    bevel.segments = bevel_segments
    bevel.profile = 0.7

    # 应用倒角修饰器
    bpy.context.view_layer.objects.active = lever
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # 创建木材材质
    mat = bpy.data.materials.new(name="LeverMaterial")
    mat.use_nodes = True  # 启用节点系统

    # 获取节点树
    node_tree = mat.node_tree

    # 清除默认节点（可选，但推荐）
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # 创建新的 Principled BSDF 节点和 Material Output 节点
    bsdf = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    material_output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")

    # 设置木材的颜色和纹理参数
    bsdf.inputs['Base Color'].default_value = (0.768533, 0.800023, 0.0236259, 1)
    bsdf.inputs['Roughness'].default_value = 0.9  # 木材表面的粗糙度

    # 使用噪声纹理来模拟木纹效果
    noise_texture = node_tree.nodes.new(type="ShaderNodeTexNoise")
    noise_texture.inputs['Scale'].default_value = 15  # 调整噪声纹理的细节级别

    # 添加“颜色渐变”节点来控制木纹的颜色分布
    color_ramp = node_tree.nodes.new(type="ShaderNodeValToRGB")
    color_ramp.color_ramp.interpolation = 'LINEAR'

    # 配置木纹颜色
    color_ramp.color_ramp.elements[0].color = (0.4, 0.2, 0.05, 1)  # 深色木纹
    color_ramp.color_ramp.elements[1].color = (0.8, 0.5, 0.3, 1)  # 浅色木纹

    # 连接节点
    node_tree.links.new(noise_texture.outputs['Fac'], color_ramp.inputs['Fac'])
    node_tree.links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])

    # 添加凹凸节点（Bump）来增强木材纹理的立体感
    bump_node = node_tree.nodes.new(type="ShaderNodeBump")
    bump_node.inputs['Strength'].default_value = 0.1  # 适中的凹凸强度
    node_tree.links.new(noise_texture.outputs['Fac'], bump_node.inputs['Height'])
    node_tree.links.new(bump_node.outputs['Normal'], bsdf.inputs['Normal'])

    # **关键步骤：将 BSDF 节点连接到 Material Output 节点**
    node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # 将材质赋予木板
    lever.data.materials.append(mat)

    # 添加刚体模拟
    bpy.ops.rigidbody.object_add()
    lever.rigid_body.type = 'PASSIVE'  # 杠杆作为被动刚体

    return lever


def create_pivot(location=(1, 1, 1), radius=1, depth=1):
    """创建平躺的圆柱体支点。"""
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location)
    pivot = bpy.context.object
    pivot.name = "Pivot"
    pivot.rotation_euler[0] = 1.5708  # 绕X轴旋转90度

    # 添加 Bevel 修饰符
    bevel_modifier = pivot.modifiers.new(name="Bevel", type='BEVEL')
    bevel_modifier.width = 0.1  # 凹槽宽度
    bevel_modifier.segments = 10  # 圆角的细分数

    # 应用 Bevel 修饰符（可选，如果需要应用）
    # bpy.context.view_layer.objects.active = pivot
    # bpy.ops.object.modifier_apply(modifier=bevel_modifier.name)

    # 创建材质并启用节点
    mat = bpy.data.materials.new(name="PivotMaterial")
    mat.use_nodes = True  # 启用节点系统

    # 获取节点树
    node_tree = mat.node_tree

    # 清除默认节点（可选，但推荐）
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # 创建节点
    bsdf = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    material_output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    bump_texture = node_tree.nodes.new(type="ShaderNodeTexNoise")
    bump_node = node_tree.nodes.new(type="ShaderNodeBump")

    # 设置节点属性
    bsdf.inputs['Base Color'].default_value = (0.1, 0.1, 0.1, 1)  # 灰黑色
    bsdf.inputs['Roughness'].default_value = 0.4  # 适度的粗糙度
    bump_texture.inputs['Scale'].default_value = 10  # 调整细节级别
    bump_node.inputs['Strength'].default_value = 0.1  # 凹凸强度

    # 连接节点
    node_tree.links.new(bump_texture.outputs['Fac'], bump_node.inputs['Height'])
    node_tree.links.new(bump_node.outputs['Normal'], bsdf.inputs['Normal'])
    node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # 将材质赋予圆柱
    if len(pivot.data.materials) == 0:
        pivot.data.materials.append(mat)  # 如果没有材质，则添加材质
    else:
        pivot.data.materials[0] = mat  # 如果已有材质，替换现有材质

    return pivot

def create_camera_1(location=(0, -7, 2), rotation = (1.2, 0, 1), focal_length=7):
    """创建相机并设置其位置、朝向和焦距。"""
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "Camera"
    camera.rotation_euler = rotation
    camera.data.lens = focal_length
    bpy.context.scene.camera = camera

def create_camera(location=(0, -7, 2), rotation = (1.2, 0, 1), focal_length=7):
    """创建相机并设置其位置、朝向和焦距。"""

    # 限制相机的位置，使其不超出地面的边界
    location_x = (location[0]/abs(location[0])) * ground_size/1.5 if abs(location[0]) > ground_size/1.5 else location[0]
    location_y = (location[1]/abs(location[1])) * ground_size/1.5 if abs(location[1]) > ground_size/1.5 else location[1]
    location_z = max(location[-1], 0.25)
    # 使用调整后的相机位置创建相机
    bpy.ops.object.camera_add(location=(location_x, location_y, location_z))

    camera = bpy.context.object
    camera.name = "Camera"
    camera.rotation_euler = rotation
    camera.data.lens = focal_length
    bpy.context.scene.camera = camera

def create_weight(end_location=[0, 0, 0], weight_value=5):
    """创建更为真实的金属砝码圆柱并添加文字。"""
    # 创建砝码圆柱并应用倒角以软化边缘
    ratio_h_r = 2.5
    unit_kg = 5 #5kg
    unit_r = 0.2
    unit_h = 0.5
    unit_volumn = cylinder_volume(unit_r,unit_h)
    current_volumn = (weight_value - unit_kg)*unit_volumn
    current_r, current_h = get_r_h_cylinder(current_volumn,ratio_h_r)
    current_r = 0.02 if current_r < 0.02 else current_r
    current_h = 0.05 if current_h < 0.05 else current_h
    flag = 0
    if current_r == 0.02  and  current_h == 0.05:
        logging.warning("too small weight")
        flag = -1
    location = end_location
    if location[0] < 0:
      factor = (weight_value - 5)*0.1
      location[0] = location[0] + current_r 
      location[2] = location[2] + current_h/2 
    if location[0] > 0:
      location[0] = location[0] - current_r
      location[2] = location[2] + current_h/2 
    bpy.ops.mesh.primitive_cylinder_add(radius=current_r, depth=current_h, location=location)

    weight = bpy.context.object
    weight.name = "Weight"

    # **应用倒角修饰器（Bevel）使边缘更圆润**
    bevel_modifier = weight.modifiers.new(name="Bevel", type='BEVEL')
    bevel_modifier.width = 0.02  # 增加倒角宽度
    bevel_modifier.segments = 16  # 增加细分数，使边缘更圆滑
    bevel_modifier.profile = 0.5  # 设置倒角剖面为圆形

    # **添加细分曲面修饰器（Subdivision Surface）**
    subdivision_modifier = weight.modifiers.new(name="Subdivision", type='SUBSURF')
    subdivision_modifier.levels = 2  # 视口显示细分级别
    subdivision_modifier.render_levels = 2  # 渲染时的细分级别

    # **启用平滑着色**
    bpy.ops.object.shade_smooth()

    # 创建银色金属材质
    mat = bpy.data.materials.new(name="SilverMaterial")
    
  
  
  
    mat.use_nodes = True  # 启用节点系统

    # 获取节点树
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # 清除默认节点
    nodes.clear()

    # 创建节点
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    material_output = nodes.new(type="ShaderNodeOutputMaterial")

    # 设置材质属性
    bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 0.95, 1)  # 接近白色
    bsdf.inputs['Metallic'].default_value = 1.0  # 完全金属
    bsdf.inputs['Roughness'].default_value = 0.2  # 适度的粗糙度

    # 添加噪声纹理来模拟细微的表面不均匀
    noise_texture = nodes.new(type='ShaderNodeTexNoise')
    noise_texture.inputs['Scale'].default_value = 50  # 调整细节级别

    # 添加颜色渐变节点来控制粗糙度的变化
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.color_ramp.interpolation = 'EASE'

    # 设置颜色渐变节点的颜色元素
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.15, 0.15, 0.15, 1)  # 较低的粗糙度
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (0.3, 0.3, 0.3, 1)  # 较高的粗糙度

    # 连接噪声纹理到颜色渐变节点
    links.new(noise_texture.outputs['Fac'], color_ramp.inputs['Fac'])

    # 将颜色渐变的输出连接到 BSDF 的 Roughness 输入
    links.new(color_ramp.outputs['Color'], bsdf.inputs['Roughness'])

    # 连接 BSDF 到材质输出
    links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # 将材质赋予砝码
    weight.data.materials.append(mat)

    return weight, flag


def create_teeter_totter(param = None, camera_location = (0, -7, 2), camera_rotation = (1.2, 0, 0), 
                         camera_focal_length = 25):
    """创建完整的跷跷板模型。"""
    if param is None:
      raise RuntimeError("!!")
    clear_scene()
    create_ground() 
    pivot = create_pivot(radius=param['pivot_r'],
                         location=(0, 0, 1*param['pivot_r']))
    lever = create_lever(
      length=param['lever_length'], width=param["lever_width"], height=param['lever_height'],
      location=(param['lever_x_offset'], 0, 2* param['pivot_r'] + 0.5* param['lever_height']))
    # 设置光标位置为 (1, 1, 1)
    bpy.context.scene.cursor.location = (0, 0, 2* param['pivot_r']+ 0.5* param['lever_height'])

    # 选择对象
    bpy.context.view_layer.objects.active = lever
    lever.select_set(True)

    # 将对象的原点设置为光标位置
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

    # 将杠杆设置为支点的子对象
    bpy.ops.object.select_all(action='DESELECT')
    pivot.select_set(True)
    lever.select_set(True)
    bpy.context.view_layer.objects.active = pivot
    bpy.ops.object.parent_set(type='OBJECT')

    # 在杠杆的两端创建砝码
    weight_left, flag_l = create_weight(end_location = [-param['lever_length']/2 + param['lever_x_offset'],0, 2*param['pivot_r'] + param['lever_height']],
                                weight_value = param['weight_value_l'])
    weight_right, flag_r = create_weight(end_location = [param['lever_length']/2 + param['lever_x_offset'], 0, 2*param['pivot_r'] + param['lever_height']],
                                 weight_value = param['weight_value_r'])
    if flag_l == -1 and flag_r == -1:
        print("both small")
        return -1

    # 将砝码设置为杠杆的子对象
    bpy.ops.object.select_all(action='DESELECT')
    lever.select_set(True)
    weight_left.select_set(True)
    weight_right.select_set(True)
    bpy.context.view_layer.objects.active = lever
    bpy.ops.object.parent_set(type='OBJECT')
    height = 2 * param["pivot_r"]
    angle_radians = get_rotation(height, param['lever_length'], param['lever_x_offset'],param['result'])
    if angle_radians == 0:
      logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

      logging.warning(f"param: {param}")
      return -1
    lever.rotation_euler[0] = 0
    lever.rotation_euler[1] = angle_radians

    create_camera(location= camera_location, rotation = camera_rotation, focal_length = camera_focal_length)
    return 0
  
  
def point_camera_at(camera_obj, target_obj):
    direction = camera_obj.location - target_obj.location 
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()

def render_scene(output_path):
  """渲染当前场景并保存图片。"""
  # 设置渲染引擎为 Cycles
  # bpy.context.scene.render.engine = 'CYCLES'
  bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
  # 设置 Cycles 使用 GPU 设备
  bpy.context.scene.cycles.device = 'GPU'

  # 获取用户首选项中的 Cycles 设置
  prefs = bpy.context.preferences
  cycles_prefs = prefs.addons['cycles'].preferences

  # 设置计算设备类型为 'METAL'
  cycles_prefs.compute_device_type = 'METAL'

  # 刷新设备列表
  cycles_prefs.get_devices()

  # 启用所有可用的 Metal 设备
  for device in cycles_prefs.devices:
      if device.type == 'METAL':
          device.use = True

  # 设置输出路径和图像格式
  bpy.context.scene.render.filepath = output_path
  bpy.context.scene.render.image_settings.file_format = 'PNG'

  # 开始渲染
  bpy.ops.render.render(write_still=True)


def setup_world_environment(hdri_path):
    """设置世界环境光照。"""
    import bpy
    import os

    if os.path.exists(hdri_path):
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # 清除默认节点
        nodes.clear()

        # 创建节点
        env_texture = nodes.new('ShaderNodeTexEnvironment')
        env_texture.image = bpy.data.images.load(hdri_path)
        background = nodes.new('ShaderNodeBackground')
        world_output = nodes.new('ShaderNodeOutputWorld')

        # 连接节点
        links.new(env_texture.outputs['Color'], background.inputs['Color'])
        links.new(background.outputs['Background'], world_output.inputs['Surface'])
    else:
        print("HDRI 文件未找到，使用默认世界设置。")

if __name__ == "__main__":
    # 定义缺失的变量
    radius = 0.3  # 设定支点半径
    lever_x = 1
    weight_r = 0.2
    weight_h = 0.5
    height = 0.1  # 设定杠杆高度
    length = 5  # 设定杠杆长度
    width = 0.7  # 设定杠杆宽度
    create_lever()
    # create_sun(location=(5, -5, 10), strength=5.0)
    setup_world_environment("../scene_support/surrounding/vintage_measuring_lab_4k.exr")
    result = parse_lever_analysis("../scene/lever/lever_analysis_formatted_combined_scenarios.txt")
    print(f"there are {len(result)} scenarios")
    camera_locations = [
      # (0, -7, 2), (0, -7, 4), (0, 7, 2), (0, 7, 4), 
      # (1, -7, 2), (1, -7, 4), (1, 7, 2), (1, 7, 4),
      # (-1, -7, 2), (-1, -7, 4), (-1, 7, 2), (-1, 7, 4),
      (-20, -30, 5), (-3, -6, 4), (-3, 6, 2), (-3, 6, 4),
      (8, 0, 6), (8, 0, 6), (8, 6, 8), 
      (8, 6, 6), 
      (-8, 0, 6), (-8, 0, 6), (-8, 6, 8), (-8, 6, 6), 
                        ]
    camera_rotations = [(1.2, 0, 0)]
    camera_focal_lengths = [15, 25,20,15,]
    for camera_location in camera_locations:
      for camera_focal_length in camera_focal_lengths:
        for i, scene_dict in (enumerate(result)):
            print(scene_dict)
            # print(f"right force: {scene_dict}, right distance: {r_d}, left force: {l_f}, left distance: {l_d}, result: {res}")
            lever_x = ((scene_dict['right_arm'] - scene_dict['left_arm'])/2) /(scene_dict['right_arm'] + scene_dict['left_arm'])
            lever_x = lever_x * length
            print("lever_x", lever_x)
            param = {"pivot_r": radius, "lever_x_offset": lever_x,
                      "lever_height": height, "lever_length": length, "lever_width": width,
                      "weight_r": weight_r, 'weight_h': weight_h, 
                      "weight_value_l": scene_dict['left_force'], "weight_value_r": scene_dict['right_force'],
                      "result": scene_dict['result']}
            status = create_teeter_totter(param, camera_location, camera_rotations[0], camera_focal_length)
            if status == -1:
              print(f"\033[31mthe wrong configuration, and skip it\033[0m")
              continue
            lever = bpy.data.objects["Lever"]

            camera = bpy.data.objects["Camera"]
            point_camera_at(camera, lever)
            render_scene(f"./output/lever_1/output_{camera_location}_{camera_focal_length}_{i}.png")
      print("complete")







import bpy
import math
import sys
import os
import re
import bpy
import math


ground_size = 10
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
        return 0
      angle_radians = math.asin(height/x)
      angle_radians = - angle_radians
    else:
      x = lever_length/2 + (lever_x_offset)
      angle_radians = math.asin(height/x)
      angle_radians = angle_radians
    return angle_radians

def create_ground(location=(0, 0, 0), size=(10, 10, 0.2)):
    bpy.ops.mesh.primitive_cube_add(location=location)
    ground = bpy.context.object
    ground.name = "Ground"
    bevel_width=0.1
    bevel_segments=8

    # 缩放平面到所需尺寸
    ground.scale = (size[0], size[1], size[2] / 2)
    
    # 应用缩放变换
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # 移动立方体，使其上平面位于z=0
    ground.location.z -= size[2] / 2


    # 进入编辑模式
    bpy.ops.object.mode_set(mode='EDIT')

    # 选择所有顶点
    bpy.ops.mesh.select_all(action='SELECT')

    # 对顶点进行倒角操作，创建圆角
    bpy.ops.mesh.bevel(affect='VERTICES')

    # 退出编辑模式
    bpy.ops.object.mode_set(mode='OBJECT')


    # 添加倒角修饰符，创建圆角边缘
    bevel = ground.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = bevel_width  # 圆角的宽度
    bevel.segments = bevel_segments  # 圆角的细分段数，越高越平滑
    bevel.profile = 0.5  # 圆角的形状，0.5是圆形
    
    # 应用倒角修饰符
    bpy.context.view_layer.objects.active = ground
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # 应用平滑着色
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

    # 加载胡桃木纹理（请替换为您的纹理路径）
    base_color_path = '/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture/Wood060_1K-JPG/Wood060_1K-JPG_Color.jpg'  # 替换为实际的颜色贴图路径
    normal_map_path = '/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture//scene_support/texture/Wood060_1K-JPG/Wood060_1K-JPG_NormalDX.jpg'     # 替换为实际的法线贴图路径
    displacement_path = "/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture//scene_support/texture/Wood060_1K-JPG_Displacement.jpg"
    roughness_path = '/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture//scene_support/texture/Wood060_1K-JPG_Roughness.jpg"'
    ao_path = '/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/texture//scene_support/texture/Wood060_1K-JPG/Wood060_1K-JPG_AmbientOcclusion.jpg"'


    if os.path.exists(base_color_path):
        texture_node.image = bpy.data.images.load(base_color_path)
        texture_node.image.colorspace_settings.name = 'sRGB'
    else:
        print(f"未找到颜色贴图：{base_color_path}")
        bsdf.inputs['Base Color'].default_value = (0.6, 0.4, 0.2, 1)  # 默认棕色

    if os.path.exists(normal_map_path):
        normal_map_node.image = bpy.data.images.load(normal_map_path)
        normal_map_node.image.colorspace_settings.is_data = True  # 设置为非颜色数据
    else:
        print(f"未找到法线贴图：{normal_map_path}")
        normal_map_node = None  # 如果没有法线贴图

    # 3. Roughness Map 粗糙度贴图
    if os.path.exists(roughness_path):
        roughness_node = nodes.new(type="ShaderNodeTexImage")
        roughness_node.image = bpy.data.images.load(roughness_path)
        roughness_node.image.colorspace_settings.is_data = True

        links.new(mapping_node.outputs['Vector'], roughness_node.inputs['Vector'])
        links.new(roughness_node.outputs['Color'], bsdf.inputs['Roughness'])
    else:
        print(f"未找到粗糙度贴图：{roughness_path}")

    # 4. Ambient Occlusion Map 环境遮挡贴图
    if os.path.exists(ao_path):
        ao_node = nodes.new(type="ShaderNodeTexImage")
        ao_node.image = bpy.data.images.load(ao_path)
        ao_node.image.colorspace_settings.is_data = True

        # 混合环境遮挡与 Base Color
        mix_rgb_node = nodes.new(type="ShaderNodeMixRGB")
        mix_rgb_node.blend_type = 'MULTIPLY'
        mix_rgb_node.inputs['Fac'].default_value = 1.0  # 完全混合

        links.new(mapping_node.outputs['Vector'], ao_node.inputs['Vector'])
        links.new(ao_node.outputs['Color'], mix_rgb_node.inputs['Color2'])
        links.new(base_color_node.outputs['Color'], mix_rgb_node.inputs['Color1'])
        links.new(mix_rgb_node.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        print(f"未找到环境遮挡贴图：{ao_path}")

    # 5. Displacement Map 位移贴图
    if os.path.exists(displacement_path):
        displacement_node = nodes.new(type="ShaderNodeTexImage")
        displacement_node.image = bpy.data.images.load(displacement_path)
        displacement_node.image.colorspace_settings.is_data = True

        displacement = nodes.new(type="ShaderNodeDisplacement")
        links.new(mapping_node.outputs['Vector'], displacement_node.inputs['Vector'])
        links.new(displacement_node.outputs['Color'], displacement.inputs['Height'])

        # 将位移节点连接到材质输出的位移端口
        links.new(displacement.outputs['Displacement'], material_output.inputs['Displacement'])
    else:
        print(f"未找到位移贴图：{displacement_path}")

    # 设置材质属性
#    bsdf.inputs['Metallic'].default_value = 0.0
#    bsdf.inputs['Roughness'].default_value = 0.4

    # 连接纹理坐标到映射节点
    links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])

    # 连接映射节点到颜色纹理和法线贴图
    links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
    if normal_map_node:
        links.new(mapping_node.outputs['Vector'], normal_map_node.inputs['Vector'])

    # 连接颜色纹理到 BSDF 基础颜色
    links.new(texture_node.outputs['Color'], bsdf.inputs['Base Color'])

    # 如果有法线贴图，连接法线节点
    if normal_map_node:
        links.new(normal_map_node.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

    # 连接 BSDF 到材质输出
    links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # 将材质赋予地面
    ground.data.materials.append(mat)

    # 添加刚体模拟，确保地面是被动物体
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'  # 地面作为被动刚体

    return ground

def create_lever(length=5, width=1, height=0.2, location=(0, 0, 0)):
    """创建更为真实的木质杠杆（木板）。"""
    import bpy
    # 创建木板
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    lever = bpy.context.object
    lever.name = "Lever"
    lever.scale[0] = length  # 调整X轴方向长度
    lever.scale[1] = width   # 调整Y轴方向宽度
    lever.scale[2] = height  # 调整Z轴方向厚度
    
    # 添加倒角修饰器
    bevel_segments = 3
    bevel_width = 0.08
    bevel = lever.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = bevel_width
    bevel.segments = bevel_segments
    bevel.profile = 0.7

    # 应用倒角修饰器
    bpy.context.view_layer.objects.active = lever
    bpy.ops.object.modifier_apply(modifier="Bevel")

    # 创建木材材质
    mat = bpy.data.materials.new(name="LeverMaterial")
    mat.use_nodes = True  # 启用节点系统

    # 获取节点树
    node_tree = mat.node_tree

    # 清除默认节点（可选，但推荐）
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # 创建新的 Principled BSDF 节点和 Material Output 节点
    bsdf = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    material_output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")

    # 设置木材的颜色和纹理参数
    bsdf.inputs['Base Color'].default_value = (0.768533, 0.800023, 0.0236259, 1)
    bsdf.inputs['Roughness'].default_value = 0.9  # 木材表面的粗糙度

    # 使用噪声纹理来模拟木纹效果
    noise_texture = node_tree.nodes.new(type="ShaderNodeTexNoise")
    noise_texture.inputs['Scale'].default_value = 15  # 调整噪声纹理的细节级别

    # 添加“颜色渐变”节点来控制木纹的颜色分布
    color_ramp = node_tree.nodes.new(type="ShaderNodeValToRGB")
    color_ramp.color_ramp.interpolation = 'LINEAR'

    # 配置木纹颜色
    color_ramp.color_ramp.elements[0].color = (0.4, 0.2, 0.05, 1)  # 深色木纹
    color_ramp.color_ramp.elements[1].color = (0.8, 0.5, 0.3, 1)  # 浅色木纹

    # 连接节点
    node_tree.links.new(noise_texture.outputs['Fac'], color_ramp.inputs['Fac'])
    node_tree.links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])

    # 添加凹凸节点（Bump）来增强木材纹理的立体感
    bump_node = node_tree.nodes.new(type="ShaderNodeBump")
    bump_node.inputs['Strength'].default_value = 0.1  # 适中的凹凸强度
    node_tree.links.new(noise_texture.outputs['Fac'], bump_node.inputs['Height'])
    node_tree.links.new(bump_node.outputs['Normal'], bsdf.inputs['Normal'])

    # **关键步骤：将 BSDF 节点连接到 Material Output 节点**
    node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # 将材质赋予木板
    lever.data.materials.append(mat)

    # 添加刚体模拟
    bpy.ops.rigidbody.object_add()
    lever.rigid_body.type = 'PASSIVE'  # 杠杆作为被动刚体

    return lever

def create_pivot(location=(1, 1, 1), radius=1, depth=1):
    """创建平躺的圆柱体支点。"""
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location)
    pivot = bpy.context.object
    pivot.name = "Pivot"
    pivot.rotation_euler[0] = 1.5708  # 绕X轴旋转90度

    # 添加 Bevel 修饰符
    bevel_modifier = pivot.modifiers.new(name="Bevel", type='BEVEL')
    bevel_modifier.width = 0.1  # 凹槽宽度
    bevel_modifier.segments = 10  # 圆角的细分数

    # 应用 Bevel 修饰符（可选，如果需要应用）
    # bpy.context.view_layer.objects.active = pivot
    # bpy.ops.object.modifier_apply(modifier=bevel_modifier.name)

    # 创建材质并启用节点
    mat = bpy.data.materials.new(name="PivotMaterial")
    mat.use_nodes = True  # 启用节点系统

    # 获取节点树
    node_tree = mat.node_tree

    # 清除默认节点（可选，但推荐）
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # 创建节点
    bsdf = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    material_output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    bump_texture = node_tree.nodes.new(type="ShaderNodeTexNoise")
    bump_node = node_tree.nodes.new(type="ShaderNodeBump")

    # 设置节点属性
    bsdf.inputs['Base Color'].default_value = (0.1, 0.1, 0.1, 1)  # 灰黑色
    bsdf.inputs['Roughness'].default_value = 0.4  # 适度的粗糙度
    bump_texture.inputs['Scale'].default_value = 10  # 调整细节级别
    bump_node.inputs['Strength'].default_value = 0.1  # 凹凸强度

    # 连接节点
    node_tree.links.new(bump_texture.outputs['Fac'], bump_node.inputs['Height'])
    node_tree.links.new(bump_node.outputs['Normal'], bsdf.inputs['Normal'])
    node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # 将材质赋予圆柱
    if len(pivot.data.materials) == 0:
        pivot.data.materials.append(mat)  # 如果没有材质，则添加材质
    else:
        pivot.data.materials[0] = mat  # 如果已有材质，替换现有材质

    return pivot

def create_camera(location=(0, -7, 2), rotation = (1.2, 0, 1), focal_length=7):
    """创建相机并设置其位置、朝向和焦距。"""

    # 限制相机的位置，使其不超出地面的边界
    location_x = (location[0]/abs(location[0])) * ground_size/1.5 if abs(location[0]) > ground_size/1.5 else location[0]
    location_y = (location[1]/abs(location[1])) * ground_size/1.5 if abs(location[1]) > ground_size/1.5 else location[1]
    location_z = max(location[-1], 0.25)
    # 使用调整后的相机位置创建相机
    bpy.ops.object.camera_add(location=(location_x, location_y, location_z))

    camera = bpy.context.object
    camera.name = "Camera"
    camera.rotation_euler = rotation
    camera.data.lens = focal_length
    bpy.context.scene.camera = camera

def create_weight(end_location=[0, 0, 0], weight_value=5):
    """创建更为真实的金属砝码圆柱并添加文字。"""
    # 创建砝码圆柱并应用倒角以软化边缘
    ratio_h_r = 2.5
    unit_kg = 5 #5kg
    unit_r = 0.2
    unit_h = 0.5
    unit_volumn = cylinder_volume(unit_r,unit_h)
    current_volumn = (weight_value - unit_kg)*unit_volumn
    current_r, current_h = get_r_h_cylinder(current_volumn,ratio_h_r)
    current_r = 0.02 if current_r < 0.02 else current_r
    current_h = 0.05 if current_h < 0.05 else current_h
    flag = 0
    if current_r == 0.02  and  current_h == 0.05:
        flag = -1
    location = end_location
    if location[0] < 0:
      factor = (weight_value - 5)*0.1
      location[0] = location[0] + current_r 
      location[2] = location[2] + current_h/2 
    if location[0] > 0:
      location[0] = location[0] - current_r
      location[2] = location[2] + current_h/2 
    bpy.ops.mesh.primitive_cylinder_add(radius=current_r, depth=current_h, location=location)

    weight = bpy.context.object
    weight.name = "Weight"

    # **应用倒角修饰器（Bevel）使边缘更圆润**
    bevel_modifier = weight.modifiers.new(name="Bevel", type='BEVEL')
    bevel_modifier.width = 0.02  # 增加倒角宽度
    bevel_modifier.segments = 16  # 增加细分数，使边缘更圆滑
    bevel_modifier.profile = 0.5  # 设置倒角剖面为圆形

    # **添加细分曲面修饰器（Subdivision Surface）**
    subdivision_modifier = weight.modifiers.new(name="Subdivision", type='SUBSURF')
    subdivision_modifier.levels = 2  # 视口显示细分级别
    subdivision_modifier.render_levels = 2  # 渲染时的细分级别

    # **启用平滑着色**
    bpy.ops.object.shade_smooth()

    # 创建银色金属材质
    mat = bpy.data.materials.new(name="SilverMaterial")
    
    mat.use_nodes = True  # 启用节点系统

    # 获取节点树
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # 清除默认节点
    nodes.clear()

    # 创建节点
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    material_output = nodes.new(type="ShaderNodeOutputMaterial")

    # 设置材质属性
    bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 0.95, 1)  # 接近白色
    bsdf.inputs['Metallic'].default_value = 1.0  # 完全金属
    bsdf.inputs['Roughness'].default_value = 0.2  # 适度的粗糙度

    # 添加噪声纹理来模拟细微的表面不均匀
    noise_texture = nodes.new(type='ShaderNodeTexNoise')
    noise_texture.inputs['Scale'].default_value = 50  # 调整细节级别

    # 添加颜色渐变节点来控制粗糙度的变化
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.color_ramp.interpolation = 'EASE'

    # 设置颜色渐变节点的颜色元素
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.15, 0.15, 0.15, 1)  # 较低的粗糙度
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (0.3, 0.3, 0.3, 1)  # 较高的粗糙度

    # 连接噪声纹理到颜色渐变节点
    links.new(noise_texture.outputs['Fac'], color_ramp.inputs['Fac'])

    # 将颜色渐变的输出连接到 BSDF 的 Roughness 输入
    links.new(color_ramp.outputs['Color'], bsdf.inputs['Roughness'])

    # 连接 BSDF 到材质输出
    links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # 将材质赋予砝码
    weight.data.materials.append(mat)

    return weight, flag

def create_teeter_totter(param = None, camera_location = (0, -7, 2), camera_rotation = (1.2, 0, 0), 
                         camera_focal_length = 25):
    """创建完整的跷跷板模型。"""
    if param is None:
      raise RuntimeError("!!")
    clear_scene()
    create_ground() 
    pivot = create_pivot(radius=param['pivot_r'],
                         location=(0, 0, 1*param['pivot_r']))
    lever = create_lever(
      length=param['lever_length'], width=param["lever_width"], height=param['lever_height'],
      location=(param['lever_x_offset'], 0, 2* param['pivot_r'] + 0.5* param['lever_height']))
    # 设置光标位置为 (1, 1, 1)
    bpy.context.scene.cursor.location = (0, 0, 2* param['pivot_r']+ 0.5* param['lever_height'])

    # 选择对象
    bpy.context.view_layer.objects.active = lever
    lever.select_set(True)

    # 将对象的原点设置为光标位置
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

    # 将杠杆设置为支点的子对象
    bpy.ops.object.select_all(action='DESELECT')
    pivot.select_set(True)
    lever.select_set(True)
    bpy.context.view_layer.objects.active = pivot
    bpy.ops.object.parent_set(type='OBJECT')

    # 在杠杆的两端创建砝码
    weight_left, flag_l = create_weight(end_location = [-param['lever_length']/2 + param['lever_x_offset'],0, 2*param['pivot_r'] + param['lever_height']],
                                weight_value = param['weight_value_l'])
    weight_right, flag_r = create_weight(end_location = [param['lever_length']/2 + param['lever_x_offset'], 0, 2*param['pivot_r'] + param['lever_height']],
                                 weight_value = param['weight_value_r'])
    if flag_l == -1 and flag_r == -1:
        print("both small")
        return -1

    # 将砝码设置为杠杆的子对象
    bpy.ops.object.select_all(action='DESELECT')
    lever.select_set(True)
    weight_left.select_set(True)
    weight_right.select_set(True)
    bpy.context.view_layer.objects.active = lever
    bpy.ops.object.parent_set(type='OBJECT')
    height = 2 * param["pivot_r"]
    angle_radians = get_rotation(height, param['lever_length'], param['lever_x_offset'],param['result'])
    if angle_radians == 0:
      return -1
    lever.rotation_euler[0] = 0
    lever.rotation_euler[1] = angle_radians

    create_camera(location= camera_location, rotation = camera_rotation, focal_length = camera_focal_length)
    return 0
def point_camera_at(camera_obj, target_obj):
    direction = camera_obj.location - target_obj.location 
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()


def setup_world_environment(hdri_path):
    """设置世界环境光照。"""
    import bpy
    import os

    if os.path.exists(hdri_path):
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # 清除默认节点
        nodes.clear()

        # 创建节点
        env_texture = nodes.new('ShaderNodeTexEnvironment')
        env_texture.image = bpy.data.images.load(hdri_path)
        background = nodes.new('ShaderNodeBackground')
        world_output = nodes.new('ShaderNodeOutputWorld')

        # 连接节点
        links.new(env_texture.outputs['Color'], background.inputs['Color'])
        links.new(background.outputs['Background'], world_output.inputs['Surface'])
    else:
        print("HDRI 文件未找到，使用默认世界设置。")

if __name__ == "__main__":
    # 定义缺失的变量
    radius = 0.3  # 设定支点半径
    lever_x = 1
    weight_r = 0.2
    weight_h = 0.5
    height = 0.1  # 设定杠杆高度
    length = 5  # 设定杠杆长度
    width = 0.7  # 设定杠杆宽度
    create_lever()
    # create_sun(location=(5, -5, 10), strength=5.0)
    setup_world_environment("/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene_support/surrounding/vintage_measuring_lab_4k.exr")
    result = parse_lever_analysis("/Users/liu/Desktop/school_academy/Case/Yin/causal_inference_dataset/causal_inference/scene/lever/lever_analysis_formatted_combined_scenarios.txt")
    print(f"there are {len(result)} scenarios")
    camera_locations = [(-10, -10, -5)]
    camera_rotations = [(1.2, 0, 0)]
    camera_focal_lengths = [15] # 25,20,15,]
    for camera_location in camera_locations:
      for camera_focal_length in camera_focal_lengths:
        for i, scene_dict in (enumerate(result)):
            print(scene_dict)
            # print(f"right force: {scene_dict}, right distance: {r_d}, left force: {l_f}, left distance: {l_d}, result: {res}")
            lever_x = ((scene_dict['right_arm'] - scene_dict['left_arm'])/2) /(scene_dict['right_arm'] + scene_dict['left_arm'])
            lever_x = lever_x * length
            print("lever_x", lever_x)
            param = {"pivot_r": radius, "lever_x_offset": lever_x,
                      "lever_height": height, "lever_length": length, "lever_width": width,
                      "weight_r": weight_r, 'weight_h': weight_h, 
                      "weight_value_l": scene_dict['left_force'], "weight_value_r": scene_dict['right_force'],
                      "result": scene_dict['result']}
            status = create_teeter_totter(param, camera_location, camera_rotations[0], camera_focal_length)
            if status == -1:
              print(f"\033[31mthe wrong configuration, and skip it\033[0m")
              continue
            lever = bpy.data.objects["Lever"]

            camera = bpy.data.objects["Camera"]
            point_camera_at(camera, lever)
      print("complete")


