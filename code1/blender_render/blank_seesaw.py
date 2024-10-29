import bpy
import os
import argparse
import sys
import random
from mathutils import Vector
import math

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
    
    
def load_blend_file(filepath):
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

def create_lever(length=5, width=1, height=0.2, location=(0, 0, 0)):
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

    # 添加刚体模拟
    bpy.ops.rigidbody.object_add()
    lever.rigid_body.type = 'PASSIVE'  # 杠杆作为被动刚体

    # 添加材质并调整反射率
    mat = bpy.data.materials.new(name="WoodMaterial")
    mat.use_nodes = True
    lever.data.materials.append(mat)

    # 获取 Principled BSDF 节点
    principled_bsdf = mat.node_tree.nodes.get("Principled BSDF")
    
    if principled_bsdf:
        # 设置材质参数，使其更接近木材
        principled_bsdf.inputs["Base Color"].default_value = (0.5, 0.3, 0.1, 1)  # 棕色
        principled_bsdf.inputs["Metallic"].default_value = 0.0  # 非金属
        principled_bsdf.inputs["Specular"].default_value = 0.1  # 低反射
        principled_bsdf.inputs["Roughness"].default_value = 1  # 增加粗糙度，减少反光

    return lever
  
def delete_object_by_name(object_name):
    """
    删除指定名称的对象。
    
    Parameters:
    - object_name: 要删除的对象名称 (str)
    """
    # 检查对象是否存在
    obj = bpy.data.objects.get(object_name)
    if obj is not None:
        # 选择并删除对象
        bpy.data.objects[object_name].select_set(True)  # 选择对象
        bpy.ops.object.delete()                         # 删除对象
        print(f"对象 '{object_name}' 已删除。")
    else:
        print(f"对象 '{object_name}' 不存在。")
  
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
    """删除当前场景中的所有对象。"""
    bpy.ops.object.select_all(action='SELECT')  # 选择所有对象
    bpy.ops.object.delete()  # 删除选中的对象
    print("清空场景完成。")

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
        # logging.warning("too small weight")
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

    return weight, flag

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

def calculate(length = 5) -> dict:
  random_offset = random.uniform(0, 1.5)
  random_left_weight = random.uniform(10, 60)
  random_right_weight = random.uniform(10, 60)
  mass = 5
  left_arm = length/2 - random_offset
  right_arm = length/2 + random_offset
  left_mass = mass * (left_arm / length)
  right_mass = mass * (right_arm / length)
  left = random_left_weight * left_arm  + left_mass * left_arm/2
  right = random_right_weight * right_arm + right_mass * right_arm/2
  
  if left > right:
    result = "left"
  elif left < right:
    result = "right"
  else:
    result = "balance"

  param = {
          "pivot_r": 0.2,
          "lever_length": 5,
          "lever_width": 1,
          "lever_height": 0.2,
          "lever_x_offset": random_offset,
          "weight_value_l": random_left_weight,
          "weight_value_r": random_right_weight,
          "result": result
      }


  return param

def add_seesaw(param = None, camera_location = (0, -7, 2), camera_rotation = (1.2, 0, 0), 
                         camera_focal_length = 25):
    """创建完整的跷跷板模型。"""
    param = calculate()

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
    angle_radians = get_rotation(height, param['lever_length'], param['lever_x_offset'], param['result'])
    if angle_radians == 0:
      logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

      logging.warning(f"param: {param}")
      return -1
    lever.rotation_euler[0] = 0
    lever.rotation_euler[1] = angle_radians
    return 0
  

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
    load_blend_file(background)
    # 3. 根据 `scene` 参数添加不同的对象
    if scene.lower() == "seesaw":
        add_seesaw()
    else:
        print(f"未识别的场景类型: {scene}，跳过特定元素添加。")

    # 4. 设置渲染参数
    set_render_parameters(output_path=render_output_path)

    target_location = (0, 0, 1)
    range_min = -15
    range_max = 15
    range_max_z = 10
    range_min_z = 0.5
    camera_location = (random.uniform(range_min, range_max), random.uniform(range_min, range_max), random.uniform(range_min_z, range_max_z))
    setting_camera(camera_location, target_location)
    render_scene()

    if save_path:
        save_blend_file(save_path)



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