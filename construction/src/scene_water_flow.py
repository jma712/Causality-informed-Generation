import bpy
import math
import numpy as np
import os
import mathutils
import sys
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.9/site-packages/") 
from tqdm import tqdm
from mathutils import Vector
def projectile_motion(h, v, g=9.8, num_points=80):
    """
    计算平抛运动的轨迹。

    参数:
    - h (float): 平抛运动的初始高度。
    - v (float): 平抛运动的水平速度。
    - g (float): 重力加速度。
    - num_points (int): 轨迹中计算的点数。

    返回:
    - x (numpy.ndarray): 水平位置数组。
    - y (numpy.ndarray): 垂直位置数组。
    """
    t_total = math.sqrt(2 * h / g) # 自由落体时间
    t = np.linspace(0, t_total, num_points) # 时间分割
    
    x = v * t # 水平位移
    y = h - 0.5 * g * t**2 # 垂直位移
    return x, y

def create_material(name="ParabolicMaterial", color=(0.232, 0.614, 0.799, 1.0), transmission=1.0, roughness=0.0, use_transparent=False):
    """
    创建材质并设置颜色、透明度和粗糙度。

    Args:
    name (str): 材质名称。
    color (tuple): 材质颜色 (R, G, B, A)。
    transmission (float): 透明度值 (0.0 到 1.0)。
    roughness (float): 粗糙度值 (0.0 到 1.0)。
    use_transparent (bool): 是否使用 Transparent BSDF 材质。
    """
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True

    nodes = material.node_tree.nodes

    # 清空节点树
    for node in nodes:
        nodes.remove(node)

    # 创建 Transparent BSDF 或 Principled BSDF
    if use_transparent:
        # Transparent BSDF 材质
        transparent_node = nodes.new(type="ShaderNodeBsdfTransparent")
        transparent_node.location = (0, 0)
        transparent_node.inputs["Color"].default_value = color

        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (300, 0)

        # 链接节点
        material.node_tree.links.new(transparent_node.outputs["BSDF"], output_node.inputs["Surface"])
        #print(f"创建 Transparent BSDF 材质 '{name}'，颜色：{color}")
    else:
        # Principled BSDF 材质
        principled = nodes.new(type="ShaderNodeBsdfPrincipled")
        principled.location = (0, 0)
        principled.inputs["Base Color"].default_value = color

        # 设置透明度和粗糙度
        if "Transmission" in principled.inputs:
            principled.inputs["Transmission"].default_value = transmission
        else:
          pass
            #print("未找到 'Transmission' 属性，跳过透明度设置。")
        principled.inputs["Roughness"].default_value = roughness

        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (300, 0)

        # 链接节点
        material.node_tree.links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])
        #print(f"创建 Principled BSDF 材质 '{name}'，颜色：{color}，透明度：{transmission}，粗糙度：{roughness}")

    return material

def create_parabolic_curve(h, v, color=(0.1, 0.5, 1.0, 1.0), curve_thickness=0.05, num_points=80):
    """
    在 Blender 中生成平抛运动的曲线。

    参数:
    - h (float): 初始高度。
    - v (float): 水平速度。
    - color (tuple): 曲线材质的颜色。
    - curve_thickness (float): 曲线的粗细（Bevel Depth）。
    - num_points (int): 曲线点的数量。
    """
    # 计算平抛运动轨迹
    x_vals, y_vals = projectile_motion(h, v, num_points=num_points)

    # 添加贝塞尔曲线
    bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0))
    curve = bpy.context.object
    curve.name = "ParabolicCurve"

    # 删除默认点，添加自定义点
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.curve.select_all(action='SELECT')
    bpy.ops.curve.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # 设置曲线点
    spline = curve.data.splines.new(type='POLY')
    spline.points.add(len(x_vals) - 1)
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        spline.points[i].co = (x, 0.0, y, 1.0)

    # 设置曲线厚度和平滑度
    curve.data.dimensions = '3D'
    curve.data.bevel_depth = curve_thickness
    curve.data.bevel_resolution = 3

    # 添加材质
    material = create_material(name="ParabolicMaterial", transmission=1.0, roughness=0.0)
    curve.data.materials.append(material)

    return x_vals, y_vals

    # 示例: 创建初始高度为10，水平速度为5的平抛曲线
    
def move_object(object_name, axis, distance):
    """
    移动指定对象在指定轴方向上的距离。

    参数:
    - object_name (str): 对象名称。
    - axis (str): 轴方向 ('X', 'Y', 或 'Z')。
    - distance (float): 移动的距离。
    """
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise ValueError(f"对象 '{object_name}' 不存在。")

    # 根据轴方向移动对象
    if axis.upper() == 'X':
        obj.location.x += distance
    elif axis.upper() == 'Y':
        obj.location.y += distance
    elif axis.upper() == 'Z':
        obj.location.z += distance
    else:
        raise ValueError(f"无效的轴方向 '{axis}'，请使用 'X', 'Y', 或 'Z'。")

    #print(f"对象 '{object_name}' 已沿 {axis} 轴移动 {distance} 单位。")

def remove_object(object_name):
    obj = bpy.data.objects.get(object_name)
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)
    else:
        pass
    


def load_scene(path):
  bpy.ops.wm.open_mainfile(filepath=path)
  
def set_render_parameters(resolution=(1920, 1080), file_format='PNG', 
                          output_path=None, circle = False):
    """设置渲染参数，包括分辨率、格式和输出路径。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    bpy.context.scene.eevee.taa_samples = 128*2
    bpy.context.scene.eevee.taa_render_samples = 128*4
    
    if circle:
      bpy.context.scene.render.engine = 'CYCLES'
      bpy.context.scene.cycles.samples = 1500  #渲染时的采样数
      # bpy.context.scene.render.resolution_percentage = 60

      bpy.context.preferences.addons[
          "cycles"
      ].preferences.compute_device_type = "CUDA" # or "OPENCL"

      # Set the device and feature set
      bpy.context.scene.cycles.device = "GPU"

      # get_devices() to let Blender detects GPU device
      bpy.context.preferences.addons["cycles"].preferences.get_devices()
      # #print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
      # for d in bpy.context.preferences.addons["cycles"].preferences.devices:
      #     d["use"] = 1 # Using all devices, include GPU and CPU
      #     #print(d["name"], d["use"])

def save_blend_file(filepath):
    """保存当前场景为指定的 .blend 文件，直接覆盖原有文件。"""
    if os.path.exists(filepath):
        #print('remove the existing file')
        os.remove(filepath)  # 删除已有文件
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    #print(f"修改后的场景已保存到：{filepath}")

def render_scene():
    """执行渲染并保存图像。"""
    bpy.ops.render.render(write_still=True)
    #print(f"渲染完成，图像已保存到：{bpy.context.scene.render.filepath}")
    
def move_camera(camera_name, new_location, target_point, focal_length=30.0):
    camera = bpy.data.objects.get(camera_name)
    if camera is None:
        return
    camera.location = new_location
    target_vector = mathutils.Vector(new_location) - mathutils.Vector(target_point)
    camera.rotation_euler = target_vector.to_track_quat('Z', 'Y').to_euler()
    camera.data.lens = focal_length    

def move_object_to_location(object_name, offset = 0):
    """
    将指定的对象移动到特定的位置。

    参数:
    - object_name (str): 要移动的对象名称。
    - location (tuple): 目标位置 (x, y, z)。
    """
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise ValueError(f"对象 '{object_name}' 不存在。")

    # 移动对象到目标位置
    if object_name == 'sphere':
        x,y,z = get_object_dimensions('sphere')
        #print(">>>>>>>>\n")
        #print(f"z:{z}, offset: {offset}")
        #print(f"z+offset: {z+offset}")
        zz = z/2 + offset
        obj.location = (0,0,zz)
    else:
        obj.location = (0,0,offset)
    # #print(f"对象 '{object_name}' 已移动到位置 {location}。")

def get_object_dimensions(object_name):
    """
    获取指定对象的 X、Y、Z 尺寸。

    参数:
    - object_name (str): 对象名称。

    返回:
    - tuple: 对象的尺寸 (X, Y, Z)。
    """
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise ValueError(f"对象 '{object_name}' 不存在。")
    
    dimensions = obj.dimensions  # 获取对象的尺寸
    return dimensions.x, dimensions.y, dimensions.z

def scale_object_xy(object_name, scale):
    """
    将指定对象的 X 和 Y 尺寸按照比例进行缩放。

    参数:
    - object_name (str): 对象名称。
    - scale (float): 缩放比例（>0）。
    """
    if scale <= 0:
        raise ValueError("缩放比例必须大于0。")
    # 获取对象
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise ValueError(f"对象 '{object_name}' 不存在。")

    # 获取当前缩放
    obj.scale.x = scale  # 缩放 X 尺寸
    obj.scale.y = scale  # 缩放 Y 尺寸

    #print(f"对象 '{object_name}' 的 X 和 Y 已按比例 {scale} 缩放。")

def scale_object_z(object_name, scale):
    """
    将指定对象的 X 和 Y 尺寸按照比例进行缩放。

    参数:
    - object_name (str): 对象名称。
    - scale (float): 缩放比例（>0）。
    """
    if scale <= 0:
        raise ValueError("缩放比例必须大于0。")

    # 获取对象
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise ValueError(f"对象 '{object_name}' 不存在。")

    # 获取当前缩放
    obj.scale.z = scale  # 缩放 X 尺寸


def setting_camera(location, target):
    """
    This function sets the camera location and target.
    The camera's position should be within the range defined by the scene bounds.
    
    Parameters:
    - location: tuple (x, y, z) representing the desired camera position.
    - target: tuple (x, y, z) representing the target point the camera should point at.
    - scene_bounds: tuple of tuples ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                    defining the allowable range for camera positioning.
    """

    # 删除已有的摄像机
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
        bpy.data.objects.remove(camera, do_unlink=True)
        print("Deleted existing camera")

    # 创建新的摄像机
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    camera.name = "Camera"
    print("Created new camera")

    # 设置摄像机朝向目标位置
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # 将摄像机设置为当前场景的活动摄像机
    bpy.context.scene.camera = camera
    print(f"Camera location set to {camera.location}, pointing towards {target}")
    


def main(total, store_dir):
  blend_path = "./database/Water_flow_scene/water_flow.blend"
  load_scene(blend_path)
  step = 50
  max_water_height = 1.28
  bottom_thickness = 0.05
  initial_radium = 0.098723/2
  initial_diameter_x = 1.02994
  initial_diameter_y = 0.921835
  initial_cylinder_scale = 10.669
  initial_ball_scale = 0.05
  initial_water_scale = 0.395
  initial_water_xy = .791
  initial_water_z = .791
  constant_water_height_scale = 0.4
  unit_water_height = 2 #(m)
  buffer = []
  start = total * step
  end = (total + 1) * step
  for iteration in tqdm(range(start, end), desc="Generating scenes"):
    np.random.seed(iteration)
    file_name = f"{iteration:05d}.png"
    tabular = os.path.join(store_dir, f"tabular.csv")
    if not os.path.exists(tabular):
      with open(tabular, 'w') as f:
        f.write("iteration,Ball Volume,Diameter of the bottom,Water height,Hole height,Water length,camera_location,imgs,initial_water_height\n")
    
    
    cup_scale_modify = np.random.uniform(0.5, 2)
    ball_scale_modify = np.random.uniform(0.5, 5)

    
    cup_scale = cup_scale_modify * initial_cylinder_scale
    ball_scale = ball_scale_modify * initial_ball_scale
    
    water_scale =  cup_scale_modify * initial_water_scale

    
    hole_coefficient = np.random.uniform(0.3, 1)
    D = hole_height = unit_water_height * constant_water_height_scale * hole_coefficient
    
    A = V_ball = 4/3 * np.pi * ((initial_radium/initial_ball_scale) * ball_scale)**3 
    B = Diameter_of_the_bottom = initial_diameter_x/initial_cylinder_scale * cup_scale
    
    displace_water_height = 4 * V_ball / (np.pi * (initial_diameter_x/initial_cylinder_scale * cup_scale)**2)
    # #print("displace_water_height: ", displace_water_height)
    initial_water_height = unit_water_height * constant_water_height_scale
    C = water_height  = displace_water_height +initial_water_height
    
    water_scale_h= (water_height/unit_water_height)
    scale_object_xy("water", water_scale)
    scale_object_xy("sphere", ball_scale)
    scale_object_z("sphere", ball_scale)
    # #print(">>>>>>\n")
    scale_object_xy("Cylinder", cup_scale)
    scale_object_z("water", water_scale_h)
    scale_object_xy("water", water_scale)
    if ((initial_diameter_x/2) * cup_scale)-initial_radium * ball_scale < 0.05:
      continue
    
    diff_water_height = water_height - hole_height
    # #print(f"water_height: {water_height}, hole_height: {hole_height}, diff_water_height: {diff_water_height}")
    initial_v = np.sqrt(2*9.8*diff_water_height)
    #print("hole_height:", hole_height)
    #print("h_diff:", diff_water_height)
    #print(f"initial_v: {initial_v}")
    # #print(f"initial_ball_scall: {initial_ball_scale}, ball_scale_modify: {ball_scale_modify}, ball_scale: {ball_scale}")
    # #print(f"new scale ball dimension: {(initial_radium/initial_ball_scale) * ball_scale}")
    # #print(f"initial_cylinder_scale: {initial_cylinder_scale}, cup_scale_modify: {cup_scale_modify}, cup_scale: {cup_scale}")
    # #print(f"new scale cup x,y dimension: {initial_diameter_x/initial_cylinder_scale * cup_scale}, {initial_diameter_y/initial_cylinder_scale * cup_scale}")
    # #print(f"initial_water_scale: {initial_water_scale}, water_scale_h: {water_scale_h}, water_scale: {water_scale}")
    # #print(f"new water dimension: {initial_water_xy / initial_water_scale * water_scale}, {initial_water_z / initial_water_scale * water_scale_h}")
    # #print(f"\ninitial_v :{initial_v}, diff_water_height: {diff_water_height}")
    
    x_vals, y_vals = create_parabolic_curve(h=hole_height, v=initial_v/1, color=(0.1, 0.8, 0.5, 1.0), curve_thickness=0.01)
    E = water_length = x_vals[-1] - x_vals[0]
    assert E == max(x_vals) - min(x_vals), f"water_length: {water_length}, E: {E}, max(x_vals): {max(x_vals)}, min(x_vals): {min(x_vals)}"
    # #print('move curve: ', (cup_scale*initial_diameter_x/initial_cylinder_scale)/2)
    # #print("move hole: ", (cup_scale*initial_diameter_x/initial_cylinder_scale)/2 - 0.414145)
    move_object(object_name="ParabolicCurve", axis='X', distance= (initial_water_xy / initial_water_scale * water_scale)/2 + + 0.05)
    move_object(object_name="hole", axis='X', distance= (initial_water_xy / initial_water_scale * water_scale)/2 + 0.05 - 0.414145)
    move_object(object_name="hole", axis='Z', distance=hole_height)
    # move_object(object_name="sphere", axis='Z', distance=-1
    # move_object(object_name="sphere", axis='Y', distance=1)
    move_object_to_location('sphere', offset=bottom_thickness)
    move_object_to_location('water', offset=bottom_thickness)
    move_camera("camera", new_location=(0.3, -4.5, 1.), target_point=(0.3, 0, 1.), focal_length=50.0)
    
    camera_locations = [(0.86630, -4.3338, 1.68476),
                        (0.8, -4.5, 1), (0.6, -1, 5), 
                        (3.5, -3, 1),(3.5, -1, 3), (3.5, -1, 5), 
                        (-3.5, -3, 2),(-2, -3, 3), (-2, -2, 5), 
                       ]
    for ii, camera_location in enumerate(camera_locations):
      set_render_parameters(resolution=(256, 256), 
                            output_path=os.path.join(store_dir, f"{iteration}_{ii}.png"), 
                            circle = True)
      setting_camera(camera_location, (0.86630, 0, 1))
      set_render_parameters(resolution=(256, 256), 
                            output_path=os.path.join(store_dir, f"{iteration}_{ii}.png"), 
                            circle = True)
      render_scene()



      buffer.append(f"{iteration},{A},{B},{C},{D},{E},{camera_location},{iteration}_{ii}.png,{initial_water_height}\n")
      if len(buffer) >= 5:  # Write every 100 lines
          with open(tabular, 'a') as f:
              f.writelines(buffer)
          buffer = []  # Clear the buffer    

      # with open(tabular, 'a') as f:
      #   f.write(f"{iteration}, {A}, {B}, {C}, {D}, {E}, {file_name},\
      #           {initial_water_height}\n")
    # save_blend_file("./water_flow_debug.blend")
    remove_object(object_name="ParabolicCurve")
    move_object(object_name="hole", axis='X', distance= -((initial_water_xy / initial_water_scale * water_scale)/2 + 0.05 - 0.414145))
    move_object(object_name="hole", axis='Z', distance=-hole_height)
    
  
  
  
if __name__ == "__main__":
  store_dir = "./database/Real_WaterFlow"
  if not os.path.exists(store_dir):
    os.makedirs(store_dir, exist_ok=True)
  else:
    os.system(f"rm -rf {store_dir}/*")
    os.makedirs(store_dir, exist_ok=True)
  for i in range(0, 1):
    main(i, store_dir)
    raise ValueError("Stop here")

