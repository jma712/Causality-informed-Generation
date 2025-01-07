import bpy
import math
import mathutils
import uuid
import os
import csv
import random
import sys
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.9/site-packages/")
from tqdm import tqdm
import numpy as np


shadow_position_limit = 0
shadow_length_limit = 0

def rotate_object(object_name, axis, degrees):
    """
    Rotate an object around a specified axis by a given degree.
    
    :param object_name: str, the name of the object to rotate
    :param axis: str, the axis to rotate around ('X', 'Y', 'Z')
    :param degrees: float, the angle to rotate in degrees
    """
    # Get the object
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        # print(f"Object '{object_name}' not found.")
        return

    # Convert degrees to radians
    radians = math.radians(degrees)
    
    # Rotate the object based on the specified axis
    if axis.upper() == 'X':
        obj.rotation_euler[0] += radians
    elif axis.upper() == 'Y':
        obj.rotation_euler[1] += radians
    elif axis.upper() == 'Z':
        obj.rotation_euler[2] += radians
    else:
        #print(f"Invalid axis '{axis}'. Use 'X', 'Y', or 'Z'.")
        return


def move_object(object_name, new_location):
    """
    将对象移动到指定位置。
    :param object_name: str，对象名称
    :param new_location: tuple，目标位置 (x, y, z)
    """
    obj = bpy.data.objects.get(object_name)
    if obj is not None:
        obj.location = new_location
        # print(f"对象 '{object_name}' 已移动到位置 {new_location}。")
    else:
        raise ValueError(f"未找到对象 '{object_name}'。")
        #print(f"未找到对象 '{object_name}'。")

def calculate_cylinder_height(object_name):
    """
    计算 Cylinder 的高度并调整其位置。
    :param object_name: str，Cylinder 对象名称
    """
    obj = bpy.data.objects.get(object_name)
    if obj is not None and obj.type == 'MESH':
        mesh = obj.data
        z_coords = [v.co.z for v in mesh.vertices]  # 获取所有顶点的 Z 坐标
        height = max(z_coords) - min(z_coords)
        obj.location = (0, 0, height / 2)  # 调整 Cylinder 的位置
        #print(f"对象 '{object_name}' 的高度为: {height}")
    else:
      raise ValueError(f"对象 '{object_name}' 不存在或不是网格对象。")
        #print(f"对象 '{object_name}' 不存在或不是网格对象。")

def move_camera(camera_name, new_location, target_point, focal_length=30.0):
    """
    将相机移动到指定位置，并对准目标点。
    :param camera_name: str，相机的名称
    :param new_location: tuple，目标位置 (x, y, z)
    :param target_point: tuple，对准的目标点 (x, y, z)
    :param focal_length: float，相机焦距
    """
    camera = bpy.data.objects.get(camera_name)
    if camera is None:
        #print(f"未找到名为 '{camera_name}' 的相机。")
        return
    camera.location = new_location
    target_vector = mathutils.Vector(new_location) - mathutils.Vector(target_point)
    camera.rotation_euler = target_vector.to_track_quat('Z', 'Y').to_euler()
    camera.data.lens = focal_length
    #print(f"相机 '{camera_name}' 已移动到 {new_location}，并对准目标点 {target_point}。")

def render(output_path):# Set output path

  # Set render engine (e.g., CYCLES, BLENDER_EEVEE)
  bpy.context.scene.render.engine = 'CYCLES'
  # using only one visible GPU
  bpy.context.preferences.addons['cycles'].preferences.get_devices()

  bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
  bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True
  bpy.context.scene.cycles.device = 'GPU'

  # Set resolution
  bpy.context.scene.render.resolution_x = 256
  bpy.context.scene.render.resolution_y = 256
  bpy.context.scene.render.resolution_percentage = 100

  # Set output file format
  bpy.context.scene.render.image_settings.file_format = 'PNG'

  # Set the output file path
  bpy.context.scene.render.filepath = output_path

  # Render the scene and save it
  bpy.ops.render.render(write_still=True)

  # #print(f"Rendered image saved to: {output_path}")
  
def calculation(x_l, y_l, x_p, y_p, l, theta):
  x_shadow_left = y_l * x_p + y_l * l * math.sin(math.radians(theta)) + x_l * l * math.cos(math.radians(theta)) - y_p * x_l
  x_shadow_left = x_shadow_left / (y_p - l * math.cos(math.radians(theta)) - y_l)
  x_shadow_left = -x_shadow_left
  
  x_shadow_right = y_l * x_p - y_p * x_l
  x_shadow_right = x_shadow_right / (y_p - y_l)
  x_shadow_right = -x_shadow_right
  
  # if x_shadow_left > x_shadow_right, replace them
  if x_shadow_left > x_shadow_right:
    x_shadow_left, x_shadow_right = x_shadow_right, x_shadow_left
  
  shadow_position = (x_shadow_left+x_shadow_right)/2
  shadow_length = x_shadow_right - x_shadow_left
  
  global shadow_length_limit, shadow_position_limit
  if shadow_position > shadow_position_limit:
    shadow_position_limit = shadow_position
  if shadow_length > shadow_length_limit:
    shadow_length_limit = shadow_length
  
  noise_position = np.random.randn() * 0.001 * 2.42150188818041
  noise_length = np.random.randn() * 0.001 * 2.111104481863011
  
  return shadow_position+noise_position, shadow_length+noise_length, x_shadow_left, x_shadow_right


if __name__ == "__main__":
  # Path to the .blend file
  blend_file_path = "./database/pendulum_real.blend"  # Replace with the actual file path
  # Open the .blend file (this replaces the current scene)
  bpy.ops.wm.open_mainfile(filepath=blend_file_path)
  # csv log file, to store the data, for example, the position of the pendulum
  header = ['iteration','x_l', "y_l", "x_p", "y_p", "l", "theta", "shadow length", "shadow point", 'left', 'right', "image_path"]
  file_path = "./database/real_pendulum/tabular.csv"
  if os.path.exists(file_path):
    # remove the whole directory
    import shutil
    shutil.rmtree(os.path.dirname(file_path))

    # raise ValueError(f"File '{file_path}' already exists.")
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))
  with open(file_path, mode='w', newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      writer.writerow(header)  # Write the header
  # set random seed
  
  random.seed(42)
  for i in tqdm(range(1, 10_000), desc="Generating images"):

    rotation_point = 0.2

    move = 1
    x_p, y_p = 0.2 + move, 0.2
    z_light = y_l = 0.38
    x_light = x_l  = random.uniform(0.05 + move, 0.35+move)
    move_object("Plane", (x_p, 0, 0))
    # get x_l from range_x_l
    
    length = l = random.uniform(0.05, 0.115)
    rotation_degree = random.uniform(-60, 60)
    move_object("cylinder", (x_p, 0, y_p))


    move_object("sun", (x_light, 0.0, z_light))


    # 调整相机位置并对准目标
    move_camera(
        camera_name="camera",            # 相机名称
        new_location=(x_p, -0.5, 0.3),  # 相机新位置
        target_point=(x_p, 0.0, 0.18),  # 拍摄目标点
        focal_length=25.0               # 相机焦距
    )

    # 移动父对象 E27 skrew Light Bulb
    move_object("E27 skrew Light Bulb", (x_light, 0.0, z_light + 0.029))

    rotate_object(
        object_name="cylinder",  # Replace with the name of your object
        axis="Y",               # Axis to rotate around ('X', 'Y', or 'Z')
        degrees=-rotation_degree           # Degrees to rotate
    )
    # render image with the current settings
    # generate a uniqe file name:
    output_path = f"./database/real_pendulum/{uuid.uuid4().hex}.png"
    render(output_path)
    shadow_position, shadow_length, left, right = calculation(x_l, y_l, x_p, y_p, l, rotation_degree)
    # write the data to the csv file
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([i, x_l, y_l, x_p, y_p, l, rotation_degree, shadow_length, shadow_position, left, right, output_path])
    rotate_object(
        object_name="cylinder",  # Replace with the name of your object
        axis="Y",               # Axis to rotate around ('X', 'Y', or 'Z')
        degrees=rotation_degree           # Degrees to rotate
    )
        
        
    