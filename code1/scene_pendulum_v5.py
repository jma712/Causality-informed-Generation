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
import argparse
import concurrent.futures
import shutil
from mathutils import Vector

# =============== Original Global Variables ===============
shadow_position_limit = 0
shadow_length_limit = 0
cylinder_name = "Cylinder"
sphere_name = "sphere_1"

# Original values for transformations
original_cylinder_scale = 3.336
original_cylinder_rotation = 0.0  # Assuming no initial rotation
original_sphere_rotation = 0.0    # Assuming no initial rotation

# Scaling factor
original_length = 0.2
sphere_diameter = 0.0075  # (m)

move = 2
x_p, y_p = 0.2 + move, 0.2
z_light = y_l = 0.38
original_cylinder_location = (x_p, 0, 0.2)
original_sphere_location = (x_p, 0, 0.1)

# =============== Functions from Original Script ===============

def render(output_path):
    """
    Render the scene to 'output_path' using a single visible GPU.
    """
    bpy.context.scene.render.engine = 'CYCLES'
    # Force detection of GPU devices
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    # Enable only the first recognized GPU (since CUDA_VISIBLE_DEVICES has hidden others)
    bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True
    bpy.context.scene.cycles.device = 'GPU'

    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.cycles.samples = 3500
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path

    bpy.ops.render.render(write_still=True)

def calculation(x_l, y_l, x_p, y_p, l, theta):
    """
    Shadow calculation logic.
    """
    global shadow_length_limit, shadow_position_limit

    x_shadow_left = y_l * x_p + y_l * l * math.sin(math.radians(theta)) + x_l * l * math.cos(math.radians(theta)) - y_p * x_l
    x_shadow_left = x_shadow_left / (y_p - l * math.cos(math.radians(theta)) - y_l)
    x_shadow_left = -x_shadow_left
  
    x_shadow_right = y_l * x_p - y_p * x_l
    x_shadow_right = x_shadow_right / (y_p - y_l)
    x_shadow_right = -x_shadow_right

    if x_shadow_left > x_shadow_right:
        x_shadow_left, x_shadow_right = x_shadow_right, x_shadow_left
  
    shadow_position = (x_shadow_left + x_shadow_right) / 2
    shadow_length = x_shadow_right - x_shadow_left

    if shadow_position > shadow_position_limit:
        shadow_position_limit = shadow_position
    if shadow_length > shadow_length_limit:
        shadow_length_limit = shadow_length

    noise_position = 0
    noise_length = 0
    return shadow_position + noise_position, shadow_length + noise_length, x_shadow_left, x_shadow_right

def render_scene(output_path="render_output.png"):
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered scene saved to {output_path}")

def apply_transformations(length_scale_factor=None, rotation_angle=None):
    """
    Apply transformations (scale & rotate) to cylinder & sphere.
    """
    cylinder = bpy.data.objects.get(cylinder_name)
    if cylinder:
        z_scale_factor = length_scale_factor
        cylinder.scale[2] = z_scale_factor * original_cylinder_scale
        bpy.context.view_layer.objects.active = cylinder
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    sphere = bpy.data.objects.get(sphere_name)
    if sphere:
        end_cylinder = original_length - (length_scale_factor) * original_length
        sphere.location = (x_p, 0, end_cylinder)
        bpy.context.scene.cursor.location = (x_p, 0, 0.2)
        bpy.context.view_layer.objects.active = sphere
        sphere.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

    for obj_name in [cylinder_name, sphere_name]:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.rotation_euler[1] += math.radians(-rotation_angle)

def reverse_transformations():
    cylinder = bpy.data.objects.get(cylinder_name)
    if cylinder:
        cylinder.scale[2] = original_cylinder_scale
        cylinder.rotation_euler[1] = original_cylinder_rotation
        cylinder.location = original_cylinder_location
        bpy.context.view_layer.objects.active = cylinder
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    sphere = bpy.data.objects.get(sphere_name)
    if sphere:
        bpy.context.view_layer.objects.active = sphere
        sphere.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
        sphere.location = original_sphere_location

def move_object(object_name, new_location):
    obj = bpy.data.objects.get(object_name)
    if obj is not None:
        obj.location = new_location
    else:
        raise ValueError(f"未找到对象 '{object_name}'。")

def move_camera(camera_name, new_location, target_point, focal_length=30.0):
    camera = bpy.data.objects.get(camera_name)
    if camera is None:
        return
    camera.location = new_location
    target_vector = mathutils.Vector(new_location) - mathutils.Vector(target_point)
    camera.rotation_euler = target_vector.to_track_quat('Z', 'Y').to_euler()
    camera.data.lens = focal_length


# =========== New Helper: run_one_iteration ===========

def run_one_iteration(i, file_path, ii = None):
    """
    The code that was originally inside the for-loop. 
    We run this for each iteration i.
    """
    import numpy as np
    
    np.random.seed(i)
    rotation_degree = np.random.uniform(-65, 65)
    length_scale_factor = np.random.uniform(0.1, 1.0)

    x_light = x_l = np.random.uniform(0.05 + move, 0.35 + move)
    length = l = original_length * length_scale_factor

    # Move the objects
    move_object("sun", (x_light, 0.0, z_light))
    move_object("E27 skrew Light Bulb", (x_light, 0.0, z_light + 0.029))

    # Apply transformations
    apply_transformations(length_scale_factor, rotation_degree)

    os.makedirs(f"./database/Real_pendulum_multi", exist_ok=True)
    output_path = f"./database/Real_pendulum_multi/{i}_{ii}.png"

    # Render
    render(output_path)

    # Shadow calculations
    shadow_position, shadow_length, left, right = calculation(
        x_l, y_l, x_p, y_p, l + (sphere_diameter / 2), rotation_degree
    )

    # Reverse transformations
    reverse_transformations()

    # Write data to CSV
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            i, x_l, y_l, x_p, y_p, l, rotation_degree,
            shadow_length, shadow_position, left, right, os.path.basename(output_path)
        ])


# =========== Worker Function for Each Process ===========

def worker(start_iter, end_iter, gpu_id, file_path):
    """
    Each process calls this to:
      1) Limit itself to one GPU (gpu_id).
      2) Open the .blend scene.
      3) Do its chunk of iterations.
    """
    import os
    import sys
    import bpy
    import numpy as np
    
    # Make only one GPU visible
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Open the .blend file
    blend_file_path = "./database/pendulum_scene.blend"  
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    # Ensure scene objects are in correct initial position
    move_object("Plane", (x_p, 0, 0))
    move_object("Cylinder", (x_p, 0, y_p))
    move_object("sphere_1", (x_p, 0, 0.1))
    move_camera(
        camera_name="camera",
        new_location=(x_p, -0.5, 0.3),
        target_point=(x_p, 0.0, 0.18),
        focal_length=25.0
    )
    camera_locatins = [(x_p, -0.5, 0.3), (x_p,  -0.3, 0.5), (x_p+0.2, -0.3, 0.5), (x_p-0.1,  -0.4, 0.4),]
                      #  (x_p-1,  -2, 0.1), (x_p+1, -0.9, 0.1), (x_p+1,  -0.9, 0.1),
                      #  (x_p+1, x_p, 0.4), (x_p+1, x_p, 0.4), (x_p+1, x_p, 0.3),]
    for i in range(start_iter, end_iter+1):
      for ii, camera_locatin in enumerate(camera_locatins):
        if ii in [0, 1, 2, 3]:
          setting_camera(camera_locatin, (x_p, 0.0, 0.18), len_=25)
        else:
          setting_camera(camera_locatin, (x_p, 0.0, 0.18), len_=120)
        run_one_iteration(i, file_path, ii)

    print(f"[GPU {gpu_id}] Finished processing iterations {start_iter}..{end_iter}")


# =========== Main with Parallel Execution ===========

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


def main():
    file_path = "./database/Real_pendulum_multi/tabular.csv"
    # If the directory already exists, remove it for a fresh run
    print(">>>")
    if os.path.exists(file_path):
        shutil.rmtree(os.path.dirname(file_path))

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    header = ['iteration','x_l', "y_l", "x_p", "y_p", "l", "theta",
              "shadow length", "shadow point", 'left', 'right', "image_path"]
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header

    # Total 10000 tasks
    total_iters = 10000
    # We'll split them among 3 GPUs
    # For example:
    #   GPU0 -> 1..3333
    #   GPU1 -> 3334..6666
    #   GPU2 -> 6667..10000
    chunk_size = total_iters // 4

    ranges = [
        (1, chunk_size), 
        (chunk_size+1, chunk_size*2),
        (chunk_size*2+1, chunk_size*3),
        (chunk_size*3+1, total_iters)
    ]

    # GPU IDs
    gpu_ids = [0, 1, 2, 3]  # Adjust if you have different GPU indexing

    # We'll use ProcessPoolExecutor to spawn 3 processes
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for (start_iter, end_iter), gpu_id in zip(ranges, gpu_ids):
            future = executor.submit(worker, start_iter, end_iter, gpu_id, file_path)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Process encountered an error: {e}")

    print("All processes complete. Check the CSV and images in ./database/Real_pendulum.")

# =========== Script Entry Point ===========

if __name__ == "__main__":
    main()