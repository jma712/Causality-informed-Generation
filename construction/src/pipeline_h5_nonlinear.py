
import argparse
import sys
import os
import random
import math
import bpy
from mathutils import Vector
from mathutils import Vector, Matrix
import csv
from datetime import datetime
import csv
import mathutils
import multiprocessing as mp

# a = the volume of the ball
# b = the height of the cylinder
# c = the distance between the ball and the cylinder
# d = the cylinder’s height above the ground
# e = the tilt angle of the cylinder

sys.path.append(os.path.abspath('/home/lds/github/Causality-informed-Generation/code1'))
from blender_render import clear_scene, disable_shadows_for_render, load_blend_file_backgournd, set_render_parameters, \
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file, rotate_object_y_axis_by_name
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.9/site-packages/")
import numpy as np



def rotate_object_around_custom_axis(obj, pivot_point, angle):
    """
    Rotates the given object around a custom pivot point along the Y-axis.

    Parameters:
        obj (bpy.types.Object): The Blender object to rotate.
        pivot_point (tuple): The (x, y, z) coordinates of the custom pivot point.
        angle (float): The rotation angle in degrees.
    """
    # Ensure the object is active and selected
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Set the 3D cursor to the custom pivot point
    bpy.context.scene.cursor.location = Vector(pivot_point)
    
    # Set the object's origin to the 3D cursor (pivot point)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    # Convert angle to radians
    radians = math.radians(angle)
    
    # Rotate the object around Y-axis
    obj.rotation_euler[1] += radians  # Y-axis corresponds to index 1 in Euler rotation

    print(f"Rotated object '{obj.name}' around Y-axis by {angle} degrees using pivot {pivot_point}.")

def are_objects_in_camera_view(camera):
    """
    Check if all objects in the scene are fully visible in the camera's view.
    
    Parameters:
        camera (bpy.types.Object): The camera object.
    
    Returns:
        bool: True if all objects are fully visible, False otherwise.
    """
    scene = bpy.context.scene

    # Ensure the camera is active
    scene.camera = camera

    for obj in bpy.data.objects:
        if obj.type not in {'MESH', 'CURVE', 'SURFACE', 'META'}:
            continue  # Skip non-visible objects like lights, cameras, etc.

        # Get the object's bounding box in world coordinates
        obj_matrix = obj.matrix_world
        bbox_world = [obj_matrix @ mathutils.Vector(corner) for corner in obj.bound_box]
        
        for corner in bbox_world:
            # Project each corner to the camera's view
            co_ndc = world_to_camera_view(scene, camera, corner)

            # Check if the corner is outside the normalized device coordinates (NDC)
            if not (0.0 <= co_ndc[0] <= 1.0 and 0.0 <= co_ndc[1] <= 1.0 and co_ndc[2] > 0.0):
                print(f"Object '{obj.name}' is partially or completely outside the view.")
                return False

    print("All objects are fully visible in the camera view.")
    return True

def world_to_camera_view(scene, cam, coord):
    """
    Project a 3D world coordinate to the camera's view space (NDC).
    
    Parameters:
        scene (bpy.types.Scene): Current scene.
        cam (bpy.types.Object): Camera object.
        coord (mathutils.Vector): 3D coordinate in world space.
    
    Returns:
        mathutils.Vector: Normalized device coordinates (x, y, z).
    """
    # Transform the coordinate into the camera's view space
    camera_matrix = cam.matrix_world.normalized().inverted()
    coord_camera = camera_matrix @ coord
    
    # Get perspective projection matrix
    camera_data = cam.data
    if camera_data.type != 'PERSP' and camera_data.type != 'ORTHO':
        raise ValueError("Unsupported camera type. Only perspective and orthographic are supported.")

    proj_matrix = camera_data.compute_viewplane(scene)
    co_ndc = proj_matrix @ coord_camera

    # Normalize
    return mathutils.Vector((co_ndc.x / co_ndc.w, co_ndc.y / co_ndc.w, co_ndc.z))

def set_render_parameters(resolution=(1920, 1080), file_format='PNG', 
                          output_path="../database/rendered_image.png", 
                          samples=500, 
                          use_denoising=True, use_transparent_bg=False):
    """设置渲染参数，包括分辨率、格式、输出路径和高质量渲染设置。"""
    # 设置分辨率和输出路径
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    
    bpy.context.scene.render.engine = 'CYCLES'
    # bpy.context.scene.render.resolution_percentage = 60
    bpy.context.scene.cycles.samples = 2800  #渲染时的采样数

    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

def main(
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend",
    csv_file= None,     
    iteration= 0,
    h = 256,
    w = 256,
    with_noise = True
  ):
    """
    Variable a = the volume of a ball;
    Variable b = the height of a cylinder;
    Variable c = the distance between the ball and the cylinder;
    Variable d = the cylinder’s height above the ground;
    Variable e = the tilt angle of the cylinder.
    """
    clear_scene()
    unique_filename = f"{iteration}"
    file_name = os.path.join(render_output_path, unique_filename+".png")
    
    background = "./database/background_magnet.blend"
    load_blend_file_backgournd(background)
    np.random.seed(iteration)

    set_render_parameters(output_path=file_name, resolution=(h, w))
    camera_location = (0, 23, 4)

    v_ball = a = np.random.uniform(1, 23)
    height_above = d = np.random.uniform(0.5, 8)
    
    # Rescale d (height_above) from [0.5, 8] to [0.2, 3.14]
    scaled_d = 0.2 + ((height_above - 0.5) * (3.14 - 0.2)) / (8 - 0.5)
    # b = np.sin(d) 
    height_c = b = np.sin(scaled_d)  # Height of the cylinder based on the scaled value
    
    # Rescale a (v_ball) from [1, 10] to [0, π/2]
    scaled_a = 0 + ((a - 1) * (np.pi / 2 - 0)) / (23 - 1)
    # c = 2con(a) + 7b
    distance = c = 2 * np.cos(scaled_a) + 7 * b
    # e = tan(c) + 2d
    angle = e = 4 * (np.tan(c) + 2 * d)
    
    r = r_ball = (3 * v_ball / (4 * math.pi))**(1/3)
    v_cylinder = 8
    r_cylinder = (v_cylinder / (math.pi * height_c))**(1/2)
    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0, 0, r))
    sphere_object = bpy.context.object  # The newly added sphere becomes the active object
    sphere_object.name = "RedSphere"

    # Create a red material and assign it to the sphere
    red_material = bpy.data.materials.new(name="RedMaterial")
    red_material.use_nodes = True
    red_bsdf = red_material.node_tree.nodes.get("Principled BSDF")
    if red_bsdf:
        red_bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # Red color with full opacity
    sphere_object.data.materials.append(red_material)

    # Add a cylinder and set its color to yellow
    bpy.ops.mesh.primitive_cylinder_add(
        radius=r_cylinder, 
        depth=b, 
        location=(float(r + r_cylinder + c), 0, float(b/2 + d))
    )
    cylinder_object = bpy.context.object  # The newly added cylinder becomes the active object
    cylinder_object.name = "GreenCylinder"

    # Create a yellow material and assign it to the cylinder
    yellow_material = bpy.data.materials.new(name="GreenMaterial")
    yellow_material.use_nodes = True
    yellow_bsdf = yellow_material.node_tree.nodes.get("Principled BSDF")
    if yellow_bsdf:
        yellow_bsdf.inputs["Base Color"].default_value = (0.0, 1.0, 0.0, 1.0) # Yellow color with full opacity
    cylinder_object.data.materials.append(yellow_material)

    obj = bpy.context.object
    
    
    # Rename the object to the specified name
    obj.name = 'Cylinder'

    original_x = r + r_cylinder + c + 2 * r_cylinder 
    original_y = 0
    original_z = d
    
    rotate_object_y_axis_by_name("Cylinder", e)
    
    # obj = load_blend_file('./database/rect_hyp.blend', 
    #                 location=(-r_ball - 0.8 - 0.2, 0, epsilon_1), scale=(1, 1, 1), rotation_angle=0)
  
    target_location = ((r + r_cylinder*2 + c) /3, 0, 4)
    camera_h = random.uniform(4, 4)
    camera_location = ((r + r_cylinder*2 + c) /3, 20, camera_h)
    setting_camera(camera_location, target_location, len_=35 )
    # Example Usage:
    camera = bpy.data.objects['Camera']  # Replace 'Camera' with your actual camera name
    # if are_objects_in_camera_view(camera):
    render_scene()
    

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, a, b, c, d, e, os.path.basename(file_name)])
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--h', type=int, help="resolution of h of the image")
    parser.add_argument('--w', type=int, help="resolution of w of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    w =  arguments.w
    h =  arguments.h
    iteration_time = arguments.size  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = f"./database/Hypothetic_V5_nonlinear_{h}x{w}/tabular.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", 'volume_ball', 'height_cylinder', 'distance_ball_cylinder', 
                             'height_cylinder_above_ground', 'tilt_angle', 'image_path'])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        render_output_path = f"./database/Hypothetic_V5_nonlinear_{h}x{w}/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in (range(arguments.iter, arguments.iter + iteration_time)):
            main(
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                h = h,
                w = w
            )


# def process_task(start_iter, batch_size, render_output_path, csv_file, h, w, gpu_id):
#     """
#     A function to handle rendering for a specific GPU.
#     """
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Set the GPU visible for this process
#     with open(csv_file, mode="a", newline="") as file:
#         writer = csv.writer(file)
#         for i in range(start_iter, start_iter + batch_size):
#             main(
#                 render_output_path=render_output_path,
#                 csv_file=csv_file,
#                 iteration=i,
#                 h=h,
#                 w=w
#             )

# def main_parallel(arguments, num_gpus, processes_per_gpu, h, w):
#     """
#     Function to manage parallel rendering across multiple GPUs.
#     """
#     total_processes = num_gpus * processes_per_gpu
#     iteration_time = arguments.size  # Total iterations to process
#     batch_size = iteration_time // total_processes  # Number of iterations per process

#     # Ensure output paths exist
#     csv_file = f"./database/Hypothetic_V5_nonlinear_{h}x{w}/tabular.csv"
#     os.makedirs(os.path.dirname(csv_file), exist_ok=True)

#     # Initialize the CSV file if it doesn't exist
#     if not os.path.exists(csv_file):
#         with open(csv_file, mode="w", newline="") as file:
#             writer = csv.writer(file)
#             writer.writerow(["iter", 'volume_ball', 'height_cylinder', 'distance_ball_cylinder', 
#                              'height_cylinder_above_ground', 'tilt_angle', 'image_path'])

#     render_output_path = f"./database/Hypothetic_V5_nonlinear_{h}x{w}/"

#     # Create and start processes
#     processes = []
#     for gpu_id in range(num_gpus):
#         for thread_id in range(processes_per_gpu):
#             start_iter = arguments.iter + (gpu_id * processes_per_gpu + thread_id) * batch_size
#             p = mp.Process(
#                 target=process_task,
#                 args=(start_iter, batch_size, render_output_path, csv_file, h, w, gpu_id)
#             )
#             p.start()
#             processes.append(p)

#     # Wait for all processes to complete
#     for p in processes:
#         p.join()

# if __name__ == "__main__":
#     # Parse arguments
#     parser = argparse.ArgumentParser(description="Blender Rendering Script")

#     parser.add_argument("--iter", type=int, help="initial number")
#     parser.add_argument("--size", type=int, help="size of each iteration")
#     parser.add_argument('--h', type=int, help="resolution of h of the image")
#     parser.add_argument('--w', type=int, help="resolution of w of the image")

#     arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
#     w =  arguments.w
#     h =  arguments.h
    
#     num_gpus = 4  # Number of GPUs available
#     processes_per_gpu = 10  # Number of processes per GPU

#     # Run parallel processing
#     main_parallel(arguments, num_gpus, processes_per_gpu, h, w)




