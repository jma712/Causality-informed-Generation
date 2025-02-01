
import argparse
import sys
import os
import numpy as np
import math
import bpy
from mathutils import Vector
from mathutils import Vector, Matrix
import csv
import csv
import multiprocessing as mp

# a = the volume of the ball
# b = the heigh of the cuboid
# c = the base area of the cuboid
# d = the base area of cone


sys.path.append(os.path.abspath('/home/lds/github/Causality-informed-Generation/code1'))
from blender_render import clear_scene, disable_shadows_for_render, load_blend_file_backgournd, set_render_parameters, \
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file, rotate_object_y_axis_by_name
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.9/site-packages/")
import numpy as np


def add_cuboid(height_cuboid, location=(0, 0, 0), 
               edge = None, 
               color=(1.0, 1.0, 1.0, 1.0)):
    """
    Adds a cuboid to the Blender scene with the specified height, location, and color.

    Args:
        height_cuboid (float): Height of the cuboid (Z dimension).
        location (tuple): Location of the cuboid's center (x, y, z).
        color (tuple): RGBA color of the cuboid (default is white).
    """
    # Add the cube (default size = 1x1x1)
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)

    # Get the added object
    cuboid = bpy.context.object
    cuboid.name = "Cuboid"

    # Scale the Z dimension to achieve the desired height
    cuboid.scale[2] = height_cuboid   # Divide by 2 because the default cube is 2x2x2
    cuboid.scale[1] = edge
    cuboid.scale[0] = edge
    # Create a new material for the cuboid
    mat = bpy.data.materials.new(name="CuboidMaterial")
    mat.diffuse_color = color  # Set the RGBA color
    cuboid.data.materials.append(mat)

# def set_render_parameters(resolution=(1920, 1080), file_format='PNG', 
#                           output_path="../database/rendered_image.png", 
#                           samples=500, 
#                           use_denoising=True, use_transparent_bg=False):
#     """设置渲染参数，包括分辨率、格式、输出路径和高质量渲染设置。"""
#     # 设置分辨率和输出路径
#     # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     bpy.context.scene.render.resolution_x = resolution[1]
#     bpy.context.scene.render.resolution_y = resolution[0]
#     bpy.context.scene.render.resolution_percentage = 100
#     bpy.context.scene.render.filepath = output_path
#     bpy.context.scene.render.image_settings.file_format = file_format
    
#     bpy.context.scene.render.engine = 'CYCLES'
#     # bpy.context.scene.render.resolution_percentage = 60
#     bpy.context.scene.cycles.samples = 2800  #渲染时的采样数

#     bpy.context.preferences.addons[
#         "cycles"
#     ].preferences.compute_device_type = "CUDA" # or "OPENCL"

#     # Set the device and feature set
#     bpy.context.scene.cycles.device = "GPU"

#     # get_devices() to let Blender detects GPU device
#     bpy.context.preferences.addons["cycles"].preferences.get_devices()
#     # print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
#     for d in bpy.context.preferences.addons["cycles"].preferences.devices:
#         d["use"] = 1 # Using all devices, include GPU and CPU
#         print(d["name"], d["use"])


def main(
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend",
    csv_file= None,     
    iteration= 0,
    h = 256,
    w = 256,
    with_noise = True
  ):
    clear_scene()
    load_blend_file_backgournd("./database/blank_background_spring.blend")
    
    file_name = f'{iteration}'
    np.random.seed(iteration)
    file_name_ = os.path.join(render_output_path, file_name)
    
    # set_render_parameters(output_path=file_name, resolution=(h, w))
    
    d = base_area_cone = np.random.uniform(0.5, 25)
    radius_cone = np.sqrt(base_area_cone / np.pi)
    
    a = volume_ball =  2 * d
    radius_ball = (3 * volume_ball / (4 * math.pi))**(1/3)
    
    b = height_cuboid = 0.2 * a
    
    c = base_area_cuboid = 0.1 * a + 0.5 * b + 0.1 * d
    edge_cuboid = np.sqrt(base_area_cuboid)
    

    height_cone = 8
    distance = 0.5
  
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius_ball, 
                                         location=(0, 0, radius_ball),
                                         segments=64,  # 水平细分数 (越高越精细)
                                          ring_count=64  # 垂直细分数 (越高越精细)
                                          )
    sphere_object = bpy.context.object  # The newly added sphere becomes the active object
    sphere_object.name = "RedSphere"
    red_material = bpy.data.materials.new(name="RedMaterial")
    red_material.use_nodes = True
    red_bsdf = red_material.node_tree.nodes.get("Principled BSDF")
    if red_bsdf:
        red_bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # Red color with full opacity
    sphere_object.data.materials.append(red_material)
    
    
    add_cuboid(height_cuboid=height_cuboid, 
               location = (radius_ball + edge_cuboid/2 + distance, 0, height_cuboid/2),
               edge = edge_cuboid,
               color=(0.0, 1.0, 0.0, 1.0))  # Add a green cuboid
    
    bpy.ops.mesh.primitive_cone_add(vertices=70, 
                                    radius1=radius_cone, 
                                    depth=height_cone, 
                                    location=(-radius_cone - distance - radius_ball, 0, height_cone / 2))
    cone = bpy.context.object  

    # Create a new material for the cone
    cone_material = bpy.data.materials.new(name="ConeMaterial")
    cone_material.use_nodes = True
    nodes = cone_material.node_tree.nodes
    principled = nodes.get("Principled BSDF")

    # Set the color to blue (RGBA)
    if principled:
        principled.inputs["Base Color"].default_value = (0.0, 0.0, 1.0, 1.0)  # Blue color
    # Assign the material to the cone
    cone.data.materials.append(cone_material)
    # (0, 21, 6)
    target_location = (0, 0, 7)
    camera_locations = [(0, 23, 3), (0, 23, 15), (0, 23, 22), 
                        (-8, 23, 3), (-8, 20, 15), (-8, 20, 20),
                        (8, 23, 3), (8, 23, 15), (8, 23, 20),
                        (17, 20, 3), (19, 20, 15), (10, 20, 20),
                        (27, 20, 3), (29, 20, 15), (28, 20, 20),]
    for it, camera_location in enumerate(camera_locations):
        file_name = os.path.join(render_output_path, file_name_+f"_{it}.png")
        # raise ValueError(file_name)
        
        setting_camera(camera_location, target_location, len_=40)
        # set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
        set_render_parameters(output_path=file_name, resolution=(h, w))
        render_scene()    
        # with open(csv_file, mode="a", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow([iteration, v_ball, high_cylinder, area, os.path.basename(file_name), 
        #                     "basal_area_cone = 0.4 * volumn_ball +  0.7 * height_cylinder" ])
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([iteration, a, b, c, d, os.path.basename(file_name)])

    
  

    # camera_location = (0, 21, 6)
    # setting_camera(camera_location, target_location, len_=40)
    # render_scene()


    # with open(csv_file, mode="a", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow([iteration, a, b, c, d, os.path.basename(file_name)])

def process_task(start_iter, end_iter, render_output_path, csv_file, h, w, gpu_id):
    """
    Function to handle a specific range of iterations on a specific GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign GPU to this process
    for i in range(start_iter, end_iter):
        main(
            render_output_path=render_output_path,
            csv_file=csv_file,
            iteration=i,
            h=h,
            w=w
        )

def main_parallel(arguments, num_gpus, processes_per_gpu, h, w):
    """
    Function to distribute unique subsets of iterations across GPUs.
    """
    iteration_time = arguments.size
    iterations_per_gpu = iteration_time // num_gpus  # Divide iterations among GPUs
    remaining_iterations = iteration_time % num_gpus  # Handle remainder

    # Ensure CSV file and output path exist
    csv_file = f"./database/Hypothetic_V4_linear_full_connected_multi/tabular.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "ID", 'volume_ball', 'height_cuboid', 'base_area_cuboid',
                "base_area_cone",'images_path'
            ])

    render_output_path = f"./database/Hypothetic_V4_linear_full_connected_multi/"
    # Create and start processes
    processes = []
    start_iter = arguments.iter

    for gpu_id in range(num_gpus):
        gpu_iterations = iterations_per_gpu + (1 if gpu_id == num_gpus - 1 else 0)  # Add remainder to the last GPU
        iterations_per_process = gpu_iterations // processes_per_gpu
        remaining_gpu_iterations = gpu_iterations % processes_per_gpu

        for thread_id in range(processes_per_gpu):
            thread_start_iter = start_iter + thread_id * iterations_per_process
            thread_end_iter = thread_start_iter + iterations_per_process

            # Add any remaining iterations for the last process
            if thread_id == processes_per_gpu - 1:
                thread_end_iter += remaining_gpu_iterations

            # Start a process for this GPU and thread
            p = mp.Process(
                target=process_task,
                args=(thread_start_iter, thread_end_iter, render_output_path, csv_file, h, w, gpu_id)
            )
            p.start()
            processes.append(p)

        # Update the start iteration for the next GPU
        start_iter += gpu_iterations

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")
    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--h', type=int, help="resolution of h of the image")
    parser.add_argument('--w', type=int, help="resolution of w of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    # w =  arguments.w
    # h =  arguments.h
    # iteration_time = arguments.size  # 每次渲染的批次数量

    # # CSV 文件路径
    # csv_file = f"./database/Hypothetic_V4_linear/tabular.csv"

    # # 检查文件是否存在
    # if not os.path.exists(csv_file):
    #     with open(csv_file, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["iter", 'volume_ball', "r_ball",'height_cylinder', "r_cylinder", "v_cylinder", 'distance_ball_cylinder', 
    #                          'tile degree','images_path'])

    # # 打开 CSV 文件，追加写入数据
    # with open(csv_file, mode="a", newline="") as file:
    #     writer = csv.writer(file)
    #     scene = "H4"
    #     render_output_path = f"./database/Hypothetic_V4_linear/"
        
    #     # 使用起始帧数循环渲染 iteration_time 个批次
    #     for i in (range(arguments.iter, arguments.iter + iteration_time)):
    #         main(
    #             scene=scene,
    #             render_output_path=render_output_path,
    #             csv_file=csv_file,
    #             iteration=i,
    #             h = h,
    #             w = w
    #         )
    h = arguments.h
    w = arguments.w
    num_gpus = 4  # Number of GPUs available
    processes_per_gpu = 10  # Number of processes per GPU

    # Run parallel execution
    main_parallel(arguments, num_gpus, processes_per_gpu, h, w)





