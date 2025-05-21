
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


# a = volume of the ball
# b = height of the cylinder
# c = the base are of the cone
# epsilan = noise: the height of the rectangular prism above the ground

# c = tan(a) + 0.7 b 

sys.path.append(os.path.abspath('/home/lds/github/Causality-informed-Generation/code1'))
from blender_render import clear_scene, disable_shadows_for_render, load_blend_file_backgournd, set_render_parameters, \
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file, rotate_object_y_axis_by_name
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.9/site-packages/")
import numpy as np


def main(
    render_output_path = "../database/rendered_image.png",
    csv_file= None,     
    iteration= 0,
    resolution = None,
    # lock = None
  ):
    
    clear_scene()
    unique_filename = f"{iteration}"
    file_name_ = os.path.join(render_output_path, unique_filename)
    
    np.random.seed(iteration)
    background = "./database/blank_background_spring.blend"
    load_blend_file_backgournd(background)

    # set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
    
    # randomly generate r from 0.5 to 15
    v_ball = np.random.uniform(0.175, 3.5)
    r_ball = (3 * v_ball / (4 * math.pi))**(1/3)    
    
    original_min, original_max = 0.175, 3.5
    new_min, new_max = 0.1, 1.5
    v_scaled = new_min + ((v_ball - original_min) * (new_max - new_min)) / (original_max - original_min)

    
    # v_scaled = 0 + ((v_ball - 0.175) * (math.pi / 2 - 0)) / (1.35 - 0.175)
    
    high_cylinder = np.random.uniform(0.2, 3.3)
    # 
    basal_area_cone = np.tan(v_scaled) +  0.7 * high_cylinder
    radius1 = (basal_area_cone / math.pi)**(1/2)
    r_cylinder =  0.8 #(v_cylinder / (math.pi * high_cylinder))**(1/2)
    
    # blender generate ball based on r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r_ball, location=(0, 0, r_ball))
    sphere = bpy.context.object  # 获取新创建的球体对象
    material_sphere = bpy.data.materials.new(name="SphereMaterial")
    material_sphere.use_nodes = True
    bsdf = material_sphere.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)  # 设置为红色 (R, G, B, Alpha)
    sphere.data.materials.append(material_sphere)

    # 添加圆柱体并设置颜色
    bpy.ops.mesh.primitive_cylinder_add(radius=r_cylinder, depth=high_cylinder, 
                                        location=(r_ball + r_cylinder + 0.2, 0, high_cylinder/2))
    cylinder = bpy.context.object  # 获取新创建的圆柱体对象
    material_cylinder = bpy.data.materials.new(name="CylinderMaterial")
    material_cylinder.use_nodes = True
    bsdf = material_cylinder.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0, 1, 0, 1)  # 设置为绿色
    cylinder.data.materials.append(material_cylinder)

    # 添加锥体并设置颜色
    bpy.ops.mesh.primitive_cone_add(
        vertices=32,              
        radius1=radius1,          
        radius2=0.0,              
        depth=3.0,                
        location=(-r_ball - radius1 - 0.2, 0, 1.5),  
        scale=(1, 1, 1)
    )
    cone = bpy.context.object  # 获取新创建的锥体对象
    material_cone = bpy.data.materials.new(name="ConeMaterial")
    material_cone.use_nodes = True
    bsdf = material_cone.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0, 0, 1, 1)  # 设置为蓝色
    cone.data.materials.append(material_cone)
    
    center = (r_ball + 1.8 * radius1 ) - (0.2 + 2 * r_cylinder + r_ball)
    target_location = (-center, 0, 2)
    # camera_location = (-center, random.uniform(20, 20), random.uniform(4.2, 4.2))
    # setting_camera(camera_location, target_location, len_=60)
    # render_scene()
    
    
    
    camera_locations = [(0, 23, 3), (0, 23, 15), (0, 23, 22), 
                        (-8, 23, 3), (-8, 20, 15), (-8, 20, 20),
                        (8, 23, 3), (8, 23, 15), (8, 23, 20),
                        (17, 20, 3), (19, 20, 15), (10, 20, 20),
                        (27, 20, 3), (29, 20, 15), (28, 20, 20),]
    for it, camera_location in enumerate(camera_locations):
        file_name = os.path.join(render_output_path, file_name_+f"_{it}.png")
        # raise ValueError(file_name)
        
        setting_camera(camera_location, target_location, len_=90)
        set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
        render_scene()    

        area = radius1 ** 2 * math.pi
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([iteration, v_ball, high_cylinder, area, os.path.basename(file_name), 
                            "basal_area_cone = 0.4 * volumn_ball +  0.7 * height_cylinder" ])


    
    
    
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, v_ball, v_scaled,r_ball, high_cylinder, radius1, basal_area_cone, os.path.basename(file_name), 
                         "basal_area_cone = np.tan(volumn_ball) +  0.7 * height_cylinder" ])
    # if lock:
    #     with lock:
    #         with open(csv_file, mode="a", newline="") as file:
    #             writer = csv.writer(file)
    #             writer.writerow([iteration, v_ball, v_scaled,r_ball,radius1, high_cylinder, basal_area_cone, os.path.basename(file_name), 
    #                       "basal_area_cone = np.tan(volumn_ball) +  0.7 * height_cylinder" ])
    # else:
    #     # 原来的写入方式作为后备
    #     with open(csv_file, mode="a", newline="") as file:
    #       writer = csv.writer(file)
    #       writer.writerow([iteration, v_ball, v_scaled,r_ball,radius1, high_cylinder, basal_area_cone, os.path.basename(file_name), 
    #                       "basal_area_cone = np.tan(volumn_ball) +  0.7 * height_cylinder" ])
    # if iteration == 14:
    #   bpy.ops.wm.save_as_mainfile(filepath ="/home/lds/github/Causality-informed-Generation/code1/database/Hypothetic_V3_nonlinear_vstructure/14.blend")
    #   raise ValueError("End of the iteration")



import argparse
import sys
import os
import csv
import multiprocessing as mp
from functools import partial

def process_batch(gpu_id, batch_data, render_output_path, csv_file, resolution):
    """处理一个批次的数据"""
    # 设置GPU环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 使用文件锁来安全地写入CSV
    # lock = mp.Lock()
    
    for i in batch_data:
        # 调用原来的main函数
        main(
            render_output_path=render_output_path,
            csv_file=csv_file,
            iteration=i,
            resolution=resolution,
            # lock=lock  # 传递锁对象给main函数
        )

def split_into_batches(start, end, num_processes):
    """将数据划分成多个批次"""
    items = list(range(start, end))
    avg = len(items) // num_processes
    remainder = len(items) % num_processes
    
    batches = []
    start_idx = 0
    
    for i in range(num_processes):
        batch_size = avg + (1 if i < remainder else 0)
        batch = items[start_idx:start_idx + batch_size]
        if batch:  # 只添加非空批次
            batches.append(batch)
        start_idx += batch_size
    
    return batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--resolution', type=int, help="resolution of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    resolution = arguments.resolution
    iteration_time = arguments.size  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = f"./database/Hypothetic_V3_nonlinear_vstructure_multi/tabuler.csv"
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "iteration",         # Iteration number
                "volume_ball",       # Volume of the ball
                "scaled_volume_ball", 
                "radius of ball",
                "height_cylinder",   # Volume of the cylinder
                "radius of cone",       # Radius of the ball
                "basal_area_cone",
                
                "imgs"
            ])

    # 设置多进程参数
    num_gpus = 4
    processes_per_gpu = 10
    total_processes = num_gpus * processes_per_gpu

    # 将数据分成多个批次
    batches = split_into_batches(
        arguments.iter,
        arguments.iter + iteration_time,
        total_processes
    )

    # 创建进程池
    processes = []
    render_output_path = f"./database/Hypothetic_V3_nonlinear_vstructure_multi/"

    # 启动进程
    for idx, batch in enumerate(batches):
        gpu_id = idx // processes_per_gpu  # 确定使用哪个GPU
        p = mp.Process(
            target=process_batch,
            args=(
                gpu_id,
                batch,
                render_output_path,
                csv_file,
                resolution
            )
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()