
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

# a = the volume of the ball
# b = the volume of the cylinder
# c = the distance between the ball and the cylinder
# d = the height of the cylinder above the ground
# epsilon_1 = the hight of rectangular prism above the ground
# epsilon_2 = the tile angle of the rectangular prism

# c = np.sin(a) + 7 * b + 5 * epsilon_1
# d = np.con(c) + 2 * epsilon_2

sys.path.append(os.path.abspath('/home/lds/github/Causality-informed-Generation/code1'))
from blender_render import clear_scene, disable_shadows_for_render, load_blend_file_backgournd, set_render_parameters, \
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file, rotate_object_y_axis_by_name
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.9/site-packages/")
import numpy as np

def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend",
    csv_file= None,     
    iteration= 0,
    h = 256,
    w = 256,
    with_noise = True
  ):
    clear_scene()
    if 'blank' in background.lower():
      background = "./database/blank_background_spring.blend"
      load_blend_file_backgournd(background)
      
    import uuid
    unique_filename = f"{uuid.uuid4().hex}"
    file_name = os.path.join(render_output_path, unique_filename+".png")
    

    set_render_parameters(output_path=file_name, resolution=(h, w))
    camera_location = (random.uniform(-0, 0), random.uniform(23, 23), random.uniform(4, 4))
    
    
    v_cylinder = b = random.uniform(0.2, 1)
    v_ball = a = random.uniform(0.2, 1)
    r = (3 * v_ball / (4 * math.pi))**(1/3)
    epsilon_1 = (0.84 + 7 * 0.3) * 0.01 * np.random.randn()
    distance = c = np.sin(a) + 7 * b + 5 * epsilon_1
    
    epsilon_2 = (0.5) * 0.01 * np.random.randn()
    d = hight =  np.cos(c) + 2 * epsilon_2
    if hight > 9.5:
        return
    if hight < 0.0:
      return 
  
    
    # blender generate ball based on r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0, 0, r))

    height_cylinder = random.uniform(0.05, 1.5)
    r_cylinder = (v_cylinder / (math.pi * height_cylinder))**(1/2)

    bpy.ops.mesh.primitive_cylinder_add(radius=r_cylinder, depth=height_cylinder, 
                                        location=(r + distance + r_cylinder, 0, height_cylinder/2 + d))

    target_location = ((r + distance) /2, 0, 2)
    camera_location = ((r + distance) /2, random.uniform(21, 21), random.uniform(2, 2))
    setting_camera(camera_location, target_location)
    render_scene()


    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, a, b, c, d, epsilon_1, 
                         epsilon_2, file_name, np.nan])



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
    csv_file = f"./database/rendered_h4_{h}x{w}_nonlinear/h4_scene_{h}x{w}.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter","volumn_ball", "volumn_cylinder", "distance", "height", 
                             "epsilon_1", "epsilon_2",  "file_name", "c = np.sin(a) + 7 b + 5 epsilon_1; d = np.cos(c) + 2 epsilon_2"])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        scene = "H4"
        render_output_path = f"./database/rendered_h4_{h}x{w}_nonlinear/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in (range(arguments.iter, arguments.iter + iteration_time)):
            main(
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                h = h,
                w = w
            )






