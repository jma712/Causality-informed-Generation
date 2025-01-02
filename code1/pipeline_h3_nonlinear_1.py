
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
# b = volume of the cylinder
# c = tilt angle of the rectangular prism
# epsilan = noise: the height of the rectangular prism above the ground

# a = sin(b) +  epsilan
# b = tan(a) + 7 b + 5 epsilan

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
    resolution = None,
    with_noise = True
  ):
    
    clear_scene()
    import uuid
    unique_filename = f"{uuid.uuid4().hex}"
    file_name = os.path.join(render_output_path, unique_filename+".png")
    
    
    if 'blank' in background.lower():
      background = "./database/blank_background_spring.blend"
      load_blend_file_backgournd(background)

    set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
    
    # randomly generate r from 0.5 to 15
    v_cylinder = random.uniform(0.1, 1.2)
    v_ball = random.uniform(0.1, 1.4)
    noise = (1.2 + 1.4) * 0.01 * np.random.randn()
    angle = np.tan(v_ball) + 7 * v_cylinder + 5 * noise
    r_ball = (3 * v_ball / (4 * math.pi))**(1/3)
    
    high_cylinder = random.uniform(0.2, 4.7)
    r_cylinder = (v_cylinder / (math.pi * high_cylinder))**(1/2)
    if r_cylinder > 2.1:
      return
    
    # blender generate ball based on r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r_ball, location=(0, 0, r_ball))
    # blender generate cylinder with volume cylinder_v
    bpy.ops.mesh.primitive_cylinder_add(radius=r_cylinder, depth=high_cylinder, location=(r_ball + r_cylinder + 0.2, 0, high_cylinder/2))
    
    obj = load_blend_file('./database/rect_hyp.blend', 
                    location=(-r_ball - 0.8 - 0.2, 0, noise), scale=(1, 1, 1), rotation_angle=0)
    
    rotate_object_y_axis_by_name('rect', angle)
    
    target_location = (0.4, 0, 2)
    camera_location = (random.uniform(0, 0), random.uniform(20, 20), random.uniform(4.2, 4.2))
    setting_camera(camera_location, target_location, len_=90)
    render_scene()
    

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, v_ball, r_ball, v_cylinder, r_cylinder, high_cylinder, angle, noise, file_name ])
    
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--resolution', type=int, help="resolution of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    resolution =  arguments.resolution
    iteration_time = arguments.size  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = f"./database/rendered_h3_nonlinear_1_{resolution}P/ref_scene_{resolution}P.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "iteration",         # Iteration number
                "volume_ball",       # Volume of the ball
                "radius_ball",       # Radius of the ball
                "volume_cylinder",   # Volume of the cylinder
                "radius_cylinder",   # Radius of the cylinder
                "height_cylinder",   # Height of the cylinder
                "tilt_angle",        # Tilt angle of the cylinder (in degrees/radians, specify unit)
                "noise_1",           # First noise parameter (explain its purpose)
                "image_path"         # Path to the associated image file
            ])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        scene = "H3"
        render_output_path = f"./database/rendered_h3_nonlinear_1_{resolution}P/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in (range(arguments.iter, arguments.iter + iteration_time)):
            main(
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                resolution = resolution
            )
