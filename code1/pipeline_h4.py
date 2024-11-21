
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


sys.path.append(os.path.abspath('/home/ulab/dxl952/Causal_project/github/Causality-informed-Generation/code1'))
from blender_render import clear_scene, disable_shadows_for_render, load_blend_file_backgournd, set_render_parameters, \
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file, rotate_object_y_axis_by_name
sys.path.append("/home/ulab/.local/lib/python3.11/site-packages")  # 请根据实际路径确认
from tqdm import tqdm

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
      
    current_time = datetime.now()
    file_name = current_time.strftime("%Y%m%d_%H%M%S")  # 格式化为 YYYYMMDD_HHMMSS
    file_name = os.path.join(render_output_path, file_name+".png")
    

    set_render_parameters(output_path=file_name, resolution=(h, w))
    camera_location = (random.uniform(-0, 0), random.uniform(23, 23), random.uniform(4, 4))
    
    r = random.uniform(0.1, 0.5)
    a = 4/3 * math.pi * r**3
    d = a / 3.5
    b = 3 * a
    height_cylinder = b
    high_above = d
    
    e = random.uniform(0.8, 1)
    
    volume_cylinder = e

    
    # blender generate ball based on r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0, 0, r))
    volume_cylinder = random.uniform(a, a+0.3)
    r_cylinder = (volume_cylinder / (math.pi * height_cylinder)) ** 0.5
    # volume_cylinder = r_cylinder * height_cylinder
    e = volume_cylinder
    c = 4 * a + 3 * b + 9 * d + 0.7 * e
    dist_ball_cylinder = c
    bpy.ops.mesh.primitive_cylinder_add(radius=r_cylinder, depth=height_cylinder, location=(r + dist_ball_cylinder+r_cylinder, 0, height_cylinder/2 + d))

    target_location = ((r + dist_ball_cylinder) /2, 0, 2)
    camera_location = ((r + dist_ball_cylinder) /2, random.uniform(20, 20), random.uniform(1, 1))
    setting_camera(camera_location, target_location)
    render_scene()
    
    # save_blend_file(save_path)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, a, b, c, d, e, file_name])



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
    csv_file = f"./database/rendered_h4_{h}x{w}/h4_scene_{h}x{w}.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", 'volume_ball', 'height_cylinder', 'distance_ball_cylinder', 'height_cylinder_above_ground', 'volume_cylinder(random.uniform(0.8, 1))','images_path', "PS: In the second hypothetical example, a = 3.5d; b = 3a; c = 4a + 3b + 9d + 0.7ε. ; In the second hypothetical example, the noise ε is the volume of the cylinder.  "])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        scene = "H4"
        render_output_path = f"./database/rendered_h4_{h}x{w}/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in tqdm(range(arguments.iter, arguments.iter + iteration_time), desc="Rendering"):
            main(
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                h = h,
                w = w
            )






