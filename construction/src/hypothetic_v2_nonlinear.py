
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
import math

# a = volume of the ball
# b = volume of the cylinder
# epsilan = noise: the tilt angle of the rectangular prism
# b = cos(a) + 5 epsilan

sys.path.append(os.path.abspath('/home/lds/github/Causality-informed-Generation/code1'))
from blender_render import clear_scene, disable_shadows_for_render, load_blend_file_backgournd, set_render_parameters, \
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file, rotate_object_y_axis_by_name
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.12/site-packages/")
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
    np.random.seed(iteration)

    unique_filename = f'{iteration:05d}'
    file_name = os.path.join(render_output_path, unique_filename+".png")
    background = "./database/blank_background_spring.blend"
    load_blend_file_backgournd(background)
    
    v_ball = np.random.uniform(0.05, 20)
    
    # scale v_ball to the range of  (-pi/2, pi/2)
    v_ball_scaled = (v_ball - 0.05) / (20 - 0.05) * (np.pi / 2 - (-np.pi / 2)) + (-np.pi / 2)
    v_cube = 6 * np.cos(v_ball_scaled) 
    
    radius_ball = (3 * v_ball / (4 * math.pi))**(1/3)
    egde_cube = (v_cube)**(1/3)
    
    distance = 0.2    

    set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
    
    
    # blender generate ball based on r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius_ball, 
                                         location=((radius_ball + egde_cube + distance)/2, 0, radius_ball),
                                          segments=64,  # 水平细分数 (越高越精细)
                                          ring_count=64  # 垂直细分数 (越高越精细)
                                         )
    ball = bpy.context.object  

    ball_material = bpy.data.materials.new(name="BallMaterial")
    ball_material.use_nodes = True
    nodes = ball_material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # 红色 (RGBA)
    ball.data.materials.append(ball_material)  
    #add cube
    bpy.ops.mesh.primitive_cube_add(size=egde_cube, location=(-(radius_ball + egde_cube + distance)/2, 0, egde_cube/2))
    cube = bpy.context.object  # 获取刚添加的立方体

    # 创建材料并设置颜色
    cube_material = bpy.data.materials.new(name="CubeMaterial")
    cube_material.use_nodes = True
    nodes = cube_material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.0, 0.9, 0.0, 1.0)
    cube.data.materials.append(cube_material)  # 将材料分配给立方体

  
    
#     target_location = (0, 0, 2)
#     camera_location = (0, 23, 3)
#     setting_camera(camera_location, target_location, len_=90)
#     render_scene()
#     # save_blend_file("debug.blend")
    
    

#     with open(csv_file, mode="a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([iteration, v_ball, radius_ball, v_ball_scaled,
#                          v_cube, egde_cube,
#                          os.path.basename(file_name)])
# ----

    target_location = (0, 0, 2)
    camera_location = (0, 23, 3)
    
    camera_locations = [(0, 23, 3), (0, 23, 15), (0, 23, 22), 
                        (-8, 23, 3), (-8, 20, 15), (-8, 20, 20),
                        (8, 23, 3), (8, 23, 15), (8, 23, 20),
                        (17, 20, 3), (19, 20, 15), (10, 20, 20),
                        (27, 20, 3), (29, 20, 15), (28, 20, 20),]
    for it, camera_location in enumerate(camera_locations):
        file_name = os.path.join(render_output_path, unique_filename+f"_{it}.png")
        set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
        setting_camera(camera_location, target_location, len_=90)
        render_scene()
    # setting_camera(camera_location, target_location, len_=90)
    # render_scene()
    # save_blend_file("debug.blend")
    
    

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, v_ball, radius_ball,
                         v_cube, egde_cube,
                         os.path.basename(file_name)])
    

  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")
    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--resolution', type=int, help="resolution of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    resolution =  arguments.resolution
    iteration_time = arguments.size  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = f"./database/Hypothetic_v2_nonlinear_multiview/tabular.csv"
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", 'volume_ball',"r_ball" , 'scaled_volume_ball',"volume_cube", "r_cylinder","edge_cube", "img_path"])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        render_output_path = f"./database/Hypothetic_v2_nonlinear_multiview/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in (range(arguments.iter, arguments.iter + iteration_time)):
            main(
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                resolution = resolution
            )
