
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
import uuid


sys.path.append(os.path.abspath('/home/lds/github/Causality-informed-Generation/code1'))
from blender_render import clear_scene, disable_shadows_for_render, load_blend_file_backgournd, set_render_parameters, \
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file, rotate_object_y_axis_by_name
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.12/site-packages/")
import numpy as np

def add_cuboid(height_cuboid, location=(0, 0, 0), color=(1.0, 1.0, 1.0, 1.0)):
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
    cuboid.scale[1] = 2
    cuboid.scale[0] = 2
    # Create a new material for the cuboid
    mat = bpy.data.materials.new(name="CuboidMaterial")
    mat.diffuse_color = color  # Set the RGBA color
    cuboid.data.materials.append(mat)

    print(f"Cuboid added at {location} with height {height_cuboid} and color {color}")

def main(
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend",
    csv_file= None,     
    iteration= 0,
    resolution = None,
    with_noise = True
  ):
    """
    a: the volume of ball
    b: the height of the cuboid
    c: the base area of the cone
    
    b = 4 * a
    c = -10a + 10b
    """
    
    clear_scene()
    file_name_ = f"{iteration:05d}"
    np.random.seed(iteration)
    # file_name = os.path.join(render_output_path, file_name+".png")
    
    background = "./database/blank_background_spring.blend"
    load_blend_file_backgournd(background)

    # set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
    
    # randomly generate r from 0.5 to 15

    volume_ball = np.random.uniform(0.5, 10) 
    radius = (3 * volume_ball / (4 * math.pi))**(1/3)
    
    
    height_cuboid = 0.5 * volume_ball
    
    
    distance =0.5
    height_cone = 6
    basic_area =  0.7 * volume_ball + .6 * height_cuboid
    radius_cone = math.sqrt(basic_area / math.pi)

    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, radius),
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

    add_cuboid(height_cuboid=height_cuboid, location = (radius + 1 + distance, 0, height_cuboid/2),
               color=(0.0, 1.0, 0.0, 1.0))  # Add a green cuboid
    
    bpy.ops.mesh.primitive_cone_add(vertices=70, 
                                    radius1=radius_cone, 
                                    depth=height_cone, 
                                    location=(-radius_cone - distance - radius, 0, height_cone / 2))
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
        
    target_location = (0, 0, 4)
    # camera_location = (0, 23, 4)
    # setting_camera(camera_location, target_location, len_=65)
    # render_scene()
    

    # with open(csv_file, mode="a", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow([iteration,volume_ball, height_cuboid,
    #                      basic_area, os.path.basename(file_name)])
    
    
    # target_location = (0, 0, 2)
    camera_location = (0, 23, 3)
    
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
    # save_blend_file("./debug.blend")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--resolution', type=int, help="resolution of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    resolution =  arguments.resolution
    iteration_time = arguments.size  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = f"./database/Hypothetic_v3_fully_connected_linear_multi/tabular.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", 'volume_ball', 'height_of_cuboid', 'base_area_cone', "img_path"])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        render_output_path = f"./database/Hypothetic_v3_fully_connected_linear_multi/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in (range(arguments.iter, arguments.iter + iteration_time)):
            np.random.seed(i)
            main(
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                resolution = resolution
            )
