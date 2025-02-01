
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
# b = the height of the cuboid
# c = the base area of the cuboid
# d = the base area of cone 
# e = the height of cone


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

    print(f"Cuboid added at {location} with height {height_cuboid} and color {color}")

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
    np.random.seed(iteration)
    background = "./database/blank_background_spring.blend"
    load_blend_file_backgournd(background)
      
    unique_filename = f"{iteration}"
    file_name_ = os.path.join(render_output_path, unique_filename)
    
    a = v_ball = np.random.uniform(0.175, 23)
    b = height_cuboid = 0.3 * a
    
    c = base_area_cuboid =  0.3 * a + 0.5 * b
    d = base_area_cone = 1.5 * c
    
    e = height_cone = 0.3 * a + 0.3 * c + 0.2 * d
    
    radius_ball = (3 * a / (4 * math.pi))**(1/3)
    edge_cuboid = math.sqrt(base_area_cuboid)
    radius_cone = math.sqrt(base_area_cone / math.pi)
    
    # set_render_parameters(output_path=file_name, resolution=(h, w))

    distance = 0.5
    
    # add ball
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius_ball, location=(0, 0, radius_ball),
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
    
  
#     target_location = (0, 0, 5)
#     camera_location = (0, 21, 5)
#     setting_camera(camera_location, target_location, len_=45)
#     render_scene()


#     with open(csv_file, mode="a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([iteration, a, b, c, d, e, os.path.basename(file_name)])
#     # save_blend_file("./debug.blend")
    
    
# ---
    target_location = (0, 0, 5)
    camera_locations = [(0, 21, 5), (0, 23, 15), (0, 23, 22), 
                        (-8, 23, 3), (-8, 20, 15), (-8, 20, 20),
                        (8, 23, 3), (8, 23, 15), (8, 23, 20),
                        (17, 20, 3), (19, 20, 15), (10, 20, 20),
                        (27, 20, 3), (29, 20, 15), (28, 20, 20),]
    for it, camera_location in enumerate(camera_locations):
        file_name = os.path.join(render_output_path, file_name_)
        # raise ValueError(file_name)
        file_name = file_name + f"_{it}.png"
        
        setting_camera(camera_location, target_location, len_=45)
        # set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
        set_render_parameters(output_path=file_name, resolution=(h, w))
        render_scene()    
        # with open(csv_file, mode="a", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow([iteration, v_ball, high_cylinder, area, os.path.basename(file_name), 
        #                     "basal_area_cone = 0.4 * volumn_ball +  0.7 * height_cylinder" ])
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
    csv_file = f"./database/Hypothetic_v5_linear_full_connected_multi/tabular.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter","volumn_ball", "height_cuboid", 
                             "base_area_cuboid", "base_area_cone", 
                             "height_cone", 
                             'images'])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        render_output_path = f"./database/Hypothetic_v5_linear_full_connected_multi/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in (range(arguments.iter, arguments.iter + iteration_time)):
            main(
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                h = h,
                w = w
            )






