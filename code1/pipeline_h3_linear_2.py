
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

# c = 4 * a + 7 * b + 5 * epsilon

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
    unique_filename = f"{iteration}"
    file_name = os.path.join(render_output_path, unique_filename+".png")
    
    
    if 'blank' in background.lower():
      background = "./database/blank_background_spring.blend"
      load_blend_file_backgournd(background)

    set_render_parameters(output_path=file_name, resolution=(resolution, resolution))
    
    # randomly generate r from 0.5 to 15
    v_ball = random.uniform(0.175, 3.5)
    r_ball = (3 * v_ball / (4 * math.pi))**(1/3)    
    
    high_cylinder = random.uniform(0.2, 3.3)
    basal_area_cone = 0.4 * v_ball +  0.7 * high_cylinder
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
    camera_location = (-center, random.uniform(20, 20), random.uniform(4.2, 4.2))
    setting_camera(camera_location, target_location, len_=80)
    render_scene()
    
    area = radius1 ** 2 * math.pi
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, v_ball, high_cylinder, area, os.path.basename(file_name), 
                         "basal_area_cone = 0.4 * volumn_ball +  0.7 * height_cylinder" ])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--resolution', type=int, help="resolution of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    resolution =  arguments.resolution
    iteration_time = arguments.size  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = f"./database/scene_h3_linear_2_{resolution}P/ref_scene_{resolution}P.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "iteration",         # Iteration number
                "volume_ball",       # Volume of the ball
                "height_cylinder",   # Volume of the cylinder
                "basal_area_cone",
                "imgs"
            ])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        scene = "H3"
        render_output_path = f"./database/scene_h3_linear_2_{resolution}P/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in (range(arguments.iter, arguments.iter + iteration_time)):
            main(
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                resolution = resolution
            )
