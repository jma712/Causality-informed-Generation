
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
import uuid

# a = the volume of the ball
# b = the height of the cylinder
# c = the distance between ball and cylinder
# d = the cylinder's height above the ground
# e = the tilt angle of the cylinder

# In the third hypothetical example, 
# b = 5a + ε1; 
# c = 6a + 2b + ε2; 
# d = 2c + ε3; 
# e = 7.5a + 4.5c + 4d + ε3.

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
    file_name = uuid.uuid4().hex
    file_name = os.path.join(render_output_path, file_name+".png")
    
    if 'blank' in background.lower():
      background = "./database/blank_background_spring.blend"
      load_blend_file_backgournd(background)

    set_render_parameters(output_path=file_name, resolution=(h, w), res = 250)
    camera_location = (random.uniform(-0, 0), random.uniform(23, 23), random.uniform(4, 4))
    
    # randomly generate r from 0.5 to 15
    r = np.random.uniform(0.1, 0.5)
    # v is the volume of the sphere based on r
    a = 4/3 * math.pi * r**3
    epsilon_1 = 0.1 * np.random.randn()
    b = 5 * a + epsilon_1
    if b< 0:
        return
    epsilon_2 = 0.1 * np.random.randn()
    c = 6 * a + 2 * b + epsilon_2
    epsilon_3 = 0.1 * np.random.randn()
    d = 2 * c + epsilon_3
    if d > 13.8:
        return
    epsilon_4 = 0.1 * np.random.randn()
    e = 7.5 * a + 4.5 * c + 4 * d + epsilon_4
    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0, 0, r))
    
    r_cylinder = random.uniform(0.9 * r, 1.5 * r)
    
    bpy.ops.mesh.primitive_cylinder_add(radius=r_cylinder, depth=b, location=(r + r_cylinder + c, 0, b/2 + d))
    obj = bpy.context.object
    
    # Rename the object to the specified name
    obj.name = 'Cylinder'

    original_x = r + r_cylinder + c + 2 * r_cylinder 
    original_y = 0
    original_z = d
    angle = e
    
    # Assume the object 'Cylinder' exists
    obj = bpy.data.objects.get('Cylinder')
    if obj:
        rotate_object_around_custom_axis(obj, (original_x, original_y, original_z), angle)  # Rotate 45 degrees
    else:
        print("Object 'Cylinder' not found.")
    
    rotate_object_y_axis_by_name("Cylinder", e)
  
    target_location = ((r + r_cylinder*2 + c) /2, 0, 4.4)
    camera_h = np.random.uniform(4, 4)
    camera_location = ((r + r_cylinder*2 + c) /2, np.random.uniform(20, 20), camera_h)
    setting_camera(camera_location, target_location, len_=35 )
    # Example Usage:
    camera = bpy.data.objects['Camera']  # Replace 'Camera' with your actual camera name
    # if are_objects_in_camera_view(camera):
    render_scene()
    
    # save_blend_file(save_path)
    

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, a, b, c, d, e, camera_h, epsilon_1, epsilon_2, epsilon_3,
                         epsilon_4, os.path.basename(file_name) ])
  

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
    csv_file = f"./database/hypothetical_v5_{h}x{w}/h5_scene_{h}x{w}.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", 'volume_ball', 'height_cylinder', 'distance_ball_cylinder', 'height_cylinder_above_ground', 'tilt_angle', 'epsilon_1', 'epsilon_2', 
                             'epsilon_3','epsilon_4', 'camera_h',"file_name" ])

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        scene = "H5"
        render_output_path = f"./database/hypothetical_v5_{h}x{w}/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in (range(arguments.iter, arguments.iter + iteration_time)):
            np.random.seed(i+1_23451)
            main(
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                h = h,
                w = w
            )






