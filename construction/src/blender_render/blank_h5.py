
import argparse
import sys
import os
import random
import math
import bpy
from mathutils import Vector
from mathutils import Vector, Matrix

sys.path.append(os.path.abspath('/Users/liu/Desktop/school_academy/Case/Yin/causal_project/Causality-informed-Generation/code1'))
from blender_render import clear_scene, disable_shadows_for_render, load_blend_file_backgournd, set_render_parameters, \
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file, rotate_object_y_axis_by_name

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


def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend"
  ):
    """
    In the first hypothetical example, the noise e is the height of the rectangular prism above the ground.  
    In the second hypothetical example, the noise e is the volume of the cylinder.  
    In the third hypothetical example, the noise e is the height of the cylinder above the ground.  
    
    Variable a = the volume of a ball;
    Variable b = the height of a cylinder;
    Variable c = the distance between the ball and the cylinder;
    Variable d = the cylinder’s height above the ground;
    Variable e = the tilt angle of the cylinder.
     
    In the third hypothetical example, b = 5a; c = 6a + 2b; d = 2c; e = 7.5a + 4.5c + 4d + 0.9e.  
    
    """
    
    clear_scene()
    disable_shadows_for_render()
    if 'blank' in background.lower():
      background = "./database/blank_background_spring.blend"
      load_blend_file_backgournd(background)

    set_render_parameters(output_path=render_output_path)
    camera_location = (random.uniform(-0, 0), random.uniform(23, 23), random.uniform(4, 4))
    
    # move_object_to_location("Weight_Cube", (0, 0, high*scale_factor+z/2))
    
    # randomly generate r from 0.5 to 15
    r = random.uniform(0.1, 0.5)
    # v is the volume of the sphere based on r
    a = 4/3 * math.pi * r**3
    b = 5 * a
    c = 6 * a + 2 * b
    d = 2 * c
    noise = random.uniform(0, 0.1)
    e = 7.5 * a + 4.5 * c + 4 * d + 0.9 * noise
    
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
    aspect_ratio=(1920, 1920)
    bpy.context.scene.render.resolution_x = aspect_ratio[0]
    bpy.context.scene.render.resolution_y = aspect_ratio[1]
    # bpy.context.scene.render.resolution_percentage = 100  # Ensure full resolution

  
    target_location = ((r + r_cylinder*2 + c) /2, 0, 3*d/2)
    camera_location = ((r + r_cylinder*2 + c) /2, random.uniform(20, 20), random.uniform(1, 1))
    setting_camera(camera_location, target_location, len_=30 )
    render_scene()
    
    save_blend_file(save_path)
    dic = { 'volume_ball': a, 'height_cylinder': b, 'distance_ball_cylinder': c, 'height_cylinder_above_ground': d, 'tilt_angle': e, 'noise': noise }
    print(dic)
    # return dic
  
if __name__ == "__main__":
  # 创建 ArgumentParser 对象
  parser = argparse.ArgumentParser(description="Blender Rendering Script")

  parser.add_argument("--background", type=str, help="背景文件路径")
  parser.add_argument("--scene", type=str, help="场景类型 (例如: Seesaw, Tennis, Magnetic)")
  parser.add_argument("--render_output_path", type=str, default="../database/rendered_image.png", help="渲染输出文件路径")
  parser.add_argument("--save_path", type=str, default="/Users/liu/Desktop/school_academy/Case/Yin/causal_project/Causality-informed-Generation/code1/database/temp.blend", help="保存场景文件路径")
  arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
  records = main(
      background=arguments.background,
      scene=arguments.scene,
      render_output_path=arguments.render_output_path,
      save_path=arguments.save_path
  ) 
  print(records)