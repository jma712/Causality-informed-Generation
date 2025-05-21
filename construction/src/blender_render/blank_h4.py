
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
move_object_to_location, render_scene, setting_camera, save_blend_file,create_rectangular_prism, rotate_object_around_edge, load_blend_file

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
    Variable d = the cylinder’s height above the ground.
     
    In the second hypothetical example, a = 3.5d; b = 3a; c = 4a + 3b + 9d + 0.7e.  
    In the third hypothetical example, b = 5a; c = 6a + 2b; d = 5c; e = 7.5a + 4.5c + 4d + 0.9e.  
    
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
    
    save_blend_file(save_path)
    dic = { 'volume_ball': a, 'height_cylinder': b, 'distance_ball_cylinder': c, 'height_cylinder_above_ground': d, 'volume_cylinder': e }
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