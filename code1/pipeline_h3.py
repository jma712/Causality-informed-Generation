
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
    
    
    In the first hypothetical example, b = 2a; c = 3a + 5b + 0.5e.  
    In the second hypothetical example, a = 3.5d; b = 3a; c = 4a + 3b + 9d + 0.7e.  
    In the third hypothetical example, b = 5a; c = 6a + 2b; d = 5c; e = 7.5a + 4.5c + 4d + 0.9e.  
    
    """
    
    clear_scene()
    disable_shadows_for_render()
    if 'blank' in background.lower():
      background = "./database/blank_background_spring.blend"
      load_blend_file_backgournd(background)

    set_render_parameters(output_path=render_output_path)

    
    # move_object_to_location("Weight_Cube", (0, 0, high*scale_factor+z/2))
    
    # randomly generate r from 0.5 to 15
    r = random.uniform(0.1, 0.9)
    # v is the volume of the sphere based on r
    ball_v = 4/3 * math.pi * r**3
    cylinder_v = 2 * ball_v
    # noise e is the height of the rectangular prism above the ground
    e = random.uniform(0, 0.2) # noise e is the height of the rectangular prism above the ground
    #  tilt angle of the rectangular prism
    angle = 3*ball_v + 5*cylinder_v + 0.5*e  
    
    # blender generate ball based on r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0, 0, r))
    # blender generate cylinder with volume cylinder_v
    r_cylinder = random.uniform(0.9 * r, 2.8 * r)
    high_cylinder = 2*cylinder_v/(math.pi*r_cylinder**2)
    bpy.ops.mesh.primitive_cylinder_add(radius=r_cylinder, depth=high_cylinder, location=(r + r_cylinder + 0.2, 0, high_cylinder/2))
    
    obj = load_blend_file('/Users/liu/Desktop/school_academy/Case/Yin/causal_project/Causality-informed-Generation/code1/database/rect_hyp.blend', 
                    location=(-r - 0.8 - 0.2, 0, e), scale=(1, 1, 1), rotation_angle=0)
    
    rotate_object_y_axis_by_name('rect', angle)
    
    dic = { 'volume_ball': ball_v, 'volume_cylinder': cylinder_v, 'height_prism': e, 'tilt_angle': angle }
    target_location = (0, 0, 3.3)
    camera_location = (random.uniform(-0, 0), random.uniform(23, 23), random.uniform(3, 3))
    setting_camera(camera_location, target_location)
    render_scene()
    
    save_blend_file(save_path)
    return dic
  
def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend",
    csv_file = None,
    iteration = 0,
    circle = False,
    resolution = 128
  ):
    clear_scene()
    current_time = datetime.now()
    file_name = current_time.strftime("%Y%m%d_%H%M%S")  # 格式化为 YYYYMMDD_HHMMSS
    file_name = os.path.join(render_output_path, file_name+".png")

    background = "./database/reflection_space.blend"
    load_blend_file_backgournd(background)


    set_render_parameters(output_path=file_name, circle = circle, resolution=(resolution, resolution))
    incident_point = generate_random_coordinates()
    reflection_point = calculate_reflection_vector(incident_point)
    random_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1)  # 随机 RGB，A 设置为 1
    incident_beam = create_laser_beam(name = "IncidentBeam", color = random_color)
    reflect_beam = create_laser_beam(name = "ReflectBeam", color = random_color)
    place_and_align_cylinder(incident_beam, incident_point)
    place_and_align_cylinder(reflect_beam, reflection_point)
    camera_location = (random.uniform(-0, 0), random.uniform(10, 10), random.uniform(2, 2))

    target_location = (0, 0, 1)
    setting_camera(camera_location, target_location)

    render_scene()
    if save_path:
        save_blend_file("./temp.blend")
        

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, incident_point,  reflection_point, camera_location,
                         random_color, file_name])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--iter", type=int, help="initial number")
    parser.add_argument("--size", type=int, help="size of each iteration")
    parser.add_argument('--resolution', type=int, help="resolution of the image")

    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    resolution =  arguments.resolution
    iteration_time = arguments.size  # 每次渲染的批次数量

    # CSV 文件路径
    csv_file = f"./database/rendered_h3_{resolution}/ref_scene_{resolution}P.csv"
    if arguments.circle:
      csv_file = f"./database/rendered_h3_{resolution}/h3_scene_circle_{resolution}P.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        init = True
        # 文件不存在，创建并写入表头
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", "incident_point", "reflection_point", "camera_location", "color", "images"])
    else:
        init = False

    # 打开 CSV 文件，追加写入数据
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        # 设置背景、场景和渲染输出路径
        background = "./database/reflection_space.blend"
        scene = "H3"
        render_output_path = f"./database/rendered_h3_{resolution}/"

        # 使用起始帧数循环渲染 iteration_time 个批次
        for i in tqdm(range(arguments.iter, arguments.iter + iteration_time), desc="Rendering"):
            main(
                background=background,
                scene=scene,
                render_output_path=render_output_path,
                csv_file=csv_file,
                iteration=i,
                circle = arguments.circle,
                resolution = resolution
            )
