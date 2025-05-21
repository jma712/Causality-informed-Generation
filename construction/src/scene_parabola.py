import bpy
import math
import mathutils
import argparse
import sys
import numpy as np
sys.path.append("/home/lds/miniconda3/envs/joe/lib/python3.9/site-packages/")
from sympy import symbols, Eq, solve
import os
import csv
import multiprocessing as mp
from functools import partial

class BlenderObjectManager:
    """用于管理Blender对象的工具类"""
    
    @staticmethod
    def copy_and_place_object(object_name, new_name, location, alpha, color):
        """
        复制对象并设置其位置、透明度和颜色
        
        Args:
            object_name (str): 要复制的对象名称
            new_name (str): 新对象的名称
            location (tuple): 新位置的(x, y, z)坐标
            alpha (float): 透明度值(0.0完全透明, 1.0完全不透明)
            color (tuple): RGBA颜色值(r, g, b, a)
        
        Returns:
            bpy.types.Object: 新创建的对象,如果失败则返回None
        """
        original_object = bpy.data.objects.get(object_name)
        if not original_object:
            print(f"未找到名为'{object_name}'的对象")
            return None

        # 复制对象
        new_object = original_object.copy()
        new_object.data = original_object.data.copy()
        new_object.name = new_name
        new_object.location = location
        
        # 链接到场景
        bpy.context.scene.collection.objects.link(new_object)
        
        # 创建和设置材质
        mat = BlenderObjectManager._create_material(new_name, alpha, color)  # 修复: 使用类名而不是cls
        new_object.data.materials.clear()
        new_object.data.materials.append(mat)
        
        print(f"已复制'{object_name}'到{location}, 重命名为'{new_name}', alpha={alpha}, color={color}")
        return new_object
    
    @staticmethod
    def _create_material(name, alpha, color):
        """创建带有透明度和颜色的材质"""
        mat = bpy.data.materials.new(name=f"{name}_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # 清除现有节点
        nodes.clear()
        
        # 创建新的节点
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (300, 0)
        
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_node.location = (0, 0)
        bsdf_node.inputs['Alpha'].default_value = alpha
        bsdf_node.inputs['Base Color'].default_value = color
        
        # 连接节点
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        # 设置混合模式
        mat.blend_method = 'BLEND'
        
        return mat

    @staticmethod
    def relocate_object(object_name, new_location):
        """
        重新定位对象到新位置
        
        Args:
            object_name (str): 要移动的对象名称
            new_location (tuple): 新位置的(x, y, z)坐标
        
        Returns:
            bpy.types.Object: 移动后的对象,如果失败则返回None
        """
        obj = bpy.data.objects.get(object_name)
        if not obj:
            print(f"未找到名为'{object_name}'的对象")
            return None
            
        obj.location = new_location
        print(f"对象'{object_name}'已移动到{new_location}")
        return obj
    
    @staticmethod
    def load_from_blend(filepath, object_name, location=(0, 0, 0), 
                        scale=(1, 1, 1), rotation_axis='Z', rotation_degree=0):
        """
        从.blend文件加载对象并设置其位置、缩放和旋转。如果对象已在场景中，则直接操作它。
        
        Args:
            filepath (str): .blend文件的完整路径
            object_name (str): 要加载的对象名称
            location (tuple): 位置坐标, 默认 (0, 0, 0)
            scale (tuple): 缩放比例, 默认 (1, 1, 1)
            rotation_axis (str): 旋转轴 ('X', 'Y', 或 'Z'), 默认 'Z'
            rotation_degree (float): 旋转角度 (度), 默认 0
            
        Returns:
            bpy.types.Object: 加载的对象或现有对象, 如果失败则返回 None
        """
        try:
            # Check if the object already exists in the current scene
            existing_object = bpy.data.objects.get(object_name)
            if existing_object:
                print(f"对象 '{object_name}' 已在场景中，直接操作该对象。")
                obj = existing_object
            else:
                # Load the object from the .blend file if it doesn't exist
                with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
                    if object_name in data_from.objects:
                        data_to.objects.append(object_name)
                    else:
                        print(f"在 {filepath} 中未找到对象 '{object_name}'")
                        return None
                
                # Link the object to the scene
                obj = bpy.data.objects[object_name]
                bpy.context.scene.collection.objects.link(obj)
                print(f"从 {filepath} 加载了对象 '{object_name}'")

            # Set transformations
            obj.location = location
            obj.scale = scale

            # Set rotation
            rotation_radians = math.radians(rotation_degree)
            axis_map = {'X': 0, 'Y': 1, 'Z': 2}
            if rotation_axis.upper() in axis_map:
                obj.rotation_euler[axis_map[rotation_axis.upper()]] = rotation_radians
            else:
                print(f"无效的旋转轴 '{rotation_axis}'. 请使用 'X', 'Y', 或 'Z'")
                return None

            print(f"对象 '{object_name}' 已成功设置: 位置={location}, 缩放={scale}, "
                  f"绕 {rotation_axis} 轴旋转 {rotation_degree}°")
            return obj

        except Exception as e:
            print(f"加载对象 '{object_name}' 失败, 错误: {e}")
            return None


class BlenderCameraManager:
    """用于管理Blender相机的工具类"""
    
    @staticmethod
    def create_and_place_camera(location, target, lens, resolution=(1024, 512)):
        """
        创建新相机并使其对准目标点，如果名为 "Camera" 的对象已存在，则使用现有相机。
        
        Args:
            location (tuple): 相机位置的 (x, y, z) 坐标。
            target (tuple): 目标点的 (x, y, z) 坐标。
            lens (float): 相机镜头的焦距。
            resolution (tuple): 渲染图像的分辨率 (宽, 高)。
        
        Returns:
            bpy.types.Object: 创建或找到的相机对象。
        """
        # 检查是否存在名为 "Camera" 的对象
        camera_object = bpy.data.objects.get("Camera")
        
        if camera_object:
            return camera_object
        else:
            # 创建新相机
            camera_data = bpy.data.cameras.new(name="Camera")
            camera_object = bpy.data.objects.new("Camera", camera_data)
            
            # 链接到场景
            bpy.context.scene.collection.objects.link(camera_object)
        
            # 设置位置和朝向
            camera_object.location = location
            direction = mathutils.Vector(target) - mathutils.Vector(location)
            camera_object.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            camera_object.data.lens = lens

            # 设置渲染分辨率
            bpy.context.scene.render.resolution_x = resolution[0]
            bpy.context.scene.render.resolution_y = resolution[1]
            bpy.context.scene.render.resolution_percentage = 100  # 100% 使用全分辨率
            
            # 设置为活动相机
            bpy.context.scene.camera = camera_object
            
        return camera_object
      
    @staticmethod
    def remove_camera(camera_name="Camera"):
        """
        删除指定名称的相机对象。
        
        Args:
            camera_name (str): 要删除的相机对象的名称。
        """
        camera_object = bpy.data.objects.get(camera_name)
        
        if camera_object and camera_object.type == 'CAMERA':
            # 从场景中取消链接并删除
            bpy.data.objects.remove(camera_object, do_unlink=True)
            print(f"Camera '{camera_name}' has been removed.")
        else:
            print(f"Camera '{camera_name}' not found or is not a valid camera.")

def move_object_to_location(object_name, location):
    """
    将指定名称的对象移动到目标位置。

    Args:
        object_name (str): 对象名称。
        location (tuple): 目标位置 (x, y, z) 的坐标。
        
    Returns:
        bool: 如果成功移动对象返回 True，如果对象不存在返回 False。
    """
    obj = bpy.data.objects.get(object_name)
    if obj:
        obj.location = location
        print(f"对象 '{object_name}' 已移动到位置 {location}")
        return True
    else:
        print(f"未找到名为 '{object_name}' 的对象。")
        return False

def calculate(deform_factor, angle):
    a = deformation = original_lenght * deform_factor
    b = angle
    v = deformation * math.sqrt(constant_k / m)

    theta = math.radians(angle)  # Angle (θ) in radians (converted from degrees)

    # Calculate L (horizontal range)
    L = (deformation**2 * constant_k * math.sin(2 * theta)) / (m * g)

    # Calculate H (maximum height)
    H = (deformation**2 * constant_k * math.sin(theta)**2) / (2 * m * g)
    return L, H 
    
def trajectory_y(x, theta, v, g=9.8):
    """
    Calculate the y-coordinate of a projectile at a given x-coordinate.
    
    Args:
        x (float): The x-coordinate.
        theta (float): The angle of projection in degrees.
        v (float): The initial velocity (m/s).
        g (float): Acceleration due to gravity (default is 9.8 m/s^2).
    
    Returns:
        float: The y-coordinate corresponding to the given x-coordinate.
    """
    # Convert angle from degrees to radians
    theta_rad = math.radians(theta)
    
    # Calculate y(x)
    y = x * math.tan(theta_rad) - (g / (2 * v**2 * math.cos(theta_rad)**2)) * x**2
    return y

def clear_scene():
    """
    Clears the current scene of all objects.
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    # print("Scene cleared.")

def load_scene(filepath):
    """
    Loads all scenes from a .blend file.
    """
    try:
        with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
            data_to.scenes = data_from.scenes
        print(f"Scenes loaded from '{filepath}'.")
    except Exception as e:
        print(f"Failed to load scenes from '{filepath}'. Error: {e}")

def render_image(output_path, resolution=(512, 256), samples=128, engine='CYCLES', use_gpu=True):
    """
    Renders an image and saves it to the specified output path.

    Args:
        output_path (str): The file path where the rendered image will be saved.
        resolution (tuple): The resolution of the rendered image (width, height).
        samples (int): The number of samples for rendering (affects quality and speed).
        engine (str): The render engine to use ('CYCLES' or 'BLENDER_EEVEE').
        use_gpu (bool): Whether to use GPU for rendering (only for Cycles).
    """
    # Set the render engine
    bpy.context.scene.render.engine = engine

    # Set resolution
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100  # Use full resolution

    # Set file format and output path
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path

    if engine == 'CYCLES':
        # Configure Cycles-specific settings
        bpy.context.scene.cycles.samples = samples

        if use_gpu:
            bpy.context.scene.cycles.device = 'GPU'
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPTIX'
            bpy.context.preferences.addons['cycles'].preferences.get_devices()
            for device in bpy.context.preferences.addons['cycles'].preferences.devices:
                device.use = True  # Enable all available GPUs
        else:
            bpy.context.scene.cycles.device = 'CPU'

    # Perform the render
    bpy.ops.render.render(write_still=True)
    print(f"Rendered image saved to: {output_path}")


def load_blend_scene(blend_file_path, scene_name=None):
    """
    Loads a scene from a .blend file into the current Blender context.
    
    Args:
        blend_file_path (str): The file path to the .blend file.
        scene_name (str, optional): The name of the scene to load. 
                                    If None, the entire scene will be loaded.
    
    Returns:
        str: A message indicating success or failure.
    """
    try:
        # Check if the file exists
        if not bpy.path.abspath(blend_file_path):
            raise FileNotFoundError(f"File not found: {blend_file_path}")
        
        # Load the .blend file
        with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
            # Load all scenes if no specific scene name is given
            if scene_name is None:
                data_to.scenes = data_from.scenes
            else:
                # Load a specific scene
                if scene_name in data_from.scenes:
                    data_to.scenes = [scene_name]
                else:
                    raise ValueError(f"Scene '{scene_name}' not found in {blend_file_path}")
        
        # Link the loaded scenes to the current context
        for scene in data_to.scenes:
            if scene is not None:
                bpy.context.window.scene = scene
                print(f"Successfully loaded scene '{scene.name}' from {blend_file_path}")
        
        return "Scene loaded successfully."

    except Exception as e:
        return f"Error loading scene: {e}"

def remove_object_by_name(object_names):
    """
    Removes an object from the Blender scene by its name.
    
    Args:
        object_name (str): The name of the object to remove.
    
    Returns:
        bool: True if the object was successfully removed, False if not found.
    """
    print(object_names)
    for object_name in object_names:
      print(f"Removing object '{object_name}'...")
      obj = bpy.data.objects.get(object_name)
      if obj:
          bpy.context.scene.collection.objects.unlink(obj)
          # Remove the object from the data
          bpy.data.objects.remove(obj)
          print(f"Object '{object_name}' has been removed.")
          # return True
      else:
          print(f"Object '{object_name}' not found.")
          # return False



def constraint_value(iteration):
    '''
    Compute the deformation factor which allows all objects to appear in the scene.
    The equation is:
       10.89 * deformation**2 * sin(theta)**2 - 1.2936 * sin(theta) * (1 - deformation) < 2.8037
    
    Args:
        theta (float): Angle in degrees.
    
    Returns:
        list: A sorted list of valid deformation factors less than 0.9.
    '''
    # Convert theta to radians
    
    np.random.seed(iteration)
    degree = np.random.uniform(10, 80)
    theta_rad = math.radians(degree)
    # Extract sin(theta)
    sin_theta = math.sin(theta_rad)
    
    # Define the symbolic variable for deformation
    deformation = symbols('deformation')
    
    # Define the inequality as an equation to solve
    lhs = 10.89 * deformation**2 * sin_theta**2 - 1.2936 * sin_theta * (1 - deformation)
    rhs = 3.9537
    equation = Eq(lhs, rhs)
    
    # Solve for deformation
    solutions = solve(equation, deformation)
    print(solutions)
    
    # Filter and process the solutions
    positive_solutions = float(max(solutions))
    positive_solutions = min(0.9, positive_solutions)

    if positive_solutions < 0:
      print(solutions)
      raise ValueError("No valid deformation factor found within the constraints (< 0.9).")
    sample_num = 100
    valid_solutions = np.linspace(0.1, positive_solutions, sample_num)   # Trim to values less than 0.
    
    res = np.random.choice(valid_solutions)
    print("degree", degree,'; deformation factor:', res)
    print(positive_solutions,solutions)
    print('------')
    return degree, res
    
def setting_camera(location, target):
    """
    This function sets the camera location and target.
    The camera's position should be within the range defined by the scene bounds.
    
    Parameters:
    - location: tuple (x, y, z) representing the desired camera position.
    - target: tuple (x, y, z) representing the target point the camera should point at.
    - scene_bounds: tuple of tuples ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                    defining the allowable range for camera positioning.
    """
    
    # Clamp the camera location within the scene bounds


    # 删除已有的摄像机
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
        bpy.data.objects.remove(camera, do_unlink=True)
        # print("Deleted existing camera")

    # 创建新的摄像机
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    camera.name = "Camera"
    # print("Created new camera")

    # 设置摄像机朝向目标位置
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # 将摄像机设置为当前场景的活动摄像机

def main(iteration = None, batch_size = None, gpu_id = None):
    # 创建对象管理器实例
    clear_scene() 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    np.random.seed(iteration)
    blend_file = "/home/lds/github/Causality-informed-Generation/code1/database/parabola_dataset/code/parabola.blend"
    blend_file_path = "/home/lds/github/Causality-informed-Generation/code1/database/parabola_dataset/code//Spring.blend"
    load_blend_scene(blend_file)
    # print(iteration, batch_size, gpu_id)
    for i in range(iteration, iteration + batch_size):
      # np.random.seed(i)
      # deform_factor = np.random.uniform(0.2, 0.8)
      degree, deform_factor = constraint_value(i)
      press_factor = 1 - deform_factor
      theta = degree
      original_spring_len = 0.66
      spring_radius = 0.15
      ball_r =0.182
      offset = spring_radius * math.cos(math.radians(theta))
      obj_manager = BlenderObjectManager()
      cam_manager = BlenderCameraManager()
      
      
      # 设置相机
      camera_locations = [(4, 0, 1), (3, 0, 3),(2, 0, 6),
                          (4, 3, 1), (3, 3, 3), (1.5, 3, 6),
                          (2, 5, 1), (1.5, 5, 2), (1, 5, 3)
                          ]
      for ii, camera_location in enumerate(camera_locations):
        cam_manager.create_and_place_camera(camera_location, (0, 0, 1), lens = 20)
      
        # 复制和放置球体
        obj_manager.copy_and_place_object(
            object_name="sphere",
            new_name="sphere_copy_1",
            location=(0, 3, 0.182),
            alpha=0.5,
            color=(0.79999, 0.003414, 0.00983, 1.0)
        )
        
        obj_manager.copy_and_place_object(
            object_name="sphere",
            new_name="sphere_copy_2",
            location=(0, 0, 1),
            alpha=0.5,
            color=(0.79999, 0.003414, 0.00983, 1.0)
        )
        
        obj_manager.copy_and_place_object(
            object_name="sphere",
            new_name="sphere_copy_3",
            location=(0, 0, 1),
            alpha=0.5,
            color=(0.79999, 0.003414, 0.00983, 1.0)
        )
          
        obj_manager.copy_and_place_object(
            object_name="sphere",
            new_name="sphere_copy_4",
            location=(0, 0, 1),
            alpha=0.5,
            color=(0.79999, 0.003414, 0.00983, 1.0)
        )
        
        # 重定位原始球体
        # obj_manager.relocate_object("sphere", (0, -3, 0.5))
        spring_len = original_spring_len * press_factor
        x = (spring_len + 0.11) *  math.cos(math.radians(theta))
        y = (spring_len + 0.11) *  math.sin(math.radians(theta))
        
        # 加载外部模型
        obj_manager.load_from_blend(
            filepath=blend_file_path,
            object_name="spring",
            location=(0, -3.0, offset),
            scale=(.03, .03 * press_factor, .03),
            rotation_axis='X',
            rotation_degree=theta
        )
        if y + offset < ball_r:
          ball_y = ball_r 
          
        else:
            ball_y = y + offset
        move_object_to_location('sphere', (0, -3 + x , ball_y))
        L , H= calculate(deform_factor, degree)
        x_3 = L / 4 
        x_4 = 3 * L/4
        deformation = deform_factor * original_spring_len
        v = deformation * math.sqrt(constant_k / m)
        H_3 = trajectory_y(x_3, theta, v)
        H_4 = trajectory_y(x_4, theta, v)
        move_object_to_location('sphere_copy_1', (0,-3 + x+ (L/2) , H+ball_y))
        move_object_to_location('sphere_copy_2', (0,-3 + x + L , 0.182))
        move_object_to_location('sphere_copy_3', (0,-3 + x + x_3  , H_3 + ball_y))
        move_object_to_location('sphere_copy_4', (0,-3 + x + x_4 , H_4 + ball_y))
        move_object_to_location('Light', (0,  -3 + x + (L/2) , 5))
        file_name = f'/home/lds/github/Causality-informed-Generation/code1/database/Real_parabola/{i}_{ii}.png'
        
        render_image(file_name)
        # save the blend file
        csv_file_path = f"/home/lds/github/Causality-informed-Generation/code1/database/Real_parabola/tabular.csv"
        # if csv does not exist, create it, and add header; else append new data
        new_data = [
          {"id": i, "deformation": deform_factor, 
          "angle": degree, "H": H, "L": L, 
          'camera_location': camera_location,
          "imgs": f"{i}.png"}
        ]
        header = ["id", "deformation", "angle", "H", "L", "camera_location","imgs"]
        if not os.path.exists(csv_file_path):
            # Create the file and write the header

            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()
                writer.writerows(new_data)  # Write new data
        else:
            # Append new data to the existing file
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writerows(new_data)  # Append new data
            
        # bpy.ops.wm.save_as_mainfile(filepath=f"/home/lds/github/Causality-informed-Generation/code1/database/parabola_dataset/code/{i}.blend")
        remove_object_by_name(["sphere_copy_1", "sphere_copy_2", "sphere_copy_3", "sphere_copy_4"])
        print("---completed---\n")
        cam_manager.remove_camera()
        



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the simulation with specified parameters.")
    parser.add_argument("--iteration", type=int, required=True, help="the iteration number")
    parser.add_argument("--batch_size", type=int, required=False, default=1000, help="the number of images to generate")
    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])

    # Parse arguments
    iteration = arguments.iteration
    iteration = 50
    
    processes_per_gpu = 10
    gpus = [2, 3]
    batch_size = int(iteration / (len(gpus) * processes_per_gpu))
    
    original_lenght = 0.66
    m = 0.1
    g = 9.8
    constant_k = 25
    # remove directory
    os.system("rm -rf /home/lds/github/Causality-informed-Generation/code1/database/parabola_dataset/generated_images/*")
    os.system("rm -rf /home/lds/github/Causality-informed-Generation/code1/database/parabola_dataset/generated_images/log.log")
    # Call the main function with parsed arguments

    # GPUs to use
    

    
    tasks = []
    for gpu_id in gpus:
        for i in range(processes_per_gpu):
            start_iteration = i * batch_size + gpus.index(gpu_id) * processes_per_gpu * batch_size
            tasks.append((start_iteration, batch_size, gpu_id))

    # Launch processes for each GPU
    processes = []
    for start_iteration, batch_size, gpu_id in tasks:
        p = mp.Process(target=main, args=(start_iteration, batch_size, gpu_id))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()
    

    main(iteration)