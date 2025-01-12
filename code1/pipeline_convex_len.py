import bpy
import numpy as np
scene_path = "/home/lds/github/Causality-informed-Generation/code1/database/convex_len_scene/len_scene.blend"
candle_path = "/home/lds/github/Causality-informed-Generation/code1/database/convex_len_scene/candle.blend"
import os
import csv
# Open the .blend file (this replaces the current scene)

def load_object_from_blend(filepath, object_name, location, scale=(1.0, 1.0, 1.0), transparency=1.0):
    """
    从 .blend 文件加载对象，并设置位置、缩放比例和透明度。

    参数:
    - filepath (str): .blend 文件路径。
    - object_name (str): 要加载的对象名称。
    - location (tuple): 目标位置 (x, y, z)。
    - scale (tuple): 缩放比例 (x, y, z)，默认为 (1.0, 1.0, 1.0)。
    - transparency (float): 透明度值 (0.0 到 1.0)，默认为 1.0（完全不透明）。
    """
    # 从 .blend 文件加载对象
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        if object_name in data_from.objects:
            data_to.objects.append(object_name)
        else:
            # print(f"对象 '{object_name}' 不存在于 {filepath} 中。")
            return None

    # 将对象加入当前场景
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.scene.collection.objects.link(obj)
            # 设置对象位置
            obj.location = location
            # 设置对象缩放
            obj.scale = scale
            # 设置对象透明度
            if transparency < 1.0:
              set_object_transparency(obj, transparency, "wax.015")
            # print(f"成功加载 '{obj.name}'，位置: {location}, 缩放: {scale}, 透明度: {transparency}。")
            return obj
    return None

def set_object_transparency(obj, transparency, material_name=None):
    """
    为对象设置透明度，只有当指定材质存在时才设置。

    参数:
    - obj (bpy.types.Object): 要设置透明度的对象。
    - transparency (float): 透明度值 (0.0 到 1.0)。
    - material_name (str): 可选，指定材质名称，仅在材质存在时设置透明度。
    """
    # 检查对象是否有材质
    if not obj.data.materials:
        # print(f"对象 '{obj.name}' 没有材质，跳过透明度设置。")
        return

    # 遍历对象的材质
    for mat in obj.data.materials:
        # 检查材质名称是否匹配
        if material_name is None or mat.name.split('.')[0] in material_name:
            # 检查材质是否启用了节点
            if mat.use_nodes:
                nodes = mat.node_tree.nodes
                principled_bsdf = nodes.get("Principled BSDF")
                if principled_bsdf:
                    # 设置透明度
                    principled_bsdf.inputs[4].default_value = transparency  # Alpha
                    # 启用透明渲染
                    mat.blend_method = 'BLEND'
                    # mat.shadow_method = 'HASHED'
                    # print(f"设置材质 '{mat.name}' 的透明度为 {transparency}。")
                    return
                else:
                  pass
                    # print(f"材质 '{mat.name}' 中未找到 'Principled BSDF' 节点，无法设置透明度。")
            else:
              pass
                # print(f"材质 '{mat.name}' 未启用节点，无法设置透明度。")
        else:
          pass
            # print(f"材质名称 '{mat.name}' 与指定名称 '{material_name}' 不匹配，跳过。")
            
            
def set_render_param(resolution=(512, 216), file_format='PNG', 
                          output_path="../database/rendered_image.png",
                          circle = True, gpu_id = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    bpy.context.scene.cycles.samples = 200  # 渲染时的采样数
    
    if circle:
      bpy.context.scene.render.engine = 'CYCLES'
      # bpy.context.scene.render.resolution_percentage = 60

      bpy.context.preferences.addons[
          "cycles"
      ].preferences.compute_device_type = "CUDA" # or "OPENCL"

      # Set the device and feature set
      bpy.context.scene.cycles.device = "GPU"

      # get_devices() to let Blender detects GPU device
      bpy.context.preferences.addons["cycles"].preferences.get_devices()
      for d in bpy.context.preferences.addons["cycles"].preferences.devices:
          d["use"] = True # Using all devices, include GPU and CPU
          # print(d["name"], d["use"])
    

def render_image():
    bpy.ops.render.render(write_still=True)
    
import bpy

def setup_camera(location=(0, 0, 10), rotation=(0, 0, 0), focal_length = None, camera_name="Camera"):
    """
    创建并设置场景中的 Camera。
    
    参数:
    - location (tuple): 相机的位置 (x, y, z)。
    - rotation (tuple): 相机的旋转 (x, y, z) 角度（以弧度表示）。
    - focal_length (float): 相机的焦距，默认值为 50mm。
    - camera_name (str): 相机的名称。
    """
    # 检查场景中是否已存在指定名称的相机
    camera = bpy.data.objects.get(camera_name)
    if not camera:
        # 如果不存在相机，创建一个新相机
        cam_data = bpy.data.cameras.new(name=camera_name)
        camera = bpy.data.objects.new(camera_name, cam_data)
        bpy.context.scene.collection.objects.link(camera)
        # print(f"新相机 '{camera_name}' 已创建。")
    else:
      pass
        # print(f"相机 '{camera_name}' 已存在，更新其设置。")

    # 设置相机的位置和旋转
    camera.location = location
    if rotation is not None:
      camera.rotation_euler = rotation

    # 设置相机的焦距
    if camera.type == 'CAMERA' and focal_length is not None:
        camera.data.lens = focal_length

    # 将相机设置为活动相机
    bpy.context.scene.camera = camera
    # print(f"相机 '{camera_name}' 设置完成，位置: {location}，旋转: {rotation}，焦距: {focal_length}mm。")

# 示例调用
# setup_camera(location=(5, -5, 10), rotation=(1.1, 0, 0.8), focal_length=35, camera_name="MainCamera")
    
    
if __name__ == "__main__":
    set_render_param()
    original_height = 0.16
    scale = 3
    height_scale = 1.3
    focus = 0.55
    focus_2 = 2 * focus
    
    range_A = u = np.arange(focus, focus_2, 0.01)
    
    range_B = np.arange(focus_2, 3.2, 0.01)
    
    transparency_range = np.linspace(0.55, 1.1, 1000)
    max_mag = 3.3 / focus
    min_mag = focus_2 / focus_2
    mag_range = np.linspace(min_mag, max_mag, 1000)

    bpy.ops.wm.open_mainfile(filepath=scene_path)
    setup_camera(location=(-0.75, 8, -0.05), rotation=None, focal_length=None, camera_name="Camera")
    render_output = "/home/lds/github/Causality-informed-Generation/code1/database/convex_len_render_images"
    csv_file = "/home/lds/github/Causality-informed-Generation/code1/database/convex_len_render_images/render.csv"
    for i in range(10000):
      np.random.seed(i)
      set_render_param(output_path = os.path.join(render_output, f"{i}.png"))
      d_a = u = np.random.uniform(0.652, focus_2)
      d_b = v = 1/ (1/focus - 1/u)
    
      # if csv does not exist, create it
  
      if not os.path.exists(csv_file):
          with open(csv_file, mode='w', newline='') as file:
              writer = csv.writer(file)
              writer.writerow(["iter", "focal length", "Dist_object", "Dist_image", 'Magnification', "imgs"])  # 表头
      
      magnification = d_b/d_a
      index = (np.abs(mag_range - magnification)).argmin()
      transparency = transparency_range[-index]
      # 追加数据到 CSV 文件
      with open(csv_file, mode='a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow([i, focus, u, v, -magnification,  f"{i}.png"])
      obj1 = load_object_from_blend(candle_path, "candle", (d_a, 0, original_height * height_scale), scale = (scale, scale, height_scale))
      obj2 = load_object_from_blend(candle_path, "candle", (-d_b, 0, -original_height * height_scale * magnification + 0.01), 
                            scale = (scale * magnification, scale * magnification, -height_scale * magnification), transparency = transparency)
      # 保存blender文件
      render_image()
      # 删除对象
      if obj1:
          bpy.data.objects.remove(obj1, do_unlink=True)  # 删除对象并从场景中取消链接

      if obj2:
          bpy.data.objects.remove(obj2, do_unlink=True)  # 删除对象并从场景中取消链接

      
    