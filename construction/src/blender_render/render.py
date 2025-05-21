import bpy
import os
import argparse
import sys
import random

def clear_scene():
    """清空当前场景中的所有对象。"""
    if bpy.context.active_object is not None:
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
def setting_camera(location, target, scene_bounds = ((-30, 30), (-30, 30), (0, 30))):
    """
    This function sets the camera location and target.
    The camera's position should be within the range defined by the scene bounds.
    
    Parameters:
    - location: tuple (x, y, z) representing the desired camera position.
    - target: tuple (x, y, z) representing the target point the camera should point at.
    - scene_bounds: tuple of tuples ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                    defining the allowable range for camera positioning.
    """
    
    # Unpack scene bounds
    from mathutils import Vector
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = scene_bounds

    # Clamp the camera location within the scene bounds
    clamped_location = (
        max(xmin, min(xmax, location[0])),
        max(ymin, min(ymax, location[1])),
        max(zmin, min(zmax, location[2]))
    )

    # Check if a camera exists, otherwise create one
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    else:
        camera = bpy.data.objects["Camera"]

    # Set the camera location
    camera.location = Vector(clamped_location)

    # Point the camera at the target location
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    print(f"Camera location set to {camera.location}, pointing towards {target}")

range_min = -30
range_max = 30
def load_blend_file(filepath):
    """导入指定的 .blend 文件中的所有对象。"""
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects  # 选择导入所有对象
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
    print("场景已导入成功！")

def add_table(location=(0, 0, 1), size=2):
    """添加一个桌子模型并设置其材质。"""
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    table = bpy.context.object
    table.name = "Table"
    
    # 设置桌子的材质
    material_table = bpy.data.materials.new(name="TableMaterial")
    material_table.use_nodes = True
    bsdf = material_table.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = material_table.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.7, 0.4, 0.2, 1)  # 设置为木质颜色
    table.data.materials.append(material_table)
    
    print("新物体 '桌子' 已添加到场景中！")

def add_camera(location=(5, -5, 5), rotation=(1.1, 0, 0.7)):
    """添加相机并设置为活动相机。"""
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.object
    camera.name = "Camera"
    bpy.context.scene.camera = camera
    print("相机已添加并设置为活动相机。")

def set_render_parameters(resolution=(1920, 1080), file_format='PNG', output_path="../database/rendered_image.png"):
    """设置渲染参数，包括分辨率、格式和输出路径。"""
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = file_format
    print("渲染参数已设置。")

def render_scene():
    """执行渲染并保存图像。"""
    bpy.ops.render.render(write_still=True)
    print(f"渲染完成，图像已保存到：{bpy.context.scene.render.filepath}")

def save_blend_file(filepath):
    """保存当前场景为指定的 .blend 文件，直接覆盖原有文件。"""
    if os.path.exists(filepath):
        print('remove the existing file')
        os.remove(filepath)  # 删除已有文件
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"修改后的场景已保存到：{filepath}")


def delete_object_by_name(object_name):
    """
    删除指定名称的对象。
    
    Parameters:
    - object_name: 要删除的对象名称 (str)
    """
    # 检查对象是否存在
    obj = bpy.data.objects.get(object_name)
    if obj is not None:
        # 选择并删除对象
        bpy.data.objects[object_name].select_set(True)  # 选择对象
        bpy.ops.object.delete()                         # 删除对象
        print(f"对象 '{object_name}' 已删除。")
    else:
        print(f"对象 '{object_name}' 不存在。")



def main(
    background = 'blank',
    scene = 'scene',
    render_output_path = "../database/rendered_image.png",
    save_path = "../database/modified_scene.blend"
  ):
    # 使用模块化的函数执行完整流程
    if 'blank' in background.lower():
      background = "./database/blank_background.blend"
    # 1. 导入背景
    load_blend_file(background)

    # 3. 添加相机
    # add_camera()
    target_location = (0, 0, 1)
    camera_location = (random.uniform(range_min, range_max), random.uniform(range_min, range_max), random.uniform(0, range_max))

    
    # 3. 根据 `scene` 参数添加不同的对象
    if scene.lower() == "seesaw":
        add_seesaw()
    elif scene.lower() == "tennis":
        add_tennis_elements()
    elif scene.lower() == "magnetic":
        add_magnetic_elements()
    elif scene.lower() == "scene":
      pass
        # add_table()
    else:
        print(f"未识别的场景类型: {scene}，跳过特定元素添加。")


    # 4. 设置渲染参数
    setting_camera(camera_location, target_location)
    set_render_parameters(output_path=render_output_path)
    # 调用函数删除名为 'Cube' 的对象
    delete_object_by_name("Cube")
    # 5. 渲染场景
    render_scene()

    # 6. 保存场景
    if save_path:
        save_blend_file(save_path)
    # save_blend_file(save_path)


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Blender Rendering Script")

    parser.add_argument("--background", type=str, help="背景文件路径")
    parser.add_argument("--scene", type=str, help="场景类型 (例如: Seesaw, Tennis, Magnetic)")
    parser.add_argument("--render_output_path", type=str, default="../database/rendered_image.png", help="渲染输出文件路径")
    parser.add_argument("--save_path", type=str, default="", help="保存场景文件路径")
    # 解析命令行参数
    arguments, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
    # arguments = parser.parse_args()
    # 将解析的参数传递给 main 函数
    main(
        background=arguments.background,
        scene=arguments.scene,
        render_output_path=arguments.render_output_path,
        save_path=arguments.save_path
    ) 