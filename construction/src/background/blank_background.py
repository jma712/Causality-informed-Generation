
import bpy
import os


"""
建立一个类，用于创建和保存一个简单的室内场景。
"""
class BlenderScene:
    def __init__(self, save_path="../database/indoor_scene.blend"):
        self.save_path = save_path
        self.wall_thickness = 0.1
        self.wall_height = 30
        self.wall_length = 100

    def clear_scene(self):
        # 清除当前场景中的所有对象
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def create_floor(self):
        # 创建地面
        bpy.ops.mesh.primitive_plane_add(size=1000, location=(0, 0, 0))
        floor = bpy.context.object
        floor.name = "Floor"
        self.assign_material(floor, (0.5, 0.5, 0.5, 1), roughness=0.8)
        return floor

    def create_walls(self):
        # 创建四面墙
        walls = []
        wall_positions = [
            (0, self.wall_length/2 + self.wall_thickness/2, self.wall_height/2),
            (self.wall_length/2 + self.wall_thickness/2, 0, self.wall_height/2),
            (0, -self.wall_length/2 - self.wall_thickness/2, self.wall_height/2),
            (-self.wall_length/2 - self.wall_thickness/2, 0, self.wall_height/2)
        ]
        scales = [
            (self.wall_length / 1, self.wall_thickness / 1, self.wall_height / 1),
            (self.wall_thickness / 1, self.wall_length / 1, self.wall_height / 1),
            (self.wall_length / 1, self.wall_thickness / 1, self.wall_height / 1),
            (self.wall_thickness / 1, self.wall_length / 1, self.wall_height / 1)
        ]

        for i, pos in enumerate(wall_positions):
            bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
            wall = bpy.context.object
            wall.scale = scales[i]
            wall.name = f"Wall{i+1}"
            self.assign_material(wall, (0.8, 0.8, 0.8, 1), roughness=0.9)
            walls.append(wall)
        return walls

    def add_light(self):
        # 创建光源
        bpy.ops.object.light_add(type='POINT', location=(0, 0, self.wall_height - 1))
        light = bpy.context.object
        light.data.energy = 100000  # 设置光源的强度
        return light

    def assign_material(self, obj, color, roughness):
        # 创建材质并检查是否有Principled BSDF节点
        material = bpy.data.materials.new(name=f"{obj.name}Material")
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = material.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Base Color"].default_value = color

        # 将材质赋予对象
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    def save_scene(self):
        # 检查文件是否存在
        if os.path.exists(self.save_path):
            print(f"文件已存在：{self.save_path}，跳过保存。")
        else:
            # 如果文件不存在，则保存当前场景
            bpy.ops.wm.save_as_mainfile(filepath=self.save_path)
            print(f"场景已保存到：{self.save_path}")

    def create_scene(self):
        # 创建整个场景
        self.clear_scene()
        self.create_floor()
        self.create_walls()
        self.add_light()
        self.save_scene()


# 使用BlenderScene类创建并保存场景
scene = BlenderScene(save_path="../database/blank_background.blend")
scene.create_scene()