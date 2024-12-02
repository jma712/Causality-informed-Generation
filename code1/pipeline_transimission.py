import bpy
import sys
sys.path.append("/home/ulab/.local/lib/python3.11/site-packages")
import math
import random
from mathutils import Vector
import time
import math
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

refraction_indices = {
    "Violet (380-450 nm)": {
        "Air": 1.0003,
        "Glass (Crown)": (1.530, 1.540),
        "Glass (Flint)": (1.610, 1.640),
        "Water": 1.343,
        "Diamond": 2.440,
        "Quartz": 1.546,
        "Acrylic": 1.490,
        "Sapphire": 1.790,
        "Emerald": 1.570
    },
    "Blue (450-495 nm)": {
        "Air": 1.0003,
        "Glass (Crown)": (1.520, 1.530),
        "Glass (Flint)": (1.600, 1.620),
        "Water": 1.342,
        "Diamond": 2.430,
        "Quartz": 1.544,
        "Acrylic": 1.488,
        "Sapphire": 1.788,
        "Emerald": 1.568
    },
    "Green (495-570 nm)": {
        "Air": 1.0003,
        "Glass (Crown)": (1.517, 1.523),
        "Glass (Flint)": (1.595, 1.615),
        "Water": 1.341,
        "Diamond": 2.415,
        "Quartz": 1.543,
        "Acrylic": 1.487,
        "Sapphire": 1.787,
        "Emerald": 1.567
    },
    "Yellow (570-590 nm)": {
        "Air": 1.0003,
        "Glass (Crown)": (1.515, 1.520),
        "Glass (Flint)": (1.590, 1.610),
        "Water": 1.340,
        "Diamond": 2.407,
        "Quartz": 1.542,
        "Acrylic": 1.486,
        "Sapphire": 1.786,
        "Emerald": 1.566
    },
    "Orange (590-620 nm)": {
        "Air": 1.0003,
        "Glass (Crown)": (1.514, 1.518),
        "Glass (Flint)": (1.585, 1.605),
        "Water": 1.339,
        "Diamond": 2.400,
        "Quartz": 1.541,
        "Acrylic": 1.485,
        "Sapphire": 1.785,
        "Emerald": 1.565
    },
    "Red (620-750 nm)": {
        "Air": 1.0003,
        "Glass (Crown)": (1.513, 1.516),
        "Glass (Flint)": (1.580, 1.600),
        "Water": 1.337,
        "Diamond": 2.390,
        "Quartz": 1.540,
        "Acrylic": 1.484,
        "Sapphire": 1.784,
        "Emerald": 1.564
    }
}

def render_image(output_path, resolution_x=1920, resolution_y=1080, samples=128):
    """
    渲染图片并保存到指定路径。
    
    :param output_path: 渲染输出路径，例如 "/path/to/output/image.png"
    :param resolution_x: 渲染图像的宽度，默认1920
    :param resolution_y: 渲染图像的高度，默认1080
    :param samples: 渲染采样数量，默认128
    """
    # 设置渲染引擎为 Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # 确保 Cycles 使用 GPU 渲染
    preferences = bpy.context.preferences.addons.get("cycles")
    if not preferences:
        raise RuntimeError("Cycles 插件未启用，请在 Blender 的 Preferences 中启用 Cycles 渲染器。")
    
    cycles_preferences = preferences.preferences
    cycles_preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoising_type = 'OPTIX'  # Use 'CUDA' if OptiX is unavailable
    print('GPU Denoising Enabled')
    bpy.context.scene.cycles.device = 'GPU'

    # 触发设备检测
    cycles_preferences.get_devices()

    # 启用 GPU 设备
    devices = cycles_preferences.devices
    if not devices:
        raise RuntimeError("未检测到任何渲染设备，请确保 Metal 渲染已启用。")
    
    for device in devices:
        if device.type == "CUDA":
            device.use = True
            print(f"启用 GPU 设备: {device.name}")

    # 设置渲染分辨率
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.context.scene.render.resolution_percentage = 100  # 分辨率百分比

    # 设置采样数量
    bpy.context.scene.cycles.samples = samples

    # 设置输出文件格式和路径
    bpy.context.scene.render.image_settings.file_format = 'PNG'  # 输出格式为 PNG
    bpy.context.scene.render.filepath = output_path

    # 渲染并保存
    bpy.ops.render.render(write_still=True)
    print(f"图像已渲染并保存到 {output_path}")

def align_cylinder_to_coordinates(cylinder_name, point1, point2):
    """
    调整圆柱体，使其两端对齐到指定的空间坐标。
    
    :param cylinder_name: 圆柱体对象的名称
    :param point1: 圆柱体一端的位置，格式为 (x, y, z)
    :param point2: 圆柱体另一端的位置，格式为 (x, y, z)
    """
    # 获取圆柱体对象
    cylinder = bpy.data.objects.get(cylinder_name)
    if not cylinder:
        raise ValueError(f"找不到名为 '{cylinder_name}' 的对象")
    
    # 计算两点之间的向量和距离
    point1_vec = Vector(point1)
    point2_vec = Vector(point2)
    direction = point2_vec - point1_vec
    length = direction.length
    direction.normalize()
    
    # 设置圆柱体的长度
    if "scale" in dir(cylinder):  # 确保圆柱体对象有scale属性
       cylinder.scale[2] = length / 2  # Z方向比例设置为半长（Blender默认圆柱体长为2单位）

    # 设置圆柱体的位置（中点）
    cylinder.location = (point1_vec + point2_vec) / 2
    
    # 计算旋转（对齐方向）
    up_vector = Vector((0, 0, 1))  # 默认圆柱体的方向
    rotation = up_vector.rotation_difference(direction)
    cylinder.rotation_mode = 'QUATERNION'
    cylinder.rotation_quaternion = rotation

def calculate_angle(opposite, adjacent):
    """
    计算角度（单位：度）
    
    :param opposite: 对边的长度
    :param adjacent: 底边（邻边）的长度
    :return: 角度（单位：度）
    """
    angle_radians = math.atan(opposite / adjacent)  # 计算弧度值
    angle_degrees = math.degrees(angle_radians)  # 转换为度数
    return angle_degrees

def calculate_x(a_degree, incline_start_point, incline_end_x):
    """
    计算斜面终点的 y 坐标 (x)。
    
    :param a_degree: 入射角度（单位：度）
    :param incline_start_point: 斜面起点坐标 (x, y, z)
    :param incline_end_x: 斜面终点的 x 坐标
    :return: 斜面终点的 y 坐标
    """
    # 转换角度为弧度
    a_radian = math.radians(a_degree)
    
    # 起点和终点的 x 值
    start_x = incline_start_point[0]
    delta_x = incline_end_x - start_x  # Δx

    # 起点的 y 值
    start_y = incline_start_point[2]
    
    # 计算终点的 y 值
    end_y = start_y - delta_x * math.tan(a_radian)
    
    return end_y

def calculate_intersection_with_rotation_(point1_line1, point2_line1, center, point_on_line, angle):
    """
    计算两条直线的交点：
    - 第一条直线由两点定义。
    - 第二条直线由圆心和绕圆心旋转一定角度的点定义。

    :param point1_line1: 第一条直线上的第一个点 (x1, y1)
    :param point2_line1: 第一条直线上的第二个点 (x2, y2)
    :param center: 圆心坐标 (cx, cy)
    :param point_on_line: 第二条直线上的一个点 (px, py)
    :param angle: 第二条直线绕圆心旋转的角度（单位：度）
    :return: 交点坐标 (x, y)，如果直线平行或重合则返回 None
    """
    x1, y1 = point1_line1
    x2, y2 = point2_line1
    cx, cy = center
    px, py = point_on_line

    # 将角度转换为弧度
    angle_rad = math.radians(angle)

    # 处理 center == point_on_line 的情况
    if cx == px and cy == py:
        # 构造一个固定距离的点，以 angle 指定的方向为准
        distance = 1  # 固定长度
        rotated_x = cx + distance * math.cos(angle_rad)
        rotated_y = cy + distance * math.sin(angle_rad)
    else:
        # 正常计算旋转后的点
        rotated_x = cx + (px - cx) * math.cos(angle_rad) - (py - cy) * math.sin(angle_rad)
        rotated_y = cy + (px - cx) * math.sin(angle_rad) + (py - cy) * math.cos(angle_rad)

    # 第二条直线的点
    x3, y3 = px, py
    x4, y4 = rotated_x, rotated_y

    # 计算直线的行列式
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # 如果分母为 0，说明直线平行或重合
    if denominator == 0:
        return None, (x3, y3), (x4, y4)

    # 使用公式计算交点
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return (px, py), (x3, y3), (x4, y4)

def calculate_intersection_with_rotation(point1_line1, point2_line1, rotation_center, angle):
    """
    计算两条直线的交点：
    - 第一条直线由两点定义。
    - 第二条直线的起点与旋转中心重合，并通过旋转角度生成另一点。

    :param point1_line1: 第一条直线上的第一个点 (x1, y1)
    :param point2_line1: 第一条直线上的第二个点 (x2, y2)
    :param rotation_center: 第二条直线的起点（也是旋转中心） (cx, cy)
    :param angle: 第二条直线绕旋转中心的旋转角度（单位：度）
    :return: 交点坐标 (x, y)，如果直线平行或重合则返回 None
    """
    x1, y1 = point1_line1
    x2, y2 = point2_line1
    cx, cy = rotation_center

    # 将角度转换为弧度
    angle_rad = math.radians(angle)
    # print("angle_rad:", angle_rad)

    # 第二条直线的另一个点，假设其在旋转角度方向上的单位距离
    
    rotated_x = cx + math.cos(angle_rad)
    # print("math.cos(angle_rad)", math.cos(angle_rad), "cx:", cx, "rotated_x:", rotated_x)
    rotated_y = cy + math.sin(angle_rad)
    # print("math.sin(angle_rad)", math.sin(angle_rad), "cy:", cy, "rotated_y:", rotated_y)

    # 第二条直线的两个点
    x3, y3 = cx, cy
    x4, y4 = rotated_x, rotated_y

    
    # 计算直线的行列式
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # 如果分母为 0，说明直线平行或重合
    if denominator == 0:
        return None, (x3, y3), (x4, y4)

    # 使用公式计算交点
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    
    # # 如果 px > x1，将交点设置为 x = -10 时对应的点
    # if px > x2:
    #     # 计算第一条直线的斜率和截距
    #     if x2 != x1:  # 避免除以 0
    #         m1 = (y2 - y1) / (x2 - x1)
    #         c1 = y1 - m1 * x1
    #         px = -10
    #         py = m1 * px + c1
    #         print("px:", px, "py:", py)
    #     else:  # 如果第一条直线是垂直的，y 不变
    #         py = y1


    return (px, py), (x3, y3), (x4, y4)

def calculate_intersection_with_rotation_2(point1_line1, point2_line1, rotation_center, angle):
    """
    计算两条直线的交点：
    - 第一条直线由两点定义。
    - 第二条直线的起点与旋转中心重合，并通过旋转角度生成另一点。

    :param point1_line1: 第一条直线上的第一个点 (x1, y1)
    :param point2_line1: 第一条直线上的第二个点 (x2, y2)
    :param rotation_center: 第二条直线的起点（也是旋转中心） (cx, cy)
    :param angle: 第二条直线绕旋转中心的旋转角度（单位：度）
    :return: 交点坐标 (x, y)，如果直线平行或重合则返回 None
    """
    x1, y1 = point1_line1
    x2, y2 = point2_line1
    cx, cy = rotation_center

    # 将角度转换为弧度
    angle_rad = math.radians(angle)
    # print("angle_rad:", angle_rad)

    # 第二条直线的另一个点，假设其在旋转角度方向上的单位距离
    
    rotated_x = cx + math.cos(angle_rad)
    # print("math.cos(angle_rad)", math.cos(angle_rad), "cx:", cx, "rotated_x:", rotated_x)
    rotated_y = cy + math.sin(angle_rad)
    # print("math.sin(angle_rad)", math.sin(angle_rad), "cy:", cy, "rotated_y:", rotated_y)
    temp_cx = rotated_x
    temp_cy = rotated_y
    # 第二条直线的两个点
    x3, y3 = cx, cy
    x4, y4 = rotated_x, rotated_y

    
    # 计算直线的行列式
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # 如果分母为 0，说明直线平行或重合
    if denominator == 0:
        return None, (x3, y3), (x4, y4)

    # 使用公式计算交点
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    
    # 如果 px > x1，将交点设置为 x = -10 时对应的点
    if px > x2:
      px = 20 * (temp_cx - cx) + cx
      py = 20 * (temp_cy - cy) + cy
      # print("px:", px, "py:", py)


    return (px, py), (x3, y3), (x4, y4)


def plot_lines_and_intersection(point1_line1, point2_line1, center, point_on_line, angle, intersection, rotated_point):
    """
    可视化两条直线及其交点。

    :param point1_line1: 第一条直线上的第一个点
    :param point2_line1: 第一条直线上的第二个点
    :param center: 圆心坐标
    :param point_on_line: 第二条直线的一个点
    :param angle: 第二条直线绕圆心旋转的角度
    :param intersection: 两条直线的交点
    :param rotated_point: 旋转后的点
    """
    x1, y1 = point1_line1
    x2, y2 = point2_line1
    x3, y3 = point_on_line
    rotated_x, rotated_y = rotated_point

    # 第一条直线的点
    x_vals_line1 = [x1, x2]
    y_vals_line1 = [y1, y2]

    # 第二条直线的点
    x_vals_line2 = [x3, rotated_x]
    y_vals_line2 = [y3, rotated_y]

    # 绘制直线
    plt.plot(x_vals_line1, y_vals_line1, label="Line 1", color="blue")
    plt.plot(x_vals_line2, y_vals_line2, label=f"Line 2 (Rotated {angle}°)", color="orange")

    # 绘制圆心和旋转点
    plt.scatter(center[0], center[1], color="green", label="Center")
    plt.scatter(x3, y3, color="purple", label="Initial Point on Line 2")
    plt.scatter(rotated_x, rotated_y, color="purple", label="Rotated Point on Line 2")

    # 绘制交点
    if intersection:
        plt.scatter(*intersection, color="red", label="Intersection Point")
        plt.text(intersection[0], intersection[1], f"  ({intersection[0]:.2f}, {intersection[1]:.2f})")

    # 设置图例和标题
    plt.legend()
    plt.title("Intersection of Two Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)  # X轴
    plt.axvline(0, color='black', linewidth=0.5)  # Y轴
    plt.show()

def presetting():
  """
  预设设置，清空场景并导入示例场景。
  """
  # 删除当前场景中的所有对象
  bpy.ops.object.select_all(action='SELECT')  # 选择所有对象
  bpy.ops.object.delete()  # 删除选中的对象

  # 删除未使用的数据块（材质、纹理、网格等）
  bpy.ops.outliner.orphans_purge(do_recursive=True)

  # print("场景已清空！")
  blend_file_path = "./database/prism.blend"

  # 加载 .blend 文件中的场景
  with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
      data_to.scenes = data_from.scenes  # 导入所有场景

  # 将所有场景中的对象链接到当前场景
  for scene in data_to.scenes:
      if scene is not None:
          for obj in scene.objects:
              bpy.context.scene.collection.objects.link(obj)

  # print(f"成功导入 .blend 文件中的所有场景和对象！")

  # 打印当前场景中的所有摄像机
  print("场景中的摄像机:")
  for obj in bpy.data.objects:
      if obj.type == 'CAMERA':
          print(f"摄像机: {obj.name}")
          
  # 找到场景中的第一台摄像机
  camera = None
  for obj in bpy.data.objects:
      if obj.type == 'CAMERA':
          camera = obj
          break

  # 如果找到摄像机，设置为活动摄像机
  if camera:
      bpy.context.scene.camera = camera
      print(f"已将摄像机设置为: {camera.name}")
  else:
      print("未找到摄像机，无法设置活动摄像机。")

def get_y(x):
    """
    Calculate the y-coordinate on the line connecting (0, 2) and (1, 0) for a given x-coordinate.
    
    Parameters:
    x (float): The x-coordinate.
    
    Returns:
    float: The corresponding y-coordinate.
    """
    # Line equation: y = mx + b
    # Slope (m) = (y2 - y1) / (x2 - x1)
    m = (0 - 2) / (1 - 0)  # Slope of the line
    b = 2  # Intercept (y when x = 0)
    y = m * x + b
    return y

presetting()

prism_angle = calculate_angle(2, 1)

dataset = "/database/transmission"
cwd = os.getcwd()
database = "/".join([cwd, dataset])
# data_base = os.path.join(cwd, data_base)

materials = ['Acrylic', 'Sapphire', 'Emerald']
for material in materials:
  if not os.path.exists(database):
    os.makedirs(database)
  csv = os.path.join(database, "transmission.csv")
  print("csv:", csv)
  if not os.path.exists(csv):
    print("The file does not exist")
    # add header
    with open(csv, "w") as f:
      f.write("incident_x;incident_y;n;incident_degree;theta_1;theta_2;theta_3;theta_4;theta_5;theta_6;theta_7;theta_8;theta_9;theta_10;incident_start_point;incline_end_point;line_start_point;line_end_point;out_start_point;out_end_point;light;limit;img_path\n")

  for light in refraction_indices.keys():
    print("current light is:", light)
    # reflection = refraction_indices[light]["Glass (Crown)"]
    # reflection = (reflection[0] + reflection[1]) / 2
    reflection = refraction_indices[light][material]
    limit = math.asin(refraction_indices[light]["Air"]/reflection)
    limit = math.degrees(limit)
    # Generate a timestamp


    for i in tqdm(np.arange(0.1, 0.85, 0.05), desc="Processing"):
      incident_x = i
      incident_y = get_y(incident_x)
      for theta_1 in np.arange(0, 90, 2):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use the timestamp in a filename
        png_filename = f"{light}_{timestamp}.png"
        incident_degree = theta_1 - (90-prism_angle)

        theta_2 = math.degrees(math.asin(
            math.sin(math.radians(theta_1)) / reflection
        ))

        theta_5 = 90 - theta_2

        theta_6 = 180-(180 - prism_angle * 2) - theta_5


        theta_3 = 90 - theta_6
        if theta_3 > limit:
          print(f"theta_3 is {theta_3} > {limit} (limit degree), skip")
          continue
        # print("prism:",prism_angle)
        # print("incident_degree:",incident_degree)
        # print("theta_1:", theta_1)
        # print("theta_2:", theta_2)
        # print("theta_5:", theta_5)
        # print("theta_6:", theta_6)
        # print("theta_3:", theta_3)
        theta_4 = math.degrees(math.asin(
            (reflection / refraction_indices[light]["Air"] ) * math.sin(math.radians(theta_3))
        ))

        theta_7 = 90 - theta_4

        theta_8 = 90 - ((180  - prism_angle - 90) + theta_7)

        line_degree = 90 - prism_angle
        line_degree = 90 - line_degree
        line_degree += theta_2 


        incident_start_point = (incident_x, 0.012055, incident_y)
        incline_end_point = (incident_x * 50, 0.012055, calculate_x(incident_degree, incident_start_point, incident_x * 50))

        line1_point1 = (0,2)
        line1_point2 = (-1, 0)
        center = (incident_start_point[0], incident_start_point[-1])  # 圆心
        point_on_line = center  # 第二条直线的初始点
        # print("line_degree:", line_degree)
        angle = -(90+line_degree) # 第二条直线绕圆心旋转的角度
        # print("angle:", angle)
        intersection, point_line2, rotated_point = calculate_intersection_with_rotation(
            line1_point1, line1_point2, point_on_line, angle
        )


        # plot_lines_and_intersection(line1_point1, line1_point2, center, point_on_line, angle, intersection, rotated_point)
        line_start_point = incident_start_point
        # print(intersection)
        line_end_point = (intersection[0], 0.012055, intersection[1])

        out_start_point = line_end_point

        theta_9 = 90 - prism_angle
        theta_10 = 90+theta_9 + theta_7
        # print("theta_4:", theta_4)
        # print("theta_7:", theta_7)
        # print("theta_8:", theta_8)
        # print("theta_9:", theta_9)
        # print("theta_10:", theta_10)



        base_side_start = (-1,0)
        base_side_end = (1,0)
        # print(-theta_10)
        point_on_out = (line_end_point[0], line_end_point[-1])  
        intersection, point_line2, rotated_point = calculate_intersection_with_rotation_2(
            base_side_start, base_side_end, point_on_out, -theta_10
        )
        # print(intersection)

        # plot_lines_and_intersection(base_side_start, base_side_end, center, point_on_line, -theta_10, intersection, rotated_point)
        out_end_point = (intersection[0], 0.012055, intersection[-1])

        # 调整圆柱体的位置和方向
        align_cylinder_to_coordinates("incident", incident_start_point, incline_end_point)
        align_cylinder_to_coordinates("line", line_start_point, line_end_point)
        align_cylinder_to_coordinates("out", out_start_point, out_end_point)
        # print(-theta_10, out_start_point, out_end_point)

        # 示例用法
        # current_time = time.time().date()
        output_file = f"{database}/{png_filename}"
        # save blend file
        # bpy.ops.wm.save_as_mainfile(filepath=f"./{current_time}_prism.blend")

        render_image(output_path=output_file, resolution_x=580, resolution_y=430, samples=100)
        # save data into csv
        with open(csv, "a") as f:
          f.write(f"{incident_x};{incident_y};{reflection};{incident_degree};{theta_1};{theta_2};{theta_3};{theta_4};{theta_5};{theta_6};{theta_7};{theta_8};{theta_9};{theta_10};{incident_start_point};{incline_end_point};{line_start_point};{line_end_point};{out_start_point};{out_end_point};{light};{limit};{output_file};{material}\n")