import tkinter as tk
from tkinter import messagebox, StringVar, OptionMenu
from PIL import Image, ImageTk
import subprocess
import os
from blender_render.render import main

# 定义 Blender 可执行文件的路径字典
app_path = {
    "dxl952": "",
    "liu": "/Applications/Blender.app/Contents/MacOS/Blender"
}

# 获取项目根目录路径
project_root = os.path.dirname(os.path.abspath(__file__))

# 切换到项目根目录
os.chdir(project_root)
print("项目工作目录已设置为:", os.getcwd())


script_paths = {
    ("Blank", "Scene"): "./blender_render/render.py",
    ("Blank", "Seesaw"): "./blender_render/blank_seesaw.py",
    # ("Blank", "Tennis"): "/path_to_scripts/blank_tennis.py",
    ("Blank", "Magnetic"): "./blender_render/blank_magnetic.py",
    ("Blank", "Reflection"): "./blender_render/blank_reflection.py",
    ("Blank", "Projection"): "./blender_render/blank_projection.py",
    ("Blank", "Spring"): "./blender_render/blank_spring.py",
    ("Blank", "H3"): "./blender_render/blank_h3.py",
    ("Blank", "H4"): "./blender_render/blank_h4.py",
    ("Blank", "H5"): "./blender_render/blank_h5.py",
    ("Blank", "Prism"): "./blender_render/blank_prism.py",
    # 添加其他组合的脚本路径
}

username = os.getlogin()

def render(scene = "Seesaw", background = "Blank"):
    try:
        # 获取用户选择的参数
        selected_background = background_var.get()
        selected_scene = scene_var.get()

        # 设定 Blender 安装路径和 .blend 文件路径
        blender_path = app_path.get(username, "")

        if not os.path.exists(blender_path):
            messagebox.showerror("错误", "无法找到 Blender 可执行文件路径。")
            return
          
        # 根据背景和场景的组合获取相应的 Python 脚本路径
        script_path = script_paths.get((selected_background, selected_scene))
        print(script_path)
        if not script_path or not os.path.exists(script_path):
            messagebox.showerror("错误", "无法找到对应的 Python 脚本。")
            return


        # 设置渲染输出文件路径
        render_output = os.path.expanduser(f"./database/imgs_UI_output/render_output_{selected_background}_{selected_scene}.png")
        command = [
            blender_path,
            "-b",
            "--python", script_path,
            "--",
            f"--background={selected_background}", 
            f"--scene={selected_scene}",
            f"--render_output_path={render_output}"
        ]
        
        print(f"正在执行命令: {' '.join(command)}")
        subprocess.run(command, check=True)

        # 加载并显示渲染结果
        display_image(render_output)
        # messagebox.showinfo("完成", "渲染成功！请查看图片展示区。")

    except subprocess.CalledProcessError:
        messagebox.showerror("错误", "渲染过程中发生错误。")
    except Exception as e:
        messagebox.showerror("错误", f"发生未知错误: {e}")

def display_image(image_path):
    # 检查渲染输出文件是否存在
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img.thumbnail((800, 800))  # 调整显示区域大小
        img_tk = ImageTk.PhotoImage(img)
        
        # 更新图片显示区域
        display_label.config(image=img_tk)
        display_label.image = img_tk
    else:
        messagebox.showerror("错误", "无法加载渲染输出图片。")

if __name__ == "__main__":
  # 设置窗口的初始化参数
  WINDOW_SIZE = "800x600+200+100"
  WINDOW_TITLE = "Blender 渲染器"

  # 默认图片路径
  DEFAULT_IMAGE_PATH = "/Users/liu/Desktop/cwru-sign.jpg"  # 替换为实际路径

  # 下拉菜单选项
  BACKGROUND_OPTIONS = ["Blank", "Lab", "Indoor", "Outdoor"]
  SCENE_OPTIONS = ["Seesaw", "Reflection", "Magnetic", "Spring", "H3", "H4", "H5", 'Prism', "Tennis",  'Scene',  "Projection"]


  # 初始化主窗口
  root = tk.Tk()
  root.geometry(WINDOW_SIZE)
  root.title(WINDOW_TITLE)

  # 下拉菜单的配置
  def create_option_menu(label_text, options, variable, default_value):
      """创建一个下拉菜单并添加到窗口中"""
      tk.Label(root, text=label_text).pack()
      variable.set(default_value)
      OptionMenu(root, variable, *options).pack()

  # 创建背景和场景下拉菜单
  background_var = StringVar(root)
  scene_var = StringVar(root)

  create_option_menu("选择背景:", BACKGROUND_OPTIONS, background_var, BACKGROUND_OPTIONS[0])
  create_option_menu("选择场景:", SCENE_OPTIONS, scene_var, SCENE_OPTIONS[0])

  # 渲染按钮
  tk.Button(root, text="Render", command=render).pack(pady=20)

  # 加载默认图片
  def load_default_image(path):
      """加载默认图片，如果不存在则返回 None"""
      if os.path.exists(path):
          img = Image.open(path)
          img.thumbnail((1280, 628))
          return ImageTk.PhotoImage(img)
      return None

  default_img_tk = load_default_image(DEFAULT_IMAGE_PATH)

  # 图片显示区域
  display_label = tk.Label(root, image=default_img_tk)
  display_label.image = default_img_tk  # 保持引用
  display_label.pack(pady=20)

  # 启动主循环
  root.mainloop()