import bpy

# 启用 Cycles 插件
if not bpy.context.preferences.addons.get("cycles"):
    bpy.ops.preferences.addon_enable(module="cycles")

# 设置渲染引擎为 Cycles 并启用 GPU
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # 或 'OPTIX'
bpy.context.scene.cycles.device = 'GPU'

# 打印设备信息
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    device.use = True
    print(f"设备名称: {device.name}, 类型: {device.type}, 已启用: {device.use}")

# 设置渲染输出文件路径
bpy.context.scene.render.filepath = "/tmp/test_render.png"

# 执行渲染并保存输出图像
bpy.ops.render.render(write_still=True)
print("渲染完成，结果保存到 /tmp/test_render.png")