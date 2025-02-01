

echo "render with circle engine"
for ((i=1; i<=10000; i+=100)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 45 帧）"
     /home/lds/Downloads/blender-4.3.2-linux-x64/blender  -b -P scene_pendulum_v5.py -- --iter $i --circle --resolution 256 --size 100
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done