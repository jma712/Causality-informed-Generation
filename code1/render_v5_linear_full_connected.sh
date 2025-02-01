rm -rf "./database/Hypothetic_v5_linear_full_connected/"
for ((i=1; i<=10000; i+=200)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 45 帧）"
    /home/lds/Downloads/blender-4.3.2-linux-x64/blender  -b -P hypothetic_v5_full_connected_linear.py -- --iter $i --size 200 --h 256 --w 256
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done
hypothetic_v5_full_connected_linear