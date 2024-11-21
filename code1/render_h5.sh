for ((i=1; i<=10000; i+=100)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 45 帧）"
    blender -b -P pipeline_h5.py -- --iter $i --size 100 --h 256 --w 100
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done