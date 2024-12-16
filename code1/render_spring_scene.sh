#!/bin/bash
# render reflection scene


for ((i=1; i<=10000; i+=50)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 45 帧）"
    ~/Downloads/blender-4.3.1-linux-x64/blender -b -P pipeline_spring.py -- --iter $i --size 50 --resolution 128
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done

