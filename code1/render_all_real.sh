#!/bin/bash

for ((i=1; i<=200; i+=50)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 45 帧）"
    /home/lds/Downloads/blender-4.3.2-linux-x64/blender  -b -P scene_spring_v3.py -- --iter $i --size 50 --resolution 256
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done

for ((i=1; i<=200; i+=100)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 25 帧）"
    /home/lds/Downloads/blender-4.3.2-linux-x64/blender  -b -P scene_seesaw_v3.py -- --iter $i --resolution 256 -S 100
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done


for ((i=1; i<=200; i+=15)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 25 帧）"
    /home/lds/Downloads/blender-4.3.2-linux-x64/blender -b -P scene_magnet_v3.py -- --iter $i --resolution 256 --overlook_only --without_2D
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done


/home/lds/Downloads/blender-4.3.2-linux-x64/blender -b -P /home/lds/github/Causality-informed-Generation/code1/scene_pendulum_v5.py
