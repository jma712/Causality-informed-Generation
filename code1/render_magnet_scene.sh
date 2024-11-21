#!/bin/bash
for ((i=1; i<=100; i+=25)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 45 帧）"
    blender -b -P pipeline_magnet.py -- --iter $i --resolution 256 --overlook_only --without_2D
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done
