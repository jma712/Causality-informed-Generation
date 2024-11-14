#!/bin/bash
# render reflection scene
for ((i=1; i<=10000; i+=45)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 45 帧）"
    /Applications/Blender.app/Contents/MacOS/Blender -b -P reflection_pipeline.py -- --iter $i
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done

echo "render with circle engine"
for ((i=1; i<=5000; i+=45)); do
    echo "正在渲染第 $i 批次的帧（从帧 $i 开始，共渲染 45 帧）"
    /Applications/Blender.app/Contents/MacOS/Blender -b -P reflection_pipeline.py -- --iter $i --circle
    echo "Blender 已完成第 $i 批次的渲染，准备重启以渲染下一个批次"
done


