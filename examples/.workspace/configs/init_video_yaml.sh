#!/bin/bash

# 配置文件路径
CONFIG_FILE="./video.yaml"
# 视频文件目录
VIDEO_DIR="../assets/video"

# 启用nullglob选项，让通配符在没有匹配时返回空列表
shopt -s nullglob

# 检查视频目录是否存在，不存在则创建
if [ ! -d "$VIDEO_DIR" ]; then
    mkdir -p "$VIDEO_DIR"
    echo "已创建视频目录: $VIDEO_DIR"
fi

# 清空或创建配置文件
echo "video_list:" > "$CONFIG_FILE"

# 遍历视频目录下的所有视频文件（支持常见视频格式）
for video_file in "$VIDEO_DIR"/*.{mp4,mov,avi,flv,mkv,wmv,webm,qt}; do
    # 检查文件是否存在
    if [ -f "$video_file" ]; then
        # 获取文件名（不包含路径）
        filename=$(basename "$video_file")
        # 写入YAML格式，description留空
        echo "  - filename: \"$filename\"" >> "$CONFIG_FILE"
        echo "    description: \"\"" >> "$CONFIG_FILE"
    fi
done

# 关闭nullglob选项
shopt -u nullglob

echo "视频配置文件已生成: $CONFIG_FILE"
