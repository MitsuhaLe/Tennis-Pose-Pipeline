import cv2
import numpy as np
import json
import os
from pathlib import Path
# 你需要根据实际保留的关节点调整这个连接关系
# 这里假设你保留的是核心关节点,按顺序定义连接
CONNECTIONS = [
    (1, 2), 
    (1, 3), (3, 5), 
    (1, 7), (7, 9), (9, 11),
     # 根据你的关节点调整
    (2, 4), (4, 6), 
    (7, 8),
    (2, 8), (8, 10), (10, 12)
    # ... 添加你需要的连接
]

def draw_frame_skeleton(frame, keypoints, color=(0, 0, 255)):
    if not keypoints or len(keypoints) == 0:
        return frame
    
    h, w = frame.shape[:2]
    frame_copy = frame.copy()
    
    # 画连线
    for start_idx, end_idx in CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            # 检查可见性
            if start_point[3] > 0.5 and end_point[3] > 0.5:
                x1, y1 = int(start_point[0] * w), int(start_point[1] * h)
                x2, y2 = int(end_point[0] * w), int(end_point[1] * h)
                cv2.line(frame_copy, (x1, y1), (x2, y2), color, 2)
    
    # 画关键点
    for kp in keypoints:
        if kp[3] > 0.5:
            x, y = int(kp[0] * w), int(kp[1] * h)
            cv2.circle(frame_copy, (x, y), 4, color, -1)
    
    return frame_copy

def draw_ppl(video_path, output_folder):
    # 读取数据
    output_file = output_folder / f"{video_path.stem}_ppl_clean.json"
    with open(output_file, 'r') as f:
        cleaned_data = json.load(f)
    print(len(cleaned_data))
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频
    output_file = output_folder / f"{video_path.stem}_ppl.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(cleaned_data):
            cleaned_frame = draw_frame_skeleton(frame.copy(), cleaned_data[frame_idx], color=(0, 0, 255))
            out.write(cleaned_frame)
        else:
            # 如果没有骨架数据，写入原始帧
            out.write(frame)
        # print(f"处理第 {frame_idx} 帧")
        frame_idx += 1

    cap.release()
    out.release()
    print("骨架视频生成完成!")

if __name__ == "__main__":
    video_path = Path("videos/demo1.mp4")
    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)
    draw_ppl(video_path, output_folder)