import cv2
import mediapipe as mp
import json
import os
from pathlib import Path

class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_skeleton_data(self, video_path, output_folder):
        """提取视频中的骨架数据并保存为json文件"""
        # # 设置路径
        # video_path = Path(video_path)
        # output_folder = Path(output_folder)
        # output_folder.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        frame_data = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"{video_path.name} 视频是否成功打开: {cap.isOpened()}")
        print(f"视频总帧数: {total_frames}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 添加这行来显示进度
            if frame_count % 100 == 0 or frame_count == total_frames:  # 每100帧打印一次,避免刷屏
                print(f"\r进度：{frame_count}/{total_frames} 帧 ({(frame_count/total_frames)*100:.2f}%)", end="", flush=True)    

            # 转换颜色并检测姿态
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # 提取关键点坐标
                landmark_data = {
                    'frame': frame_count,
                    'landmarks': []
                }
                for landmark in results.pose_landmarks.landmark:
                    landmark_data['landmarks'].append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_data.append(landmark_data) 

        cap.release()

        # 保存数据到json文件
        output_file = output_folder / f"{video_path.stem}_pipeline.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'video_name': video_path.name,
                'total_frames': frame_count,
                'landmarks_data': frame_data
            }, f, indent=2)
        
        print(f"已保存特征文件: {output_file.name}")

    def close(self):
        """显式关闭Pose模型释放资源"""
        self.pose.close()
        # print("Pose model resources have been released.")


