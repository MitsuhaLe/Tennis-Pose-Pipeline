import numpy as np
import json
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2

class PoseCleaner:
    def __init__(self, visibility_threshold=0.5):
        """
        网球姿态数据清洗器
        visibility_threshold: 可见度阈值，低于此值的点视为不可靠
        """
        self.visibility_threshold = visibility_threshold
        
        # 定义网球动作分析需要的关键点索引
        self.key_landmarks = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        self.video_info = None
    
    def load_json(self, json_path):
        """加载JSON数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 保存视频元信息
        self.video_info = {
            'video_name': data.get('video_name', ''),
            'total_frames': data.get('total_frames', 0)
        }
        
        # 返回真正的帧数据
        return data['landmarks_data']
    
    def extract_landmark_sequences(self, landmarks_data):
        """
        从landmarks_data中提取关键点序列
        返回: (frames, key_landmarks, 4) 的数组
        """
        num_frames = len(landmarks_data)
        num_key_landmarks = len(self.key_landmarks)
        
        # 初始化数组
        sequences = np.zeros((num_frames, num_key_landmarks, 4))
        
        for i, frame_data in enumerate(landmarks_data):
            landmarks = frame_data['landmarks']
            
            # 只提取我们需要的关键点
            for j, (name, idx) in enumerate(self.key_landmarks.items()):
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    sequences[i, j, 0] = landmark['x']
                    sequences[i, j, 1] = landmark['y']
                    sequences[i, j, 2] = landmark['z']
                    sequences[i, j, 3] = landmark['visibility']
        
        return sequences
    
    def detect_outliers(self, sequences, z_threshold=3.0):
        """
        检测异常值，使用Z-score方法
        z_threshold: Z分数阈值，超过此值视为异常
        """
        num_frames, num_landmarks, _ = sequences.shape
        outlier_mask = np.zeros((num_frames, num_landmarks), dtype=bool)
        
        for landmark_idx in range(num_landmarks):
            for coord_idx in range(3):  # x, y, z
                values = sequences[:, landmark_idx, coord_idx]
                
                # 计算Z-score
                mean = np.mean(values)
                std = np.std(values)
                
                if std > 0:
                    z_scores = np.abs((values - mean) / std)
                    outlier_mask[:, landmark_idx] |= (z_scores > z_threshold)
        
        return outlier_mask
    
    def smooth_trajectory(self, sequences, window_length=5, polyorder=2):
        """
        使用Savitzky-Golay滤波平滑轨迹，去除抖动
        window_length: 窗口长度（必须是奇数）
        polyorder: 多项式阶数
        """
        num_frames, num_landmarks, num_coords = sequences.shape
        smoothed = sequences.copy()
        
        # 确保window_length是奇数且不超过帧数
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, num_frames)
        
        if window_length < polyorder + 2:
            print(f"警告: 帧数太少，跳过平滑处理")
            return smoothed
        
        for landmark_idx in range(num_landmarks):
            for coord_idx in range(3):  # x, y, z
                values = sequences[:, landmark_idx, coord_idx]
                
                try:
                    smoothed[:, landmark_idx, coord_idx] = savgol_filter(
                        values, window_length, polyorder
                    )
                except Exception as e:
                    print(f"平滑关键点 {landmark_idx} 坐标 {coord_idx} 时出错: {e}")
        
        return smoothed
    
    def interpolate_low_confidence(self, sequences):
        """
        对低置信度的点进行插值修复
        """
        num_frames, num_landmarks, _ = sequences.shape
        interpolated = sequences.copy()
        
        for landmark_idx in range(num_landmarks):
            visibility = sequences[:, landmark_idx, 3]
            low_conf_mask = visibility < self.visibility_threshold
            
            if np.all(low_conf_mask):
                print(f"警告: 关键点 {landmark_idx} 所有帧置信度都很低")
                continue
            
            if np.any(low_conf_mask):
                # 找到高置信度的帧
                good_indices = np.where(~low_conf_mask)[0]
                bad_indices = np.where(low_conf_mask)[0]
                
                # 对x, y, z分别插值
                for coord_idx in range(3):
                    good_values = sequences[good_indices, landmark_idx, coord_idx]
                    
                    if len(good_indices) > 1:
                        # 使用线性插值
                        interp_func = interp1d(
                            good_indices, good_values,
                            kind='linear',
                            fill_value='extrapolate'
                        )
                        interpolated[bad_indices, landmark_idx, coord_idx] = interp_func(bad_indices)
        
        return interpolated
    
    def clean(self, video_path, output_folder, smooth=True, interpolate=True, remove_outliers=True):
        """
        完整的数据清洗流程
        """
        json_path = output_folder / f"{video_path.stem}_pipeline.json"
        print(f"开始清洗数据: {json_path}")
        
        # 1. 加载数据
        landmarks_data = self.load_json(json_path)
        print(f"视频: {self.video_info['video_name']}, 总帧数: {self.video_info['total_frames']}")
        
        # 2. 提取关键点序列
        sequences = self.extract_landmark_sequences(landmarks_data)
        print(f"提取了 {len(self.key_landmarks)} 个关键点，共 {sequences.shape[0]} 帧")
        
        # 3. 检测并处理异常值
        if remove_outliers:
            outlier_mask = self.detect_outliers(sequences)
            num_outliers = np.sum(outlier_mask)
            print(f"检测到 {num_outliers} 个异常值")
            
            # 将异常值的visibility设为0，后续会被插值
            for frame_idx, landmark_idx in zip(*np.where(outlier_mask)):
                sequences[frame_idx, landmark_idx, 3] = 0
        
        # 4. 插值修复低置信度点
        if interpolate:
            sequences = self.interpolate_low_confidence(sequences)
            print("完成低置信度点插值")
        
        # 5. 平滑轨迹
        if smooth:
            sequences = self.smooth_trajectory(sequences)
            print("完成轨迹平滑")
        
        # 6. 保存清洗后的数据
        # 在保存之前先转换一下
        cleaned_sequence_serializable = self.convert_to_serializable(sequences)
        # 然后再保存
        output_file = output_folder / f"{video_path.stem}_ppl_clean.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_sequence_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"已保存特征文件: {output_file.name}")
    
    def get_landmark_name(self, index):
        """根据索引获取关键点名称"""
        names = list(self.key_landmarks.keys())
        if 0 <= index < len(names):
            return names[index]
        return f"landmark_{index}"

    def convert_to_serializable(self,obj):
        """递归地将NumPy数组转换为Python列表"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def save_cleaned_data(self, sequences, output_path):
        pass


# 使用示例
if __name__ == "__main__":
    pass