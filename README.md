# Tennis Pose Pipeline 🎾

一个基于 MediaPipe 的网球动作姿态识别和分析系统。该项目使用计算机视觉技术从网球视频中提取、清洗和可视化人体骨架关键点。

## 效果演示 🎬

![处理前后对比](images/demo_comparison.png)

左图为原始视频帧，右图为经过骨架检测和可视化处理的结果（红点为关键点，绿线为骨骼连接）

## 功能特性 ✨

- **姿态提取**: 使用 MediaPipe Pose 从视频中自动检测和提取人体 33 个关键点的空间坐标
- **数据清洗**: 智能过滤低质量的检测结果，使用 Z-score 异常值检测和 Savitzky-Golay 平滑
- **关键点筛选**: 自动识别网球动作相关的 13 个关键关节点（肩、肘、腕、胯、膝、踝等）
- **可视化**: 生成带有骨架骨骼连接的注释视频输出
- **批量处理**: 支持单视频或文件夹批量处理
- **灵活配置**: 支持自定义可见度阈值和处理参数

## 项目结构 📁

```
tennis_pipe_pose/
├── run.py                 # 主入口脚本
├── pose/
│   ├── extract.py        # 姿态提取模块（MediaPipe）
│   ├── clean.py          # 数据清洗模块
│   └── draw.py           # 骨架可视化模块
├── videos/               # 输入视频文件夹
│   ├── demo_forehand.mp4
│   └── demo_serve.mp4
├── outputs/              # 输出文件夹
│   ├── *_pipeline.json   # 原始提取数据
│   ├── *_ppl_clean.json  # 清洗后的骨架数据
│   └── *_ppl.mp4         # 可视化输出视频
├── demo_comparison.png  # 处理前后对比图片
└── README.md
```

## 依赖安装 📦

### 系统要求
- **Python 3.9.25** 
- **MediaPipe 0.10.5** （依赖mediapipe.solutions）
- OpenCV (cv2)
- NumPy
- SciPy

### 安装步骤

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install python==3.9.25 mediapipe==0.10.5 opencv-python numpy scipy matplotlib
```

## 使用说明 🚀

### 处理单个视频

```bash
python run.py --video videos/demo1.mp4 --output_folder outputs
```

### 批量处理文件夹中的所有视频

```bash
python run.py --input_folder videos --output_folder outputs
```

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--video` | 指定单个视频文件路径 | `videos/demo1.mp4` |
| `--input_folder` | 指定包含多个视频的文件夹 | `videos` |
| `--output_folder` | 输出结果保存路径（**默认：outputs**） | `outputs` |

> 注意：`--video` 和 `--input_folder` 不能同时使用，只能选其一

## 处理流程 🔄

1. **姿态提取** (`extract.py`)
   - 逐帧读取视频
   - 使用 MediaPipe Pose 模型检测人体 33 个关键点
   - 保存为 JSON 格式：`{frame, landmarks: [{x, y, z, visibility}, ...]}`

2. **数据清洗** (`clean.py`)
   - 过滤低可见度的关键点（默认阈值 0.5）
   - 检测并修复异常值（Z-score 方法）
 - 对坐标序列进行平滑处理（Savitzky-Golay 滤波）
    - 输出清洗后的 [13 个关键关节点](#关键点定义-🦴)


3. **可视化** (`draw.py`)
   - 读取清洗后的骨架数据
   - 在视频帧上绘制骨骼连接和关键点
   - 生成标注后的输出视频

## 输出文件说明 📊

### 1. `*_pipeline.json` - 原始提取数据
```json
{
  "video_name": "demo1.mp4",
  "total_frames": 600,
  "landmarks_data": [
    {
      "frame": 1,
      "landmarks": [
        {"x": 0.5, "y": 0.4, "z": 0.0, "visibility": 0.99},
        ...
      ]
    }
  ]
}
```

### 2. `*_ppl_clean.json` - 清洗后的关键点数据
```json
[
  [
    [x, y, z, visibility],  # 鼻子 (nose)
    [x, y, z, visibility],  # 左肩 (left_shoulder)
    ...
  ],
  ...
]
```

### 3. `*_ppl.mp4` - 可视化视频
带有骨骼骨架连接和关键点标注的视频输出

## 关键点定义 🦴

| 编号 | 关键点 | 编号 | 关键点 |
|------|--------|------|--------|
| 0 | 鼻子 (nose) | 7 | 左髋 (left_hip) |
| 1 | 左肩 (left_shoulder) | 8 | 右髋 (right_hip) |
| 2 | 右肩 (right_shoulder) | 9 | 左膝 (left_knee) |
| 3 | 左肘 (left_elbow) | 10 | 右膝 (right_knee) |
| 4 | 右肘 (right_elbow) | 11 | 左踝 (left_ankle) |
| 5 | 左腕 (left_wrist) | 12 | 右踝 (right_ankle) |
| 6 | 右腕 (right_wrist) | - | - |

## 配置参数 ⚙️

### PoseExtractor（提取器）
```python
PoseExtractor()
# 参数：
# - model_complexity: 0 (轻), 1 (标准，默认), 2 (重)
# - min_detection_confidence: 检测置信度 (0-1)
# - min_tracking_confidence: 追踪置信度 (0-1)
```

### PoseCleaner（清洗器）
```python
PoseCleaner(visibility_threshold=0.5)
# 参数：
# - visibility_threshold: 可见度阈值 (0-1)
```

## 技术栈 🛠️

- **MediaPipe**: 人体姿态检测
- **OpenCV**: 视频处理和帧读写
- **NumPy**: 数值计算和数据处理
- **SciPy**: 信号处理（平滑滤波）

## 常见问题 ❓

### Q: 视频生成失败或为空
A: 检查以下问题：
- 清洗后的 JSON 文件是否存在且非空
- 视频编码器支持（mp4v / avc1）
- 输出路径是否有写入权限

### Q: 检测精度不够
A: 尝试以下方案：
- 提高 `model_complexity` 参数（更准确但更慢）
- 调整 `min_detection_confidence` 和 `min_tracking_confidence`
- 确保视频光照充足、人物清晰

### Q: 处理速度很慢
A: 优化建议：
- 降低 `model_complexity` 参数
- 降低视频分辨率
- 使用 GPU 加速（需要 CUDA 环持）

## 示例输出 📹

处理后得到的输出视频包含：
- 红色的骨骼连接线（关节间的连接）
- 红色的关键点圆圈（13 个关键关节点）
- 完整的视频帧和原始分辨率

## 许可证 📄

MIT License

## 作者 👨‍💻

网球姿态分析项目 - 计算机视觉应用

## 更新日志 📝

### v1.0.0 (2025-12-19)
- ✅ 初版发布
- ✅ 支持视频姿态提取
- ✅ 支持数据清洗和平滑
- ✅ 支持骨架可视化
- ✅ 支持批量处理


