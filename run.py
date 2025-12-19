import argparse
import os
from pathlib import Path
import pose.draw
from pose.extract import PoseExtractor
from pose.clean import PoseCleaner


def process_video(input_video_path, output_video_path):
    """处理单个视频"""
    # 设置路径
    video_path = Path(input_video_path)
    output_folder = Path(output_video_path)
    output_folder.mkdir(exist_ok=True)

    # 初始化骨架提取器
    pose_extractor = PoseExtractor()
    pose_extractor.extract_skeleton_data(video_path, output_folder)
    pose_extractor.close()

    # 清洗骨架数据
    cleaner = PoseCleaner(visibility_threshold=0.5)
    cleaner.clean(video_path, output_folder)

    # 绘制骨架并保存输出视频
    pose.draw.draw_ppl(video_path, output_folder)
    print(f"Processed {video_path} and saved to {output_folder}")

def process_folder(input_folder, output_folder):
    """处理文件夹中的所有视频"""
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):  # 可以根据需要修改文件类型
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, filename)
            
            # 处理每个视频
            process_video(input_video_path, output_folder)

def main():
    parser = argparse.ArgumentParser(description="Process tennis serve videos.")
    
    # 支持两种输入方式：单个视频或文件夹
    parser.add_argument('--video', help='Path to a single video to process')
    parser.add_argument('--input_folder', help='Path to a folder containing videos to process')
    parser.add_argument('--output_folder', help='Path to the output folder to save processed videos', default='outputs')
    args = parser.parse_args()

    # 检查输入参数
    if args.video and args.input_folder:
        print("Error: You can only specify one of --video or --input_folder, not both.")
        return

    # 处理单个视频
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: The video file {args.video} does not exist.")
            return
        if not args.output_folder:
            print("Error: Please specify --output_folder for single video processing.")
            return

        # output_video_path = os.path.join(args.output_folder, os.path.basename(args.video))
        output_video_path = args.output_folder
        print(output_video_path)
        process_video(args.video, output_video_path)

    # 处理文件夹中的所有视频
    elif args.input_folder:
        if not os.path.exists(args.input_folder):
            print(f"Error: The input folder {args.input_folder} does not exist.")
            return
        if not args.output_folder:
            print("Error: Please specify --output_folder for batch video processing.")
            return

        os.makedirs(args.output_folder, exist_ok=True)
        process_folder(args.input_folder, args.output_folder)

    else:
        print("Error: Please specify either --video or --input_folder.")
        return

if __name__ == '__main__':
    main()