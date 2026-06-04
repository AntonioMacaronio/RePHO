import os
import argparse
import imageio
import numpy as np


def imagefolder_to_video(inputdir, outputfile, fps=30):
    # 检查输入文件夹是否存在
    if not os.path.isdir(inputdir):
        print(f"输入文件夹不存在: {inputdir}")
        return

    # 获取所有图片文件，按文件名排序
    images = sorted([
        os.path.join(inputdir, f)
        for f in os.listdir(inputdir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    if not images:
        print(f"输入文件夹 {inputdir} 没有找到图片")
        return

    # 获取第一张图片的尺寸
    first_frame = imageio.imread(images[0])
    height, width = first_frame.shape[:2]
    if args.resize is not None:
        ratio = args.resize / min(width, height)
        width = int(round(width * ratio))
        height = int(round(height * ratio))

    writer = imageio.get_writer(outputfile, fps=fps)
    if args.flip_time:
        images = images[::-1]
    for i, img_path in enumerate(images):
        frame = imageio.imread(img_path)
        if args.resize is not None:
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((width, height), Image.LANCZOS)
            frame = np.asarray(img)

        if frame.shape[:2] != (height, width):
            print(f"警告: 图片 {img_path} 尺寸与第一帧不一致，可能导致视频异常")
        writer.append_data(frame)
        if i % 50 == 0:
            print(f"已添加 {i} 帧到视频...")

    writer.close()
    print(f"finished! video saved at: {outputfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将图片文件夹生成视频")
    parser.add_argument("--inputdir", type=str, required=True, help="输入图片文件夹路径")
    parser.add_argument("--outputfile", type=str, required=True, help="输出视频文件路径")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率，默认30fps")
    parser.add_argument("--resize", type=int, default=None, help="缩放尺寸（宽或高，等比缩放）")
    parser.add_argument("--flip_time", action='store_true', default=False, help="是否翻转时间顺序")
    args = parser.parse_args()

    imagefolder_to_video(args.inputdir, args.outputfile, args.fps)
