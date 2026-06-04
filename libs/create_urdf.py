import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--obj_name', type=str, default='boxmedium')
argparser.add_argument('--out_path', type=str, default='/data-local/dingbang/phys_hoi_recon/InterMimic_2dproj917/behave_attouch_output/sub01_boxmedium_005')
args = argparser.parse_args()

template_path = './libs/template.urdf'

# 读取模板文件
with open(template_path, 'r') as f:
    urdf_content = f.read()

# 替换所有 "template" 为 obj_name
urdf_content = urdf_content.replace("template", args.obj_name)

# 确保输出目录存在
os.makedirs(args.out_path, exist_ok=True)

# 输出文件名，例如 boxmedium.urdf
output_file = os.path.join(args.out_path, args.obj_name, f"{args.obj_name}.urdf")

# 写入新的 urdf 文件
with open(output_file, 'w') as f:
    f.write(urdf_content)

print(f"urdf_generation_done: {output_file}")
