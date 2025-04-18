import os
import logging
import shutil
from argparse import ArgumentParser

parser = ArgumentParser("Data setup")
parser.add_argument("--colmap_executable", default="C:/Users/asus/Downloads/colmap-x64-windows-cuda/COLMAP.bat", type=str)
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)

args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"

for i in range(100,169):
    cur_dir = f"frame{i:0>6}"
    os.makedirs(f'{args.source_path}/{cur_dir}/input/', exist_ok=True)
    shutil.copytree(f'{args.source_path}/background/distorted',f'{args.source_path}/{cur_dir}/distorted')
    shutil.copytree(f'{args.source_path}/background/sparse',f'{args.source_path}/{cur_dir}/sparse')
    for j in range(27):
        shutil.copy(f'{args.source_path}/foreground/{i:0>4}.{j:0>3}.png',f'{args.source_path}/{cur_dir}/input/{j:0>3}.png')
    
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + args.source_path + f"/{cur_dir}/input \
        --input_path " + args.source_path + f"/{cur_dir}/distorted/sparse/0 \
        --output_path " + args.source_path + f"/{cur_dir}" +"\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(args.source_path + f"/{cur_dir}/sparse")
    os.makedirs(args.source_path + f"{cur_dir}/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(args.source_path, cur_dir, "sparse", file)
        destination_file = os.path.join(args.source_path, cur_dir, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    if(args.resize):
        print("Copying and resizing...")

        # Resize images.
        os.makedirs(args.source_path + f"{cur_dir}/images_2", exist_ok=True)
        os.makedirs(args.source_path + f"{cur_dir}/images_4", exist_ok=True)
        os.makedirs(args.source_path + f"{cur_dir}/images_8", exist_ok=True)
        # Get the list of files in the source directory
        files = os.listdir(args.source_path + f"{cur_dir}/images")
        # Copy each file from the source directory to the destination directory
        for file in files:
            source_file = os.path.join(args.source_path, cur_dir, "images", file)

            destination_file = os.path.join(args.source_path, cur_dir, "images_2", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
            if exit_code != 0:
                logging.error(f"50% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

            destination_file = os.path.join(args.source_path, cur_dir, "images_4", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
            if exit_code != 0:
                logging.error(f"25% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

            destination_file = os.path.join(args.source_path, cur_dir, "images_8", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
            if exit_code != 0:
                logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)
