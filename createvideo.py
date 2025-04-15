import os
import shutil
import imageio

def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)

# try:
#     for i in range(295):
#         print(i)
#         shutil.copy(f'./output/infer/room/room{i}/2_views_1000Iter_1xPoseLR/interp/ours_1000/renders/00001.png',f'./room/output/img-{str(6*i+7).zfill(5)}.png')
# except:
#     pass
# images_to_video('./room/input', './original.mp4')
# images_to_video('./room/output', './generated.mp4')


# try:
#     for i in range(50):
#         shutil.copy(f'./vangogh_og/0000{(i%5)+1}.png',f'./vangogh2/{str(2*i).zfill(5)}.png')
#         shutil.copy(f'./vangogh_og/0000{(i%5)+1}.png',f'./vangogh2/{str(2*i+1).zfill(5)}.png')
# except:
#     pass
# try:
#     for i in range(50):
#         for j in range(22):
#             shutil.copy(f'./rpd_output/rpd{i+1}/{str(j).zfill(5)}.png',f'./rpd_render/{i+1+j*50}.png')
# except:
#     pass

# try:
#     for i in range(1100):
#             shutil.copy(f'./output/rpd{(i%50)+1}/{str(int(i//12)%22).zfill(5)}.png',f'./rpd_render/{i}.png')
# except:
#     pass

# try:
#     for i in range(169):
#             shutil.copy(f'./output/rpd/frame{i:0>6}/image0.png',f'./rpd_output/{i}.png')
# except:
#     pass
images_to_video('./rpd_output', './rpd.mp4',24)