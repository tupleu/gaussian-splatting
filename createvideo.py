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
#             shutil.copy(f'./output/background/train/ours_30000/renders/{i:0>3}.pt',f'./background/{i+1:0>3}.pt')
# except:
#     pass
# images_to_video('./rpd_output', './rpd.mp4',24)

# flip = False
# for i in range(300):
#     # os.makedirs(f'./minecraft_hall/frame{i:0>6}/input',exist_ok=True)
#     # if i % 2 == 0:
#     #     continue
#     j = i % 10
#     k = i % 26
#     if flip:
#         k = 25 - k
#     if j == 0:
#         shutil.copy(f'./output/background_hall/train/ours_30000/gt/{k:0>5}.png',f'./minecraft_hall_output/{i:0>3}.png')
#     else:
#         shutil.copy(f'./output/minecraft_hall/frame{j:0>6}/train/ours_2000/gt/{k:0>5}.png',f'./minecraft_hall_output/{i:0>3}.png')
#     if not flip and k == 25:
#         flip = True
#     if flip and k == 0:
#         flip = False

# for filename in sorted(os.listdir('./cornellbox/background/input')):
ind = [
    0,
    1,
    2,
    3,
    4,
    9,
    8,
    7,
    6,
    5,
    10,
    11,
    12,
    13,
    14,
    19,
    18,
    17,
    16,
    15,
    20,
    21,
    22,
    23,
    24,
    23,
    22,
    21,
    20,
    15,
    16,
    17,
    18,
    19,
    14,
    13,
    12,
    11,
    10,
    5,
    6,
    7,
    8,
    9,
    4,
    3,
    2,
    1,
]
for i in range(288):
    # if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
    # os.makedirs(f'./cornellbox/frame{i-1:0>6}/input',exist_ok=True)
    # for j in range(1,26):
    #     shutil.move(f'./output_diffuse/{i:0>4}.{j:0>3}.png',f'./cornellbox/frame{i-1:0>6}/input/{j:0>3}.png')
    f = i % 100
    k = i//6 % 48
    shutil.copy(f'./output/cornellbox/frame{f:0>6}/train/ours_600/gt/{ind[k]:0>5}.png', f'./cb_gt/{i:0>4}.png')

    

# try:
#     for i in range(30):
#         shutil.copy(f'./output/flame_steak/frame{i:0>6}/000.png',f'./flame_steak_graphs/{i:0>3}.png')
# except:
#     pass

# quit()

# import numpy as np
# import open3d

# # Read point cloud from PLY
# pcd1 = open3d.io.read_point_cloud("points3Dog.ply")
# points = np.asarray(pcd1.points)
# normals = np.asarray(pcd1.normals)
# colors = np.asarray(pcd1.colors)

# # # Sphere center and radius
# # center = np.array([1.586, -8.436, -0.242])
# # radius = 0.5

# # # Calculate distances to center, set new points
# # distances = np.linalg.norm(points - center, axis=1)
# # pcd1.points = open3d.utility.Vector3dVector(points[distances <= radius])
# # print(points)
# # pcd1.points = points[:1]
# pcd1.points = open3d.utility.Vector3dVector(points[:1000])
# pcd1.normals = open3d.utility.Vector3dVector(normals[:1000])
# pcd1.colors = open3d.utility.Vector3dVector(colors[:1000])
# # 
# # Write point cloud out
# open3d.io.write_point_cloud("points3D.ply", pcd1)