import cv2

for i in range(11,21):
    vidcap = cv2.VideoCapture(f'./flame_steak/cam{i:0>2}.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"./flame_steak/foreground/{count:0>4}.{i:0>3}.png", image)
        success,image = vidcap.read()
        count += 1