import cv2
import argparse
import os
import pdb

video_path = "/media/zhr/文件/FaceRec-master/video_pre/text.mp4"
outputPath_image = "/media/zhr/文件/FaceRec-master/image_output"
cap = cv2.VideoCapture(video_path)
ret,frame = cap.read()
while True:
    ret,frame = cap.read()
        
    # 每隔100帧保存一张图片
    frame_interval = 100
    # 统计当前帧
    frame_count = 1
    # 保存图片个数
    count = 0

    if frame_count % frame_interval == 0:
        filename = os.path.sep.join([outputPath, "test_{}.jpg".format(count)])
        cv2.imwrite(filename, frame)
        count += 1
        print("保存图片:{}".format(filename))
    frame_count += 1
vc.release()
print("[INFO] 总共保存：{}张图片".format(count))