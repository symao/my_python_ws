import cv2
import numpy as np
import os
import sys

def video_combine(video1, video2, save_file):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        print('Video %s is empty'%video1)
        return

    if not ret2:
        print('Video %s is empty'%video2)
        return

    rows1,cols1 = frame1.shape[:2]
    rows2,cols2 = frame2.shape[:2]
    if not (rows1==rows2 and cols1==cols2):
        print('Video size not the same. (%dx%d)!=(%dx%d)'%(cols1,rows1,cols2,rows2))
        return

    writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"XVID"), cap1.get(cv2.CAP_PROP_FPS), (cols1,rows1))
    while ret1:
        writer.write(frame1)
        ret1, frame1 = cap1.read()

    while ret2:
        writer.write(frame2)
        ret2, frame2 = cap2.read()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        video_combine(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Usage: video_combine.py [video1] [video2] [save_file]')