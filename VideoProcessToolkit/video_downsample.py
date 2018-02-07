import cv2
import numpy as np
import os
import sys

def video_dowmsample(video_file, downsample, save_file):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()

    rows,cols = frame.shape[:2]
    writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"XVID"), cap.get(cv2.CAP_PROP_FPS), (cols,rows))
    while ret:
        writer.write(frame)
        for i in range(int(downsample)):
            ret, frame = cap.read()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        video_dowmsample(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    else:
        print('Usage: video_downsample.py [video_file] [downsample] [save_file]')