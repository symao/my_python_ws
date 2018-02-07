import cv2
import numpy as np
import os
import sys

def video_resize(video_file, rate, save_file):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()

    rows,cols = frame.shape[:2]
    size = (int(cols*rate),int(rows*rate))
    writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"XVID"), cap.get(cv2.CAP_PROP_FPS), size)
    while ret:
        writer.write(cv2.resize(frame,size))
        ret, frame = cap.read()

if __name__ == '__main__':
    if len(sys.argv) == 4:
        video_resize(sys.argv[1], float(sys.argv[2]), sys.argv[3])
    else:
        print('Usage: video_resize.py [video_file] [rate] [save_file]')