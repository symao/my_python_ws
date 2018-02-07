import numpy as np
import cv2
import sys

def video_convert(video_file, out_file):
    cap = cv2.VideoCapture(video_file)
    if cap.isOpened():
        flag, img = cap.read()
        if flag:
            r,c = img.shape[:2]
            writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"XVID"), cap.get(cv2.CAP_PROP_FPS), (c,r))
            while flag:
                writer.write(img)
                flag, img = cap.read()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Use: video_convert.py [video_file] [out_file]')
    else:
        video_convert(sys.argv[1], sys.argv[2])