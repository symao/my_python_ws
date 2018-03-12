import numpy as np
import cv2
import sys
import os
import shutil

def video_reverse(video_file, out_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print('video revert failed.')
        return
    print("start reversing...")
    tmpdir = 'tmp'
    while os.path.exists(tmpdir):
        tmpdir += '_1'
    os.makedirs(tmpdir)
    flag, img = cap.read()
    r,c = img.shape[:2]
    cnt = 0
    files = []
    while flag:
        f = os.path.join(tmpdir,'%06d.jpg'%cnt)
        cv2.imwrite(f,img)
        files.append(f)
        cnt+=1
        flag, img = cap.read()
    if len(files)>0:
        files = files[::-1]
        writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"XVID"), cap.get(cv2.CAP_PROP_FPS), (c,r))
        for f in files:
            writer.write(cv2.imread(f))
    shutil.rmtree(tmpdir)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Use: video_reverse.py [video_file] [out_file]')
    else:
        video_reverse(sys.argv[1], sys.argv[2])
