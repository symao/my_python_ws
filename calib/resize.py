import os
import cv2

def resize(dir):
    files = [os.path.join(dir,f) for f in os.listdir(dir)]

    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img,None,fx = 0.5,fy = 0.5)
        cv2.imwrite(f,img)

resize('left')    
resize('right')    
resize('stereo')    