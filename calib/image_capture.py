import cv2
import os
import numpy as np


def capture_mono(img_dir, cam_id=0):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cnt = 0
    cap = cv2.VideoCapture(cam_id)
    if cap.isOpened():
        while 1:
            ret, img = cap.read()
            cv2.imshow("img",img)
            key = cv2.waitKey(30)
            if key == 27:
                break
            elif key == ord('s'):
                f = '%s/image%06d.png'%(img_dir,cnt)
                cv2.imwrite(f,img)
                print('save img '+f)
                cnt+=1
    else:
        print("Cannot open camera %d"%cam_id)

def capture_stereo(img_dir, cam_id_left, cam_id_right):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cnt = 0
    cap_left = cv2.VideoCapture(cam_id_left)
    cap_right = cv2.VideoCapture(cam_id_right)
    if cap_left.isOpened() and cap_right.isOpened():
        while 1:
            retl, imgl = cap_left.read()
            retr, imgr = cap_right.read()
            cv2.imshow("img",np.hstack([imgl,imgr]))
            key = cv2.waitKey(30)
            if key == 27:
                break
            elif key == ord('s'):
                fl = '%s/left%06d.png'%(img_dir,cnt)
                cv2.imwrite(fl,imgl)
                fr = '%s/right%06d.png'%(img_dir,cnt)
                cv2.imwrite(fr,imgr)
                print('save img %s and %s'%(fl,fr))
                cnt+=1
    else:
        bad_id = []
        if not cap_left.isOpened:
            bad_id.append(cam_id_left)
        if not cap_right.isOpened:
            bad_id.append(cam_id_right) 
        print("Cannot open camera "+str(bad_id))

if __name__ == '__main__':
    print('capture_left')
    capture_mono("data/left/",0)
    print('capture_right')
    capture_mono("data/right/",1)
    print('capture_stereo')
    capture_stereo("data/stereo",0,1)
