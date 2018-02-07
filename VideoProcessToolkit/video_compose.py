import cv2
import numpy as np
import os
import sys

origin = [0,0]
raw_origin = [x for x in origin]
lbutton_down = False
lbutton_down_pos = []
def mouse_cb(event,x,y,flags,param):
    global origin,lbutton_down,lbutton_down_pos,raw_origin
    if event == cv2.EVENT_LBUTTONDOWN:
        lbutton_down_pos = [x,y]
        lbutton_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        lbutton_down = False
        raw_origin = [x for x in origin]
    if lbutton_down:
        dx = x - lbutton_down_pos[0]
        dy = y - lbutton_down_pos[1]
        new_x = raw_origin[0]+dx
        new_y = raw_origin[1]+dy
        origin[0] = np.clip(new_x,0,param[0]-1)
        origin[1] = np.clip(new_y,0,param[1]-1)

def video_compose(video1, video2, save_file):
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

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    rows1,cols1 = frame1.shape[:2]
    rows2,cols2 = frame2.shape[:2]

    if fps1!=fps2:
        print('Video fps not the same. %d!=%d'%(fps1,fps2))
        return

    winname = 'image'
    cv2.namedWindow(winname)
    cv2.createTrackbar('resize', winname, 100, 200, lambda x:x)
    cv2.setMouseCallback(winname,mouse_cb,(cols1,rows1))

    while True:
        img1 = np.copy(frame1)
        rate = (cv2.getTrackbarPos('resize',winname))/100.0
        img2 = cv2.resize(frame2, None, fx=rate, fy=rate)
        tr,tc = img2.shape[:2]
        r1 = float(rows1-origin[1])/tr
        r2 = float(cols1-origin[0])/tc
        if min(r1,r2)<1:
            img2 = cv2.resize(img2, None, fx=min(r1,r2), fy=min(r1,r2))
        w = 2
        img2[:,w-1] = 0
        img2[:,-w] = 0
        img2[w-1,:] = 0
        img2[-w,:] = 0
        b0,b1 = origin
        tr,tc = img2.shape[:2]
        img1[b1:b1+tr,b0:b0+tc] = img2

        cv2.imshow(winname,img1)
        key = cv2.waitKey(100)

        if key==27:
            break
        elif key == ord('s'):
            print('saving...')
            writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"XVID"), fps1, (cols1,rows1))
            while ret1:
                img1 = np.copy(frame1)
                if ret2:
                    img1[b1:b1+tr,b0:b0+tc] = cv2.resize(frame2, (tc,tr))
                    ret2, frame2 = cap2.read()
                writer.write(img1)
                ret1, frame1 = cap1.read()
            print('Save video in %s'%save_file)
            break


    # writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"XVID"), , (cols1,rows1))
    # while ret1:
    #     writer.write(frame1)
    #     ret1, frame1 = cap1.read()

    # while ret2:
    #     writer.write(frame2)
    #     ret2, frame2 = cap2.read()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        video_compose(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Usage: video_compose.py [video1] [video2] [save_file]')