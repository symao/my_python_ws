import cv2
import numpy as np
import os
import sys

def get_input(info,default,default_type):
    t = raw_input(info)
    if t == '':
        return default_type(default)
    else:
        return default_type(t)

def cut_sub_video(video_file, start_idx, end_idx, save_file, bbox = None):
    stride = get_input('input stride[1]:',1,int)
    cap = cv2.VideoCapture(video_file)
    for i in range(start_idx+1):
        ret, frame = cap.read()
    if bbox != None:
        rows = bbox[3]-bbox[1]
        cols = bbox[2]-bbox[0]
    else:    
        rows,cols = frame.shape[:2]
    writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"XVID"), cap.get(cv2.CAP_PROP_FPS), (cols,rows))
    idx = start_idx
    while ret and idx<end_idx:
        if bbox != None:
            writer.write(frame[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        else:
            writer.write(frame)
        for i in range(stride):
            ret, frame = cap.read()
            idx+=1


bbox = [-1,-1,-1,-1]
select_mode = False

def mouse_cb(event,x,y,flags,param):
    global bbox,select_mode
    if flags & cv2.EVENT_FLAG_CTRLKEY:
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox[0] = x
            bbox[1] = y
            select_mode = True
        elif event == cv2.EVENT_LBUTTONUP:
            select_mode = False
        if select_mode == True:
            bbox[2] = x
            bbox[3] = y

def grab_video_from_video(video_file):
    global bbox
    cap = cv2.VideoCapture(video_file)
    images = []
    ret, frame = cap.read()
    r,c = frame.shape[:2]
    rate = min(320.0/c,480.0/r)
    while ret:
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),None,fx=rate,fy=rate)
        images.append(frame)
        ret, frame = cap.read()

    print('img:%d'%len(images))

    save_file = os.path.splitext(video_file)[0]+'_cut'+'.avi'
    image_cnt = len(images)
    frame_idx = 0
    start_idx = 0
    end_idx = image_cnt-1
    winname = 'image'
    cv2.namedWindow(winname)
    cv2.createTrackbar('frame_idx',winname,frame_idx, image_cnt-1, lambda x:x)
    cv2.setMouseCallback(winname,mouse_cb)

    print('================ video cut =================')
    print('b: select begin frame   e: select end frame ')
    print('ctrl+left mouse button: crop video')
    print('s: save video')

    while True:
        frame_idx = cv2.getTrackbarPos('frame_idx',winname)
        if bbox[0]>=0 and bbox[1]>=0 and bbox[2]>=0 and bbox[3]>=0:
            im_draw = cv2.cvtColor(images[frame_idx], cv2.COLOR_GRAY2BGR)
            im_draw = cv2.rectangle(im_draw, tuple(bbox[:2]),tuple(bbox[2:4]),(0,0,255),2)
            cv2.imshow(winname, im_draw)
        else:
            cv2.imshow(winname, images[frame_idx])
        key = cv2.waitKey(30);
        if key==ord('b'):
            start_idx = frame_idx
            print("select begin frame %d, press 'e' to select end frame.\n"%start_idx)
        elif key==ord('e'):
            end_idx = frame_idx
            print("select end frame %d, press 's' to save video.\n"%end_idx)
        elif key==ord('s'):
            b = raw_input('Input save path[%s]:'%save_file)
            save_file = b if not b == '' else save_file
            print('saving...')
            if bbox[0]>=0 and bbox[1]>=0 and bbox[2]>=0 and bbox[3]>=0:
                cut_sub_video(video_file,start_idx,end_idx,save_file,[int(x/rate) for x in bbox])
            else:
                cut_sub_video(video_file,start_idx,end_idx,save_file)
            print('save video to "%s", start-end idx:[%d %d], press "esc" to exit'%(save_file,start_idx,end_idx))
        elif key==27:
            break
    cv2.destroyWindow(winname)

def video_cut(video_file):
    global bbox
    cap = cv2.VideoCapture(video_file)
    images = []
    ret, frame = cap.read()
    r,c = frame.shape[:2]
    rate = min(320.0/c,480.0/r)
    while ret:
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),None,fx=rate,fy=rate)
        images.append(frame)
        ret, frame = cap.read()

    print('img:%d'%len(images))

    save_file = os.path.splitext(video_file)[0]+'_cut'+'.avi'
    image_cnt = len(images)
    frame_idx = 0
    start_idx = 0
    end_idx = image_cnt-1
    winname = 'image'
    cv2.namedWindow(winname)
    cv2.createTrackbar('frame_idx',winname,frame_idx, image_cnt-1, lambda x:x)
    cv2.setMouseCallback(winname,mouse_cb)

    print('================ video cut =================')
    print('b: select begin frame   e: select end frame ')
    print('ctrl+left mouse button: crop video')
    print('s: save video')

    while True:
        frame_idx = cv2.getTrackbarPos('frame_idx',winname)
        if bbox[0]>=0 and bbox[1]>=0 and bbox[2]>=0 and bbox[3]>=0:
            im_draw = cv2.cvtColor(images[frame_idx], cv2.COLOR_GRAY2BGR)
            im_draw = cv2.rectangle(im_draw, tuple(bbox[:2]),tuple(bbox[2:4]),(0,0,255),2)
            cv2.imshow(winname, im_draw)
        else:
            cv2.imshow(winname, images[frame_idx])
        key = cv2.waitKey(30);
        if key==ord('b'):
            start_idx = frame_idx
            print("select begin frame %d, press 'e' to select end frame.\n"%start_idx)
        elif key==ord('e'):
            end_idx = frame_idx
            print("select end frame %d, press 's' to save video.\n"%end_idx)
        elif key==ord('s'):
            break
            # b = raw_input('Input save path[%s]:'%save_file)
            # save_file = b if not b == '' else save_file
            # print('saving...')
            # if bbox[0]>=0 and bbox[1]>=0 and bbox[2]>=0 and bbox[3]>=0:
            #     cut_sub_video(video_file,start_idx,end_idx,save_file,[int(x/rate) for x in bbox])
            # else:
            #     cut_sub_video(video_file,start_idx,end_idx,save_file)
            # print('save video to "%s", start-end idx:[%d %d], press "esc" to exit'%(save_file,start_idx,end_idx))
        elif key==27:
            break
    cut_rect = (0,0,c,r)
    if bbox[0]>=0 and bbox[1]>=0 and bbox[2]>=0 and bbox[3]>=0:
        cut_rect = bbox
    cv2.destroyWindow(winname)
    bbox = [-1,-1,-1,-1]
    select_mode = False
    return start_idx, end_idx, cut_rect

if __name__ == '__main__':
    if len(sys.argv) == 2:
        grab_video_from_video(sys.argv[1])
    else:
        print('Usage: grab_video.py [data_file]')