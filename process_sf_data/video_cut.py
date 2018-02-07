import cv2
import numpy as np
import os
import sys

def cut_sub_video(video_file, start_idx, end_idx, save_file):
    cap = cv2.VideoCapture(video_file)
    for i in range(start_idx+1):
        ret, frame = cap.read()

    rows,cols = frame.shape[:2]
    writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"XVID"), 30, (cols,rows))
    idx = start_idx
    while ret and idx<end_idx:
        writer.write(frame)
        ret, frame = cap.read()
        idx+=1

def grab_video_from_video(video_file, ts_file=None, imu_file=None):
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
    cv2.namedWindow('image')
    cv2.createTrackbar('frame_idx','image',frame_idx, image_cnt-1, lambda x:x)

    print('================ video capture =============')
    print('b: select begin frame   e: select end frame ')
    print('s: save video                               ')

    while True:
        frame_idx = cv2.getTrackbarPos('frame_idx','image')
        cv2.imshow('image', images[frame_idx])
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
            if os.path.isabs(save_file) and not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            cut_sub_video(video_file,start_idx,end_idx,save_file)
            
            if ts_file != None and os.path.exists(ts_file):
                tstamp = open(ts_file).readlines()
                save_txt_file = os.path.splitext(save_file)[0]+'ts.txt'
                with open(save_txt_file,'w') as fw:
                    for i in range(start_idx,end_idx):
                        fw.write(tstamp[i])
                
                if imu_file != None and os.path.exists(imu_file):
                    start_ts = float(tstamp[start_idx])
                    imu_lines = open(imu_file).readlines()
                    save_txt_file = os.path.splitext(save_file)[0].replace('img','imu')+'.txt'
                    for imu_idx in range(len(imu_lines)):
                        if float(imu_lines[imu_idx].split(' ')[0])>start_ts:
                            break
                    if imu_idx > 0:
                        imu_idx-=1
                    with open(save_txt_file,'w') as fw:
                        for i in range(imu_idx, len(imu_lines)):
                            fw.write(imu_lines[i])

            print('save video to "%s", start-end idx:[%d %d], press "esc" to exit'%(save_file,start_idx,end_idx))
        elif key==27:
            break

if __name__ == '__main__':
    if len(sys.argv) == 2:
        grab_video_from_video(sys.argv[1])
    elif len(sys.argv) == 3:   
        grab_video_from_video(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:   
        grab_video_from_video(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Usage: grab_video.py [data_file]')
        print('   or: grab_video.py [data_file] [ts_file]')
