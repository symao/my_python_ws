import cv2
import numpy as np
from math import *
import os
import string

logo_file = '/home/symao/Pictures/logo/logo.jpg'
logo_size = 400
logo_img = cv2.resize(cv2.imread(logo_file,cv2.IMREAD_GRAYSCALE),(logo_size,logo_size))

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def create_logolist(img,scales):
    logo_list = []
    for k in scales:
        timg = cv2.resize(img,None,fx=k,fy=k)
        kp, des = orb.detectAndCompute(timg,None)
        logo_list.append((timg,kp,des))
    return logo_list

def process_img(tar_img,logo_list):
    t1 = cv2.getTickCount()
    tar_kp, tar_des = orb.detectAndCompute(tar_img,None)
    t2 = cv2.getTickCount()
    keypt_cnt = len(tar_kp)
    if keypt_cnt<30:
        return
    
    imgbgr = cv2.cvtColor(tar_img,cv2.COLOR_GRAY2BGR)
    for logo_img,logo_kp,logo_des in logo_list:
        logo_height,logo_width = logo_img.shape
        thres = np.clip(logo_height/30.0+6,15,25)
        matches = bf.match(logo_des,tar_des)
        matches = sorted(matches, key = lambda x:x.distance)
        if(len(matches)<=thres):
            continue

        src_pts = np.float32([logo_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([ tar_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
        # center = cv2.perspectiveTransform(np.float32([[(logo_width-1)/2.0,(logo_height-1)/2.0]]).reshape(1, -1, 2), H).reshape(-1, 2)
        inliers = np.sum(mask)
        t3 = cv2.getTickCount()
        print('orb:%.2f ms  +  homography:%.2f ms = cost:%.2f ms'%(float(t2-t1)/cv2.getTickFrequency()*1000.0,
                                                          float(t3-t2)/cv2.getTickFrequency()*1000.0,
                                                          float(t3-t1)/cv2.getTickFrequency()*1000.0))
        if inliers<=thres:
            # imgshow = cv2.drawMatches(logo_img,logo_kp,imgbgr,tar_kp,matches, None, flags=2, matchesMask =mask.ravel().tolist())
            # cv2.imshow('match',imgshow)
            # cv2.waitKey()
            continue
        corners = np.float32([[0,0],[0, logo_height-1],[logo_width-1,logo_height-1],[logo_width-1,0],[logo_width/2.0,logo_height/2.0]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
        imgbgr = cv2.polylines(imgbgr, [corners[:-1]], True, (0, 0, 255),2)
        imgbgr = cv2.circle(imgbgr,tuple(corners[-1]),5,(0,255,0),-1)
        imgshow = cv2.drawMatches(logo_img,logo_kp,imgbgr,tar_kp,matches, None, flags=2, matchesMask =mask.ravel().tolist())
        cv2.imshow('match',imgshow)
        break

    cv2.imshow('live',imgbgr)

def demo_offline():
    logo_list = create_logolist(logo_img,[1.5,0.5,0.25])

    tardir = '/home/symao/data/landmarkset/live_data/kitti_format_qvga/train/images/'
    tarfiles = sorted([x for x in os.listdir(tardir) if '.png' in x])
    for idx,f in enumerate(tarfiles):
        if idx<276:
            continue
        print(idx)
        tar_file = tardir+f
        tar_img = cv2.imread(tar_file,cv2.IMREAD_GRAYSCALE)
        process_img(tar_img,logo_list)
        key = cv2.waitKey(0)
        if key == 27:
            break

def demo_online():
    logo_list = create_logolist(logo_img,[1,1.5,0.5,0.25,0.125])

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while 1:
            ret, img = cap.read()
            process_img(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),logo_list)
            key = cv2.waitKey(30)
            if key == 27:
                break
    else:
        print('ERROR: Camera open failed.')

if __name__ == '__main__':
    demo_online()
    # demo_offline()
    