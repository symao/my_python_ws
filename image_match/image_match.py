import cv2
import numpy as np
from math import *
import os
import string

class ImageMatch:
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __init__(self):
        self.tar_kpt = np.array([])
        self.tar_des = np.array([])
        self.tar_img = np.array([])
        self.K = np.float32([[500,0,320],[0,500,240],[0,0,1]])

    def setTargetImage(self,img,mask=None):
        self.tar_img = img.copy()
        self.tar_kpt, self.tar_des = self.orb.detectAndCompute(img, mask)
        r,c = img.shape[:2]
        self.K = np.float32([[500,0,c/2],[0,500,r/2],[0,0,1]])

    @staticmethod
    def featureMatch(kpt1,des1,kpt2,des2):
        matches = ImageMatch.bf.match(des1,des2)
        src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
        inlier_matches = [x for idx,x in enumerate(matches) if mask[idx]>0]
        return H, inlier_matches
    
    @staticmethod
    def recover3d(pts, img_size):
        return np.float32([[0.5-float(k[1])/img_size[1], float(k[0])/img_size[0]-0.5, 0] for k in pts])

    def matchImage(self, img, mask=None):
        cur_kpt, cur_des = self.orb.detectAndCompute(img, mask)
        H, matches = ImageMatch.featureMatch(self.tar_kpt, self.tar_des, cur_kpt, cur_des)

        pts3d = ImageMatch.recover3d([self.tar_kpt[m.queryIdx].pt for m in matches], (self.tar_img.shape[1],self.tar_img.shape[0]))
        pts2d = np.float32([cur_kpt[m.trainIdx].pt for m in matches])
        _,rvec,tvec,inliers = cv2.solvePnPRansac(pts3d, pts2d, self.K, None)

        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # tarh,tarw = self.tar_img.shape[:2]
        # corners = np.int32(cv2.perspectiveTransform(np.float32([[0,0],[tarw,0],[tarw,tarh],[0, tarh]]).reshape(1, -1, 2), H).reshape(-1, 2))
        corners,_ = cv2.projectPoints(np.float32([[0.5,-0.5,0],[0.5,0.5,0],[-0.5,0.5,0],[-0.5,-0.5,0]]), rvec, tvec, self.K, None)
        corners = np.int32(corners).reshape(-1,2)

        color_img = cv2.polylines(color_img, [corners], True, (0, 0, 255),2)
        show_img = cv2.drawMatches(self.tar_img,self.tar_kpt,color_img,cur_kpt,matches,None)
        cv2.imshow('matches',show_img)


if __name__ == '__main__':
    logo_file = '/home/symao/Pictures/logo/logo.jpg'
    logo_size = 200
    logo_img = cv2.resize(cv2.imread(logo_file, cv2.IMREAD_GRAYSCALE),(logo_size,logo_size))
    im = ImageMatch()
    im.setTargetImage(logo_img)

    tardir = '/home/symao/data/landmarkset/live_data/kitti_gray_qvga/train/images/'
    tarfiles = sorted([x for x in os.listdir(tardir) if '.png' in x])
    for f in tarfiles:
        img = cv2.imread(tardir+f, cv2.IMREAD_GRAYSCALE)
        x1,y1,x2,y2 = [int(float(x)) for x in open(tardir+'../labels/'+f.replace('.png','.txt')).readline().split(' ')[4:8]]
        mask = img.copy()*0
        cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
        im.matchImage(img,mask)

        # show_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(show_img, (x1,y1), (x2,y2), (0,0,255), -1)
        # cv2.imshow('img',mask)
        key = cv2.waitKey()
        if key == 27:
            break