import cv2
import os
import numpy as np
from math import sqrt

def calib_intrinsic(imgs, board_size, board_step, show = False):
    board_cols,board_rows = board_size
    obj = np.zeros((board_rows*board_cols,3), np.float32)
    obj[:,:2] = np.mgrid[0:board_cols,0:board_rows].T.reshape(-1,2)*board_step

    imgpoints = []
    objpoints = []

    # detect corners
    delete_idx = []
    for idx,img in enumerate(imgs):
        img = np.copy(img)
        img_height,img_width = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, board_size)
        if found==False:
            delete_idx.append(idx)
            continue

        cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(img, board_size, corners, found)

        if show:
            cv2.imshow('corners',img)
            cv2.waitKey(0)

        imgpoints.append(corners)
        objpoints.append(obj)

    imgs = [imgs[i] for i in range(len(imgs)) if i not in delete_idx]

    # calibration
    print('use %d images, calibrating...'%(len(imgpoints)))
    ret, K, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_width,img_height), None, None, None, None)

    # # rectify
    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,distortion,(img_width,img_height),1,(img_width,img_height))
    # mapx,mapy = cv2.initUndistortRectifyMap(K,distortion,None,newcameramtx,(img_width,img_height),5)
    # for img in imgs:
    #     img = np.copy(img)
    #     dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    #     cv2.imshow('rectify',dst)
    #     cv2.waitKey(1)


    # reproject error
    '''
    NOTE:this is RMSE, assume each point dist err is e_i, there are N points, 
    then RMSE = \sqrt{\sum_i^N(e_i*e_i)/N)}
    '''
    total_err = 0
    total_n = 0
    for i in range(len(objpoints)):
        img = np.copy(imgs[i])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        corners = imgpoints[i]
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, distortion)
        errs = [cv2.norm(t[0,:]) for t in (corners-imgpoints2)]
        total_err += sum([x*x for x in errs])
        total_n += len(errs)
        for k in range(len(corners)):
            pt1 = imgpoints2[k,0]
            pt2 = corners[k,0]
            img = cv2.line(img,tuple(pt1),tuple(pt2),(0,0,255),3)
        if show:
            cv2.imshow('reproject',img)
            cv2.waitKey(0)
    mean_error = sqrt(total_err/total_n)

    # with open("./intrinsic.yaml","w") as f:
    #     f.write('K: '+str(K)+'\n')
    #     f.write('D: '+str(distortion)+'\n')

    print('reproject_error:%f %f'%(ret,mean_error))
    print('K',K)
    print('D',distortion)

    return ret,K,distortion


if __name__ == '__main__':
    # board_rows = 6
    # board_cols = 8
    # board_step = 0.28
    board_rows = 5
    board_cols = 7
    board_step = 0.3
    board_sz = (board_cols,board_rows)

    # img_dir = '/home/symao/data/2017-8-17shiyan/calib/'
    # img_files = sorted([img_dir+x for x in os.listdir(img_dir) if 'right' in x])
    
    # img_dir = 'data_cam_left/'
    # img_files = sorted([img_dir+x for x in os.listdir(img_dir) if '.png' in x or '.jpg' in x])
    # imgs = [cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in img_files]

    # img_dir = 'data_stereo/'
    # img_files = sorted([img_dir+x for x in os.listdir(img_dir) if 'right' in x])
    # imgs = [cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in img_files]

    img_dir = 'data_logi_c920/'
    img_files = sorted([img_dir+x for x in os.listdir(img_dir)])
    imgs = [cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in img_files]

    calib_intrinsic(imgs, board_sz, board_step)