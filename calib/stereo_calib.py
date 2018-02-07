import cv2
import os
import string
import numpy as np
import yaml
from calib_intrinsic import *

def select_data(calib_file_dir):
    left_img_files = sorted([x for x in os.listdir(calib_file_dir) if 'left' in x])
    right_img_files = sorted([x for x in os.listdir(calib_file_dir) if 'right' in x])

    for rf in right_img_files:
        lf = string.replace(rf,'right','left')
        if not os.path.exists(calib_file_dir+lf):
            os.remove(calib_file_dir+rf)

    i = 0
    while i<len(left_img_files):
        lf = left_img_files[i]
        rf = string.replace(lf,'left','right')
        if os.path.exists(calib_file_dir+lf) and os.path.exists(calib_file_dir+rf):
            imgl = cv2.imread(calib_file_dir+lf);
            imgr = cv2.imread(calib_file_dir+rf);
            cv2.imshow('left',imgl)
            cv2.imshow('right',imgr)
            k = cv2.waitKey()
            if k == 27:
                break
            elif k == ord('d'):
                os.remove(calib_file_dir+lf)
                os.remove(calib_file_dir+rf)
                print('delete %s and %s'%(calib_file_dir+lf,calib_file_dir+rf))
            elif k == ord('p'):
                i-=2
        else:
            if os.path.exists(calib_file_dir+lf):
                os.remove(calib_file_dir+lf)
            if os.path.exists(calib_file_dir+rf):
                os.remove(calib_file_dir+rf)
        i+=1

def show_rectify(imagelist1, imagelist2, K1, distortion1, K2, distortion2, img_size, R,T):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, distortion1, K2, distortion2, img_size, R, T, alpha = 0.1)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, distortion1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, distortion2, R2, P2, img_size, cv2.CV_32FC1)

    for idx,img1 in enumerate(imagelist1):
        img2 = imagelist2[idx]
        left_img_remap = cv2.remap(img1, map1x, map1y, cv2.INTER_LANCZOS4)
        right_img_remap = cv2.remap(img2, map2x, map2y, cv2.INTER_LANCZOS4)

        left_img_remap = cv2.cvtColor(left_img_remap,cv2.COLOR_GRAY2BGR)
        right_img_remap = cv2.cvtColor(right_img_remap,cv2.COLOR_GRAY2BGR)
        for line in range(0, int(right_img_remap.shape[0] / 20)):
            left_img_remap[line * 20, :] = (0, 0, 255)
            right_img_remap[line * 20, :] = (0, 0, 255)
        cv2.imshow('winname', np.hstack([left_img_remap, right_img_remap]))
        cv2.waitKey()

def stereo_calib_with_intrinsic(imgs1, imgs2, board_size, board_step, K1, D1, K2, D2, show = False):
    imagelist1 = []
    imagelist2 = []
    imagePoints1 = []
    imagePoints2 = []
    object_points = []

    board_cols, board_rows = board_size
    obj = np.zeros((board_rows*board_cols,3), np.float32)
    obj[:,:2] = np.mgrid[0:board_cols,0:board_rows].T.reshape(-1,2)*board_step
    # print(obj)

    half_patch = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    for idx,img1 in enumerate(imgs1):
        img2 = imgs2[idx];
        img_height, img_width  = img1.shape[:2]
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        
        found1, corners1 = cv2.findChessboardCorners(img1, board_size)
        found2, corners2 = cv2.findChessboardCorners(img2, board_size)

        if found1 and found2:
            cv2.cornerSubPix(gray1, corners1, (half_patch, half_patch), (-1, -1), criteria)
            cv2.drawChessboardCorners(img1, board_size, corners1, found1)
            cv2.cornerSubPix(gray2, corners2, (half_patch, half_patch), (-1, -1), criteria)
            cv2.drawChessboardCorners(img2, board_size, corners2, found2)
            if show:
                cv2.imshow('image1', img1)
                cv2.imshow('image2', img2)
                k = cv2.waitKey(0)
            else:
                k = 0
            if k==27:
                return
            elif k == ord('d'):
                pass
            else:
                imagePoints1.append(corners1)
                imagePoints2.append(corners2)
                object_points.append(obj)
                imagelist1.append(gray1)
                imagelist2.append(gray2)
        else:
            print('%d detect corners failed.'%idx,found1,found2)

    print('use %d image pairs, stereo calibrating...'%len(object_points))
    ret, K1, D1, K2, D2, R, T, E, F = \
                cv2.stereoCalibrate(object_points, imagePoints1, imagePoints2, K1, D1, K2, D2,\
                    (img_width, img_height), None, None, None, None,\
                    # cv2.CALIB_RATIONAL_MODEL)
                    cv2.CALIB_FIX_INTRINSIC)

    print('calibration done, reproject error: %.3f'%ret)

    print('K1=',K1)
    print('D1=',D1)
    print('K2=',K2)
    print('D2=',D2)
    print('R=',R)
    print('T=',T)

    if show:
        show_rectify(imagelist1, imagelist2, K1, D1, K2, D2, (img_width, img_height), R,T)

    return ret,K1,D1,K2,D2,R,T;

def stereo_calib(calib_file_dir, board_size, board_step):
    left_img_files = sorted([calib_file_dir+x for x in os.listdir(calib_file_dir) if 'left' in x])
    right_img_files = sorted([calib_file_dir+x for x in os.listdir(calib_file_dir) if 'right' in x])

    ret1,K1,distortion1 = calib_intrinsic([cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in left_img_files],board_size, board_step)
    ret2,K2,distortion2 = calib_intrinsic([cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in right_img_files],board_size, board_step)

    imagelist1 = []
    imagelist2 = []
    imagePoints1 = []
    imagePoints2 = []
    object_points = []

    obj = np.zeros((6*8,3), np.float32)
    obj[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*board_step
    # print(obj)

    half_patch = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for lf in left_img_files:
        rf = string.replace(lf,'left','right')
        img1 = cv2.imread(lf);
        img2 = cv2.imread(rf);
        img_height, img_width  = img1.shape[:2]
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        
        found1, corners1 = cv2.findChessboardCorners(img1, board_size)
        found2, corners2 = cv2.findChessboardCorners(img2, board_size)


        if found1 and found2:
            cv2.cornerSubPix(gray1, corners1, (half_patch, half_patch), (-1, -1), criteria)
            cv2.drawChessboardCorners(img1, board_size, corners1, found1)
            cv2.cornerSubPix(gray2, corners2, (half_patch, half_patch), (-1, -1), criteria)
            cv2.drawChessboardCorners(img2, board_size, corners2, found2)
            cv2.imshow('image1', img1)
            cv2.imshow('image2', img2)
            k = cv2.waitKey(1)
            if k==27:
                return
            elif k == ord('d'):
                os.remove(lf)
                os.remove(rf)
                print('delete %s and %s'%(lf,rf))
            else:
                imagePoints1.append(corners1)
                imagePoints2.append(corners2)
                object_points.append(obj)
                imagelist1.append(gray1)
                imagelist2.append(gray2)
        else:
            print('%s detect corners failed.'%lf,found1,found2)

    # ret, K1, distortion1, rvecs, tvecs = cv2.calibrateCamera(object_points, imagePoints1, (img_width,img_height),None,None)
    # print('calib intrinsic 1 rmse:%.3f'%ret)
    # ret, K2, distortion2, rvecs, tvecs = cv2.calibrateCamera(object_points, imagePoints2, (img_width,img_height),None,None)
    # print('calib intrinsic 2 rmse:%.3f'%ret)

    print('use %d image pairs, stereo calibrating...'%len(object_points))
    retval, K1, distortion1, K2, distortion2, R, T, E, F = \
                cv2.stereoCalibrate(object_points, imagePoints1, imagePoints2, K1, distortion1, K2, distortion2,\
                    (img_width, img_height), None, None, None, None,\
                    # cv2.CALIB_RATIONAL_MODEL)
                    cv2.CALIB_FIX_INTRINSIC)

    print('calibration done, reproject error: %.3f'%retval)

    print('K1=',K1)
    print('distortion1=',distortion1)
    print('K2=',K2)
    print('distortion2=',distortion2)
    print('R=',R)
    print('T=',T)

    show_rectify(imagelist1, imagelist2, K1, distortion1, K2, distortion2, (img_width, img_height), R,T)

    with open('stereo_calib.yaml','w') as f:
        yaml.dump({'K1':K1.tolist(),'distortion1':distortion1.tolist(),\
                   'K2':K2.tolist(),'distortion2':distortion2.tolist(),\
                   'R':R.tolist(),'T':T.tolist()},f)


if __name__ == '__main__':
    # board_size = (7,5)
    # board_step = 0.03
    board_size = (8,6)
    board_step = 0.045
    # calib_file_dir = "data_stereo/"
    # select_data(calib_file_dir)

    # left_img_files = sorted([calib_file_dir+x for x in os.listdir(calib_file_dir) if 'left' in x])
    # cnt = 0
    # for lf in left_img_files:
    #     rf = string.replace(lf,'left','right')
    #     os.rename(lf,'%sleft%d.jpg'%(calib_file_dir,cnt))
    #     os.rename(rf,'%sright%d.jpg'%(calib_file_dir,cnt))
    #     cnt+=1

    # stereo_calib(calib_file_dir, board_size, board_step)


    img_dir = 'data/left/'
    left_imgs = [cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in sorted([img_dir+x for x in os.listdir(img_dir)])]
    img_dir = 'data/right/'
    right_imgs = [cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in sorted([img_dir+x for x in os.listdir(img_dir)])]
    img_dir = 'data/stereo/'
    imgs1 = []
    imgs2 = []
    for fl in sorted([img_dir+x for x in os.listdir(img_dir) if 'left' in x]):
        rl = fl.replace('left','right')
        if os.path.exists(rl):
            imgs1.append(cv2.imread(fl))
            imgs2.append(cv2.imread(rl))

    ret1, K1, D1 = calib_intrinsic(left_imgs, board_size, board_step)
    ret2, K2, D2 = calib_intrinsic(right_imgs, board_size, board_step)

    ret, K1, D1, K2, D2, R, T = stereo_calib_with_intrinsic(imgs1, imgs2, board_size, board_step, K1, D1, K2, D2, True)
    img_size = (left_imgs[0].shape[1],left_imgs[0].shape[0])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T, alpha = 0.1)
    fs = cv2.FileStorage("stereo_calib.yml", cv2.FILE_STORAGE_WRITE)
    fs.write("K1",K1)
    fs.write("K2",K2)
    fs.write("D1",D1)
    fs.write("D2",D2)
    fs.write("R",R)
    fs.write("T",T)
    fs.write("R1",R1)
    fs.write("R2",R2)
    fs.write("P1",P1)
    fs.write("P2",P2)
    fs.write("Q",Q)
    fs.write("Sz",(img_size[1],img_size[0]))
