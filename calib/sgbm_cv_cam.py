import cv2
import os
import string
import numpy as np
import yaml

def nothing(x):
    pass

def read_stereo_video(video_file):
    cap = cv2.VideoCapture(video_file)
    imagelist1 = []
    imagelist2 = []
    ret, frame = cap.read()
    rows = frame.shape[0]/2
    while ret:
        imagelist1.append(cv2.cvtColor(frame[:rows,:,:],cv2.COLOR_BGR2GRAY))
        imagelist2.append(cv2.cvtColor(frame[rows:,:,:],cv2.COLOR_BGR2GRAY))
        ret, frame = cap.read()
    return imagelist1,imagelist2

def rectify_video(video_file,calib_yml):
    calibs = yaml.load(open(calib_yml).read())
    K1 = np.array(calibs['K1'])
    distortion1 = np.array(calibs['distortion1'])
    K2 = np.array(calibs['K2'])
    distortion2 = np.array(calibs['distortion2'])
    R = np.array(calibs['R'])
    T = np.array(calibs['T'])

    imagelist1,imagelist2 = read_stereo_video(video_file)
    rows,cols = imagelist1[0].shape[:2]
    img_size = (cols,rows)

    out_file = '_rectify.'.join(video_file.rsplit('.',1))
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"XVID"), 30, (cols,rows*2))

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, distortion1, K2, distortion2, img_size, R, T, alpha = 0.1)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, distortion1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, distortion2, R2, P2, img_size, cv2.CV_32FC1)

    for idx,img1 in enumerate(imagelist1):
        img2 = imagelist2[idx]
        left_img_remap = cv2.remap(img1, map1x, map1y, cv2.INTER_LANCZOS4)
        right_img_remap = cv2.remap(img2, map2x, map2y, cv2.INTER_LANCZOS4)
        left_img_remap = cv2.cvtColor(left_img_remap,cv2.COLOR_GRAY2BGR)
        right_img_remap = cv2.cvtColor(right_img_remap,cv2.COLOR_GRAY2BGR)
        writer.write(np.vstack([left_img_remap,right_img_remap]))


if __name__ == '__main__':
    calib_file_dir = "stereo_test/"
    # video_file = '/home/symao/data/stereo.avi'
    use_video = True
    video_file = '/home/symao/data/mynteye/20171107vins/img.avi'
    # rectify_video(video_file,'stereo_calib.yaml')
    # exit()

    # if 1:
    #     leftimgfiles = [calib_file_dir+x for x in os.listdir(calib_file_dir) if 'left' in x]
    #     imagelist1 = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in leftimgfiles]
    #     imagelist2 = [cv2.imread(x.replace('left','right'), cv2.IMREAD_GRAYSCALE) for x in leftimgfiles]
    # else:
    #     imagelist1,imagelist2 = read_stereo_video(video_file)

    fs = cv2.FileStorage("stereo_calib.yml", cv2.FILE_STORAGE_READ)
    K1 = fs.getNode("K1").mat()
    K2 = fs.getNode("K2").mat()
    D1 = fs.getNode("D1").mat()
    D2 = fs.getNode("D2").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()

    img_size = (640,480)

    # disparity range is tuned for 'aloe' image pair
    window_size = 11
    min_disp = 0
    num_disp = 96
    P1 = 8
    P2 = 32
    disp12MaxDiff = 20
    uniquenessRatio = 15
    speckleWindowSize = 100
    speckleRange = 32
    preFilterCap = 31
    mode = 1
    pre_method = 0
    pre_win = 5
    post_method = 0
    post_win = 5

    cv2.namedWindow('param setting')
    cv2.namedWindow('winname')
    cv2.createTrackbar('window_size','param setting',window_size,19,nothing)
    cv2.createTrackbar('num_disp','param setting',num_disp,192,nothing)
    cv2.createTrackbar('P1','param setting',8,10,nothing)
    cv2.createTrackbar('P2','param setting',32,40,nothing)
    cv2.createTrackbar('disp12MaxDiff','param setting',disp12MaxDiff,30,nothing)
    cv2.createTrackbar('uniquenessRatio','param setting',uniquenessRatio,100,nothing)
    cv2.createTrackbar('speckleWindowSize','param setting',speckleWindowSize,200,nothing)
    cv2.createTrackbar('speckleRange','param setting',speckleRange,255,nothing)
    cv2.createTrackbar('preFilterCap','param setting',preFilterCap,100,nothing)
    cv2.createTrackbar('mode','param setting',mode,2,nothing)
    cv2.createTrackbar('pre_method','param setting',pre_method,5,nothing)
    cv2.createTrackbar('pre_win','param setting',pre_win,20,nothing)
    cv2.createTrackbar('post_method','param setting',post_method,5,nothing)
    cv2.createTrackbar('post_win','param setting',post_win,20,nothing)


    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T, alpha = 0.1)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    if use_video:
        cap = cv2.VideoCapture(video_file)
    else:
        cap_left = cv2.VideoCapture(0)
        cap_right = cv2.VideoCapture(1)

    idx = 0
    while True:
        # img1 = imagelist1[idx]
        # img2 = imagelist2[idx]
        if use_video:
            ret,img = cap.read()
            if not ret:
                exit()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rows = img.shape[0]/2
            left_img_remap = img[:rows]
            right_img_remap = img[rows:]
        else:
            retl, img1 = cap_left.read()
            retr, img2 = cap_right.read()
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            left_img_remap = cv2.remap(img1, map1x, map1y, cv2.INTER_LANCZOS4)
            right_img_remap = cv2.remap(img2, map2x, map2y, cv2.INTER_LANCZOS4)
            
        window_size = cv2.getTrackbarPos('window_size','param setting')
        window_size = window_size if window_size%2 == 1 else window_size+1
        cv2.setTrackbarPos('window_size','param setting',window_size)

        num_disp = cv2.getTrackbarPos('num_disp','param setting')
        num_disp = num_disp if num_disp%16 == 0 else num_disp/16*16+16
        cv2.setTrackbarPos('num_disp','param setting',num_disp)

        P1 = cv2.getTrackbarPos('P1','param setting')
        P2 = cv2.getTrackbarPos('P2','param setting')
        P2 = P2 if P2>P1 else P1
        cv2.setTrackbarPos('P2','param setting',P2)

        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','param setting')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','param setting')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','param setting')
        speckleRange = cv2.getTrackbarPos('speckleRange','param setting')
        preFilterCap = cv2.getTrackbarPos('preFilterCap','param setting')
        mode = cv2.getTrackbarPos('mode','param setting')
        
        pre_method = cv2.getTrackbarPos('pre_method','param setting')
        pre_win = cv2.getTrackbarPos('pre_win','param setting')
        pre_win = pre_win if pre_win%2 == 1 else pre_win+1
        cv2.setTrackbarPos('pre_win','param setting',pre_win)

        post_method = cv2.getTrackbarPos('post_method','param setting')
        post_win = cv2.getTrackbarPos('post_win','param setting')
        post_win = post_win if post_win%2 == 1 else post_win+1
        cv2.setTrackbarPos('post_win','param setting',post_win)

        if pre_method==1:
            img1 = cv2.blur(img1,(pre_win,pre_win))
            img2 = cv2.blur(img2,(pre_win,pre_win))
        elif pre_method==2:
            img1 = cv2.medianBlur(img1,pre_win)
            img2 = cv2.medianBlur(img2,pre_win)
        elif pre_method==3:
            img1 = cv2.GaussianBlur(img1,(pre_win,pre_win),pre_win)
            img2 = cv2.GaussianBlur(img2,(pre_win,pre_win),pre_win)
        elif pre_method==4:
            img1 = cv2.bilateralFilter(img1,pre_win,pre_win*2,pre_win/2)
            img2 = cv2.bilateralFilter(img2,pre_win,pre_win*2,pre_win/2)

        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = window_size,
            P1 = P1*window_size**2,
            P2 = P2*window_size**2,
            # P1 = P1,
            # P2 = P2,
            disp12MaxDiff = disp12MaxDiff,
            uniquenessRatio = uniquenessRatio,
            speckleWindowSize = speckleWindowSize,
            preFilterCap = preFilterCap,
            speckleRange = speckleRange,
            mode = mode
            )
        
        # left_img_remap = cv2.imread('/home/symao/workspace/libelas/img/cones_left.pgm', cv2.IMREAD_GRAYSCALE)
        # right_img_remap = cv2.imread('/home/symao/workspace/libelas/img/cones_right.pgm', cv2.IMREAD_GRAYSCALE)
        # if aa>0:
        #     left_img_remap[:-aa] = left_img_remap[aa:]
        disp = stereo.compute(left_img_remap, right_img_remap).astype(np.float32)/16.0

        if post_method==1:
            disp = cv2.blur(disp,(post_win,post_win))
        elif post_method==2:
            if post_win>5:
                post_win = 5
            disp = cv2.medianBlur(disp,post_win)
        elif post_method==3:
            disp = cv2.GaussianBlur(disp,(post_win,post_win),post_win)
        elif post_method==4:
            disp = cv2.bilateralFilter(disp,post_win,post_win*2,post_win/2)

        left_img_remap = cv2.cvtColor(left_img_remap,cv2.COLOR_GRAY2BGR)
        right_img_remap = cv2.cvtColor(right_img_remap,cv2.COLOR_GRAY2BGR)
        for line in range(0, int(right_img_remap.shape[0] / 20)):
            if line%5==0:
                color = (0,0,255)
            elif line%5==1:
                color = (255,0,0)
            elif line%5==2:
                color = (0,255,0)
            elif line%5==3:
                color = (255,255,0)
            elif line%5==4:
                color = (255,0,255)
            left_img_remap[line * 20, :] = color
            right_img_remap[line * 20, :] = color

        cv2.imshow('disparity', (disp-min_disp)/num_disp)
        cv2.imshow('winname', np.hstack([left_img_remap, right_img_remap]))

        key = cv2.waitKey(30)
        if key == 27:
            break
        elif key>=81 and key<=84:
            continue
        elif key == ord('p'):
            idx = np.clip(idx-1, 0, 100000)
        # elif key == ord('s'):
        else:
            idx += 1

    cv2.destroyAllWindows()