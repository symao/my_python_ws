import cv2
import tf
import numpy as np
from numpy.core.umath import deg2rad
import pdb

def quat2dcm(q_wxyz):
    w,x,y,z = q_wxyz
    return tf.transformations.quaternion_matrix([x,y,z,w]).T[:3,:3]

def euler2dcm(rpy):
    x,y,z = rpy
    return tf.transformations.euler_matrix(x,y,z,'rxyz').T[:3,:3]

def video2images(video_file):
    cap = cv2.VideoCapture(video_file)
    images = []
    ret, frame = cap.read()
    while ret:
        images.append(frame)
        ret, frame = cap.read()
    cap.release()
    return images

raw_data_vicon = [map(float, x.strip().split(' ')) for x in open('/home/symao/data/uav/20170928/20170928_163012_associate_cut.txt','r').readlines()]
raw_data_algor = [map(float, x.strip().split(' ')) for x in open('/home/symao/data/uav/20170928/20170928_163012_cut.avi.txt','r').readlines()]

images = video2images('/home/symao/data/uav/20170928/20170928_163012_cut.avi')

T_logo2vicon = np.array([[0,0,-1,-10.2693], [0,1,0,2.86025], [1,0,0,1.6325], [0,0,0,1]])
T_vicon2logo = np.linalg.inv(T_logo2vicon)

pts3d = np.array([[0.5,-0.5,0],[0.5,0.5,0],[-0.5,0.5,0],[-0.5,-0.5,0]])
K = np.array([[810,0,329],[0,810,243.0],[0,0,1]])

first = True
for idx in range(275,2000):
    i_data_vicon = raw_data_vicon[idx]
    i_data_algor = raw_data_algor[idx]
    i_data_vicon[5:8] = map(deg2rad,i_data_vicon[5:8])
    i_data_vicon[2:5] = map(lambda x:x/1000,i_data_vicon[2:5])

    T_logo2cam = np.identity(4)
    T_logo2cam[:3,:3] = quat2dcm(i_data_algor[3:7])
    T_logo2cam[:3,3] = i_data_algor[:3]

    T_body2vicon = np.identity(4)
    T_body2vicon[:3,:3] = euler2dcm(i_data_vicon[5:8]).T
    T_body2vicon[:3,3] = i_data_vicon[2:5]

    if first:
        T_body2cam = np.matmul(np.matmul(T_logo2cam,T_vicon2logo),T_body2vicon)
        first = False
        # print("T_body2vicon=",T_body2vicon)
        # print("T_vicon2logo=",T_vicon2logo)
        # print("T_body2logo=",np.matmul(T_vicon2logo,T_body2vicon))
        # print("T_logo2cam=",T_logo2cam)
        print("T_body2cam=",T_body2cam)

    T_est_logo2cam = np.matmul(T_body2cam, np.matmul(np.linalg.inv(T_body2vicon),T_logo2vicon))
    print(T_est_logo2cam)

    rvec,_ = cv2.Rodrigues(T_est_logo2cam[:3,:3])
    tvec = T_est_logo2cam[:3,3]
    pts2d,_ = cv2.projectPoints(pts3d,rvec,tvec,K,None)
    pts2d = np.int32(pts2d).reshape(-1,2)

    img = np.copy(images[idx][:480])
    cv2.polylines(img, [pts2d], True, (0,0,255), 3)
    cv2.imshow('img',img)
    key = cv2.waitKey()
    if key==27:
        break

# t_body2vicon = np.array([-7504.000000, 2756.000000, 1786.000000])/1000.0
# R_vicon2body = 

