import cv2
import tf
import numpy as np
from numpy.core.umath import deg2rad
import pdb

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

for idx in range(275,2000):
    i_data_vicon = raw_data_vicon[idx]
    i_data_algor = raw_data_algor[idx]
    i_data_vicon[5:8] = map(deg2rad,i_data_vicon[5:8])
    i_data_vicon[2:5] = map(lambda x:x/1000,i_data_vicon[2:5])

    T_logo2cam = tf.transformations.quaternion_matrix(i_data_algor[4:7]+i_data_algor[3:4]).T
    T_logo2cam[:3,3] = np.array(i_data_algor[:3])

    T_body2vicon = tf.transformations.euler_matrix(i_data_vicon[5],i_data_vicon[6],i_data_vicon[7],'rxyz').T
    T_body2vicon[:3,3] = i_data_vicon[2:5]

    T_body2cam = np.matmul(np.matmul(T_logo2cam,T_vicon2logo),T_body2vicon)
    # print(T_logo2cam)
    # print(T_vicon2logo)
    # print(T_body2vicon)

    print(T_body2cam)
    # print(idx,tf.transformations.euler_from_matrix(T_body2cam[:3,:3]))
    # euler = tf.transformations.euler_from_matrix(T_body2vicon[:3,:3])
    # print("%d %.3f %.3f %.3f"%(idx,euler[0],euler[1],euler[2]))

    cv2.imshow('img',images[idx])
    cv2.waitKey()

# t_body2vicon = np.array([-7504.000000, 2756.000000, 1786.000000])/1000.0
# R_vicon2body = 

