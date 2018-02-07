import rosbag
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Imu
import pdb
import numpy as np
import os
import sys

def write_rosbag(video_file,imgts_file,imu_file,save_file):
    imu_topic = 'cv_camera/image_left'
    img_topic = 'imu0'

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    bag = rosbag.Bag(save_file, 'w')
    bridge = CvBridge()

    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    frame_idx = 0
    imu_idx = 0

    imgts_data = [float(x) for x in open(imgts_file).readlines()]
    imu_data = [[float(x) for x in line.strip().split(' ')] for line in open(imu_file).readlines()]

    while ret and frame_idx<1000:
        while rospy.Time.from_sec(imu_data[imu_idx][0]) < rospy.Time.from_sec(imgts_data[frame_idx]):
            imu_msg = Imu()
            imu_ts = rospy.Time.from_sec(imu_data[imu_idx][0])
            imu_msg.header.stamp = imu_ts
            imu_msg.header.frame_id = '/mynt_imu_frame'
            imu_msg.linear_acceleration.x = imu_data[imu_idx][1] * 9.8
            imu_msg.linear_acceleration.y = imu_data[imu_idx][2] * 9.8
            imu_msg.linear_acceleration.z = imu_data[imu_idx][3] * 9.8
            imu_msg.linear_acceleration_covariance[0] = 0.04
            imu_msg.linear_acceleration_covariance[1] = 0
            imu_msg.linear_acceleration_covariance[2] = 0
            imu_msg.linear_acceleration_covariance[3] = 0
            imu_msg.linear_acceleration_covariance[4] = 0.04
            imu_msg.linear_acceleration_covariance[5] = 0
            imu_msg.linear_acceleration_covariance[6] = 0
            imu_msg.linear_acceleration_covariance[7] = 0
            imu_msg.linear_acceleration_covariance[8] = 0.04
            imu_msg.angular_velocity.x = imu_data[imu_idx][4] / 57.2956
            imu_msg.angular_velocity.y = imu_data[imu_idx][5] / 57.2956
            imu_msg.angular_velocity.z = imu_data[imu_idx][6] / 57.2956
            imu_msg.angular_velocity_covariance[0] = 0.02
            imu_msg.angular_velocity_covariance[1] = 0
            imu_msg.angular_velocity_covariance[2] = 0
            imu_msg.angular_velocity_covariance[3] = 0
            imu_msg.angular_velocity_covariance[4] = 0.02
            imu_msg.angular_velocity_covariance[5] = 0
            imu_msg.angular_velocity_covariance[6] = 0
            imu_msg.angular_velocity_covariance[7] = 0
            imu_msg.angular_velocity_covariance[8] = 0.02
            bag.write(imu_topic, imu_msg, imu_msg.header.stamp)
            imu_idx+=1

        img_msg = bridge.cv2_to_imgmsg(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),'mono8')
        img_msg.header.stamp = rospy.Time.from_sec(imgts_data[frame_idx])
        img_msg.header.frame_id = '/mynt_left_frame'
        bag.write(img_topic, img_msg, img_msg.header.stamp)
        print(frame_idx)
        ret, frame = cap.read()
        frame_idx+=1
    # bag.reindex()
    bag.close()

if __name__ == '__main__':
    video_file = '/home/symao/data/mynteye/20171107vins_outside/1/img.avi'
    imgts_file = '/home/symao/data/mynteye/20171107vins_outside/1/imgts.txt'
    imu_file = '/home/symao/data/mynteye/20171107vins_outside/1/imu.txt'
    save_file = '/home/symao/data/mynteye/20171107vins_outside/1/out.bag'

    write_rosbag(video_file,imgts_file,imu_file,save_file)
