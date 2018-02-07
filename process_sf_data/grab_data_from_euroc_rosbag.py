import rosbag
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import sys

if __name__ == '__main__':
    bag_file = '/home/symao/data/euroc/MH_04_difficult.bag'
    bag = rosbag.Bag(bag_file)
    bridge = CvBridge()
    images = []
    save_dir = os.path.splitext(bag_file)[0]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = None
    f_imgts = open(os.path.join(save_dir,'imgts.txt'),'w')
    f_imu = open(os.path.join(save_dir,'imu.txt'),'w')

    for topic, msg, t in bag.read_messages(topics=['/cam0/image_raw', '/imu0']):
        if topic == '/imu0':
            f_imu.write('%f %f %f %f %f %f %f\n'%(msg.header.stamp.to_sec(),
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))

        if topic == '/cam0/image_raw':
            img = bridge.imgmsg_to_cv2(msg, "bgr8")
            ts = msg.header.stamp
            print(ts.to_sec())
            r,c = img.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(save_dir,'img.avi'), cv2.VideoWriter_fourcc(*"XVID"), 30, (c,r))
            writer.write(img)
            f_imgts.write('%f\n'%ts.to_sec())

    bag.close()

