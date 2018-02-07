import matplotlib.pyplot as plt
import numpy as np

datadir = '/home/symao/data/mynteye/data(1)/'

img_ts = [float(x.strip()) for x in open(datadir+'imgts.txt','r').readlines()]
img_ts = np.array(img_ts[:-1])

imu_data = [[float(y) for y in s.strip().split(' ')] for s in open(datadir+'imu.txt','r').readlines()]
imu_data = np.array(imu_data[:-1])

print(imu_data.shape)

img_diff = img_ts[1:]-img_ts[:-1]
imu_diff = imu_data[1:,0] - imu_data[:-1,0]
# imu_diff2 = imu_data[1:,1] - imu_data[:-1,1] + 0.01

plt.subplot(3,1,1)
plt.plot(img_diff)
plt.subplot(3,1,2)
plt.plot(imu_diff)
# plt.plot(imu_diff2)
plt.subplot(3,1,3)
plt.plot(imu_data[:,1:])
plt.show()