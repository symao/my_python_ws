import matplotlib.pyplot as plt
import numpy as np

raw_data = np.array([[float(y) for y in s.strip().split(' ')]
    for s in open('/home/symao/data/uav/20170928/20170928_163012_associate_cut.txt','r').readlines()])
plt.subplot(2,1,1)
plt.plot(raw_data[:,2:5])
plt.subplot(2,1,2)
plt.plot(raw_data[:,5:8])
plt.show()