import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import tf

def load_file(traj_file):
    return np.array([[float(i) for i in x.strip().split(' ')] for x in open(traj_file).readlines()[1:]])

def normalize_traj(data):
    if len(data)<=0:
        return
    def getT(i):
        T = tf.transformations.quaternion_matrix(data[i,[6,7,8,5]]).T
        T[:3,3] = data[i,[2,3,4]]
        return T
    def T2xyzq(T):
        qwxyz = tf.transformations.quaternion_from_matrix(T)[[3,0,1,2]]
        pxyz = T[:3,3]
        return pxyz,qwxyz
    T0_inv = np.linalg.inv(getT(0))
    for i in range(len(data)):
        T = np.dot(T0_inv, getT(i))
        p,q = T2xyzq(T)
        data[i,2:5] = p
        data[i,5:9] = q

from matplotlib import pyplot as plt   
from matplotlib import animation
from math import sin
import numpy as np
import sys

def calc_dist(traj):
    dt = traj[1:,2:5]-traj[:-1,2:5]
    return np.sum(np.linalg.norm(dt,axis=1))

def main_dynamic():
    # first set up the figure, the axis, and the plot element we want to animate   
    figure = plt.figure() 
    ax=figure.add_subplot(111, projection='3d')

    # animation function.  this is called sequentially   
    def animate(i):
        data_svo2_binary = load_file('/home/symao/.ros/trajectory_svo2.0_binary.txt')
        # data_svo2_mono = load_file('/home/symao/.ros/trajectory_svo_mono.txt')
        data_svo2_stereo = load_file('/home/symao/.ros/trajectory_svo_stereo.txt')
        ax.cla()
        ax.plot(data_svo2_binary[:,2],data_svo2_binary[:,3],data_svo2_binary[:,4])
        # ax.plot(data_svo2_mono[:,2],data_svo2_mono[:,3],data_svo2_mono[:,4])
        ax.plot(data_svo2_stereo[:,2],data_svo2_stereo[:,3],data_svo2_stereo[:,4],'r')
        plt.legend(labels = ['binary', 'mono', 'stereo'], loc = 'best')
        plt.xlabel('x')
        plt.ylabel('y')

        print(calc_dist(data_svo2_binary))

    anim = animation.FuncAnimation(figure, animate,  frames=50, interval=1000)  
    plt.show()

def main_static():
    data_svo2_binary = load_file('/home/symao/.ros/trajectory_svo2.0_binary.txt')
    data_svo2_mono = load_file('/home/symao/.ros/trajectory_svo_mono.txt')
    data_svo2_stereo = load_file('/home/symao/.ros/trajectory_svo_stereo.txt')
    # normalize_traj(data_svo2_stereo)

    figure=plt.figure()
    ax=figure.add_subplot(111, projection='3d')
    
    ax.plot(data_svo2_binary[:,2],data_svo2_binary[:,3],data_svo2_binary[:,4])
    ax.plot(data_svo2_mono[:,2],data_svo2_mono[:,3],data_svo2_mono[:,4])
    ax.plot(data_svo2_stereo[:,2],data_svo2_stereo[:,3],data_svo2_stereo[:,4])
    plt.legend(labels = ['binary', 'mono', 'stereo'], loc = 'best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    main_dynamic()