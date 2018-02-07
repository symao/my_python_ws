from matplotlib import pyplot as plt   
from matplotlib import animation
from math import sin
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv)>1:
        score_file = str(sys.argv[1])
        # first set up the figure, the axis, and the plot element we want to animate   
        fig = plt.figure() 
        ax1 = fig.add_subplot(1,1,1, xlim=(0,100), ylim=(0, 10), title='training score')

        # animation function.  this is called sequentially   
        def animate(i):
            scores = [float(x.strip().split(' ')[1]) for x in open(score_file).readlines()]
            scores = scores[::10]
            mean_scores = [np.mean(scores[:i+1][-100:]) for i in range(len(scores))]
            # print(scores,mean_scores)
            ax1.cla()
            ax1.plot(scores)
            ax1.plot(mean_scores)

        anim1 = animation.FuncAnimation(fig, animate,  frames=50, interval=1000)  
        plt.show()