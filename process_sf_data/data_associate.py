import numpy as np

img_lines = [x.strip() for x in open('/home/symao/data/uav/20171011/12/cut.txt','r').readlines()]
imu_lines = [x.strip() for x in open('/home/symao/data/uav/20171011/12/20170928_183515_imu.txt','r').readlines()]

img_list = [float(x) for x in img_lines]
imu_list = [[float(y) for y in x.split(' ')[:-1]] for x in imu_lines]

img_asscoiate = [-1]*len(img_list)
img_error = [0]*len(img_list)

max_diff = 0.01

potential_matches = [(a-b,ia,ib) for ia,a in enumerate(img_list)
                                    for ib,b in enumerate([x[0] for x in imu_list])
                                    if abs(a-b)<max_diff]

sorted(potential_matches, key=lambda x:abs(x[0]))

for e,ia,ib in potential_matches:
    if img_asscoiate[ia] == -1:
        img_asscoiate[ia] = ib
        img_error[ia] = e


with open('associate.txt','w') as fp:
    for i,t in enumerate(img_asscoiate):
        if t>-1:
            fp.write(img_lines[i]+' '+imu_lines[t]+'\n')
        else:
            fp.write(img_lines[i]+' 0'*len(imu_lines[t].split(' '))+'\n')