import pdb
import numpy as np
import cv2
from map_env import Environment
import copy

class GridmapDijkstra:
    def __init__(self, gridmap=None, cur_pos=None, max_step=10):
        if gridmap is None or cur_pos is None:
            pass
        else:
            self.set_map(gridmap, cur_pos, max_step)

    def set_map(self, gridmap, cur_pos, max_step = 10):
        r,c = gridmap.shape[:2]
        foo_obs = lambda x,y: gridmap[int(x),int(y)]<0
        # distance map: >max_step:invalid  0,1,2,3...distance to current position
        self.distmap = np.zeros((r,c),'int')+ max_step + 1000
        self.distmap[cur_pos] = 0
        # direction map: -1:invalid  -2: indicate current position 0~3:indicate optimal path dir from parent to current cell
        self.dirmap = np.zeros((r,c),'int') - 1
        self.dirmap[cur_pos] = -2 # special value to represent origin

        bound_list = [cur_pos]
        for d in range(1, 1+max_step):
            new_bound = []
            for p in bound_list:
                for x,y in np.array([(0,-1),(0,1),(1,0),(-1,0)]) + p:
                    q = (x,y)
                    if x<0 or y<0 or x>=r or y>=c or foo_obs(x,y) or self.distmap[q]<=max_step:
                        continue
                    self.distmap[q] = d
                    self.dirmap[q] = self.__path2dir(p,q)
                    new_bound.append(q)
            bound_list = copy.deepcopy(new_bound)

    def dist2target(self, t):
        return self.distmap[t]

    def path2target(self, t):
        r,c = self.dirmap.shape[:2]
        path = []
        x,y = t
        while self.dirmap[x,y] != -2:
            if x<0 or y<0 or x>=r or y>=c:
                return []
            curdir = self.dirmap[x,y]
            path.append((x,y))

            if curdir==0:
                x+=1
            elif curdir==1:
                x-=1
            elif curdir==2:
                y+=1
            elif curdir==3:
                y-=1
            else:
                return []
        return path[::-1]

    def __path2dir(self, cur_pos, next_pos):
        dx = next_pos[0] - cur_pos[0]
        dy = next_pos[1] - cur_pos[1]
        if dx==-1 and dy==0:
            return 0
        elif dx==1 and dy==0:
            return 1
        elif dx==0 and dy==-1:
            return 2
        elif dx==0 and dy==1:
            return 3
        else:
            return -1

mouse_x = -1
mouse_y = -1
def on_mouse(event,x,y,flags,param):
    global mouse_x,mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y

if __name__ == '__main__':
    winname = 'astar'
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname,on_mouse)
    map_size = 20
    sim = Environment(map_size)
    # sim.load('maps/002.map')

    dijkstra = GridmapDijkstra()
    dijkstra.set_map(sim.map, (sim.posx,sim.posy), map_size*map_size)

    while True:
        K = 20
        drawimg = sim.get_show_map(K)
        if mouse_x>0 and mouse_y>0:
            path = dijkstra.path2target((int(mouse_y/K),int(mouse_x/K)))
            if len(path)>0:
                print(path)
                for i,j in path:
                    drawimg[i*K:i*K+K, j*K:j*K+K] = (0,120,150)
        cv2.imshow(winname,drawimg)
        key = cv2.waitKey(100)
        if key == 27:
            exit()