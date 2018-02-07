import pdb
import numpy as np
import cv2
import copy
from map_env import Environment
from dijkstra import GridmapDijkstra

class GridmapDijkstraOffline:
    def __init__(self, gridmap):
        print('calc pathes...')
        self.path_table = pathplan_ltb(gridmap)
        print('done')
    def pathplan(self, pstart, pend):
        return self.path_table[pstart[0]][pstart[1]][pend[0]][pend[1]]

def pathplan_ltb(gridmap):
    r,c = gridmap.shape[:2]
    path_table = []
    for x1 in range(r):
        l1 = []
        for y1 in range(c):
            l2 = []
            for x2 in range(r):
                l3 = []
                for y2 in range(c):
                    l3.append([])
                l2.append(l3)
            l1.append(l2)
        path_table.append(l1)

    for x1 in range(r):
        for y1 in range(c):
            dijkstra = GridmapDijkstra(gridmap, (x1, y1), 10000)
            for x2 in range(r):
                for y2 in range(c):
                    path = dijkstra.path2target((x2,y2))
                    path_table[x1][y1][x2][y2] = copy.deepcopy(path)
    return path_table

select_x = -1
select_y = -1
select_x2 = -1
select_y2 = -1
K = 20
def on_mouse(event,x,y,flags,param):
    global select_x,select_y,select_x2,select_y2
    if event == cv2.EVENT_LBUTTONDOWN:
        select_x = int(y/K)
        select_y = int(x/K)
        print('start:',select_x,select_y)
    if event == cv2.EVENT_RBUTTONDOWN:
        select_x2 = int(y/K)
        select_y2 = int(x/K)
        print('end:',select_x2,select_y2)


if __name__ == '__main__':
    winname = 'dijskra'
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname,on_mouse)
    env = Environment()
    planer = GridmapDijkstraOffline(env.map)

    while True:
        drawimg = env.get_show_map(K)
        if select_x>=0 and select_y>=0 and select_x2>=0 and select_y2>=0:
            print(select_x,select_y,select_x2,select_y2)
            path = planer.pathplan((select_x,select_y),(select_x2,select_y2))
            if len(path)>0:
                print(path)
                for i,j in path:
                    drawimg[i*K:i*K+K, j*K:j*K+K] = (0,120,150)
            drawimg[select_x*K:select_x*K+K, select_y*K:select_y*K+K] = (0,0,150)
        cv2.imshow(winname,drawimg)
        key = cv2.waitKey(100)
        if key == 27:
            exit()