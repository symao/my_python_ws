import numpy as np
import cv2
from map_env import Environment

class Node:  
    def __init__(self, parent, x, y, step, cost):  
        self.parent = parent  
        self.x = x
        self.y = y
        self.step = step
        self.cost = cost
    def __str__(self):
        return "<(%d,%d):%d:%d>"%(self.x,self.y,self.step,self.cost)

def get_best_idx(openlist):
    return np.argmin([p.cost for p in openlist])

def node_in_list(node, nodelist):
    for idx,n in enumerate(nodelist):
        if node.x==n.x and node.y==n.y:
            return idx
    return -1

def pathplan_astar(gridmap, cur, tar, max_step = 20):
    foo_obs = lambda x,y: gridmap[int(x),int(y)]<0
    foo_cost = lambda step,x,y: step + (abs(tx-x)+abs(ty-y))*1.1
    
    cx,cy = cur
    tx,ty = tar
    r,c = gridmap.shape[:2]
    if cx<0 or cy<0 or tx<0 or ty<0 or cx>=r or tx>=r or cy>=c or ty>=c or foo_obs(tx,ty):
        return []
    elif cx==tx and cy==ty:
        return []

    openlist = [Node(None, cx, cy, 0, foo_cost(0,cx,cy))]
    closelist = []

    ite = 0
    while len(openlist)>0 and ite<1000:
        ite += 1
        idx = get_best_idx(openlist)
        cur_node = openlist[idx]
        del openlist[idx]
        closelist.append(cur_node)
        if cur_node.step >= max_step:
            continue
        for x,y in [(cur_node.x+i,cur_node.y+j) for i,j in [(0,-1),(0,1),(1,0),(-1,0)]]:
            if x<0 or x>=r or y<0 or y>=c or foo_obs(x,y):
                continue
            n = Node(cur_node, x, y, cur_node.step+1, foo_cost(cur_node.step+1,x,y))
            if n.x==tx and n.y==ty:
                # find path
                path = []
                while n.parent!=None:
                    path.append((n.x,n.y))
                    n = n.parent
                return path[::-1]
            if node_in_list(n, closelist) >= 0:
                continue
            t = node_in_list(n, openlist)
            if t>=0 and n.cost<openlist[t].cost:
                openlist[t] = n
            else:
                openlist.append(n)
    return []

def pathplan_astar_with_score(gridmap, cur, tar, max_step = 20):
    foo_obs = lambda x,y: gridmap[int(x),int(y)]<0
    foo_hcost = lambda x,y: (abs(tx-x)+abs(ty-y))*1.01
    
    cx,cy = cur
    tx,ty = tar
    r,c = gridmap.shape[:2]
    if cx<0 or cy<0 or tx<0 or ty<0 or cx>=r or tx>=r or cy>=c or ty>=c or foo_obs(tx,ty):
        return []
    elif cx==tx and cy==ty:
        return []

    openlist = [Node(None, cx, cy, 0, foo_hcost(cx,cy))]
    closelist = []

    ite = 0
    while len(openlist)>0 and ite<1000:
        ite += 1
        idx = get_best_idx(openlist)
        cur_node = openlist[idx]
        del openlist[idx]
        closelist.append(cur_node)
        if cur_node.step >= max_step:
            continue
        for x,y in [(cur_node.x+i,cur_node.y+j) for i,j in [(0,-1),(0,1),(1,0),(-1,0)]]:
            if x<0 or x>=r or y<0 or y>=c or foo_obs(x,y):
                continue
            n = Node(cur_node, x, y, cur_node.step+1,
                cur_node.cost - foo_hcost(cur_node.x, cur_node.y) + foo_hcost(x,y) - 0.5*max(0, gridmap[int(x),int(y)]))
            if n.x==tx and n.y==ty:
                # find path
                path = []
                while n.parent!=None:
                    path.append((n.x,n.y))
                    n = n.parent
                return path[::-1]
            if node_in_list(n, closelist) >= 0:
                continue
            t = node_in_list(n, openlist)
            if t>=0 and n.cost<openlist[t].cost:
                openlist[t] = n
            else:
                openlist.append(n)
    return []

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
    sim = Environment(20)
    while True:
        K = 20
        drawimg = sim.get_show_map(K)
        if mouse_x>0 and mouse_y>0:
            path = pathplan_astar(sim.map,(0,0),(int(mouse_y/K),int(mouse_x/K)),100)
            if len(path)>0:
                for i,j in path:
                    drawimg[i*K:i*K+K, j*K:j*K+K] = (0,120,150)
        cv2.imshow(winname,drawimg)
        key = cv2.waitKey(100)
        if key == 27:
            exit()
