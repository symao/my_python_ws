import numpy as np
import cv2
import os
from map_env import Environment

def on_mouse(event,x,y,flags,param):
    param.mouse_cb(event,x,y,flags)

class MapCreator:
    def __init__(self, map_size = 12, obs_cnt = 20, pkg_cnt = 5):
        self.env = Environment(map_size, obs_cnt, pkg_cnt)
        self.K = 20
        self.left_mouse_down = False
        self.right_mouse_down = False

    def mouse_cb(self,event,x,y,flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_mouse_down = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_mouse_down = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.right_mouse_down = False

        r = int(y/self.K)
        c = int(x/self.K)
        if r<0 or c<0 or r>=self.env.map.shape[0] or c>=self.env.map.shape[1]:
            return
        if self.left_mouse_down:
            self.env.map[r,c] = -1
        elif self.right_mouse_down:
            self.env.map[r,c] = 0

    def watch(self, winname='map'):
        cv2.namedWindow(winname)
        cv2.setMouseCallback(winname, on_mouse, self)
        K = self.K
        while True:
            img = self.env.get_show_map(K)
            cv2.imshow(winname, img)
            key = cv2.waitKey(100)
            if key == 27 or key == ord('s') or key == 32 or key == 13 or key == 10:
                return key
            elif key == ord('r'):
                self.env.reset(self.env.obs_cnt,self.env.pkg_cnt)
            elif key == ord('c'):
                self.env.map[:] = 0

def create_maps():
    save_path = 'maps'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    MAP_SIZE = 12
    OBS_CNT = 20
    mc = MapCreator(MAP_SIZE, OBS_CNT, 0)
    for i in range(0,200):
        mc.env.reset(np.random.randint(5,50), 0)
        save_file = os.path.join(save_path,'%03d.map'%i)
        if os.path.exists(save_file):
            mc.env.load(save_file)
            print('%s exists, load it'%save_file)
        key = mc.watch()
        if key==ord('s'):
            mc.env.save(save_file)
            print('save map %s'%save_file)
        elif key==27:
            break

def modify_map(map_file):
    if not os.path.exists(map_file):
        print('no files, %s'%map_file)
        return
    mc = MapCreator()
    mc.env.load(map_file)
    key = mc.watch()
    if key==ord('s'):
        mc.env.save(map_file)
        print('save map %s'%map_file)

if __name__ == '__main__':
    create_maps()
    # modify_map('maps/001.map')
