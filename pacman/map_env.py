import numpy as np
import cv2

class Environment:
    def __init__(self, map_size = 12, obs_cnt = 20, pkg_cnt = 24):
        self.map_size = map_size
        self.reset(obs_cnt, pkg_cnt)

    def reset(self, obs_cnt = 20, pkg_cnt = 24):
        self.obs_cnt = obs_cnt
        self.pkg_cnt = pkg_cnt
        self.step = 0
        self.score = 0
        self.posx = 0
        self.posy = 0
        #map definition: 0-free  -1-occupy 1~10
        if False: #use random map
            while True:
                self.map = np.zeros((self.map_size,self.map_size),'int')
                self.map[tuple(np.random.randint(0,self.map_size,obs_cnt)),tuple(np.random.randint(0,self.map_size,obs_cnt))] = -1
                self.map[0,0] = 0
                if self.__check_connect_domain(self.map, self.posx, self.posy, lambda x:x<0, self.map_size*1.2)>=self.map_size*1.2:
                    break
        else: #use fix map
            self.map = np.zeros((self.map_size,self.map_size),'int')
            walls = [(1,2),(2,3),(4,3),(4,4),(9,6),(8,6),(6,5),(6,8),(6,9),(3,9),(2,9),(2,10),(9,2),(8,2)]
            for x,y in walls:
                self.map[x,y] = -1
        self.__add_package(self.pkg_cnt)

    def restart(self):
        self.step = 0
        self.score = 0
        self.posx = 0
        self.posy = 0
        self.map[self.map>0] = 0
        self.__add_package(self.pkg_cnt)        

    def load(self, file):
        data = [int(x) for x in open(file).readline().strip().split(' ')]
        self.map_size = data[0]
        self.map = np.reshape(data[1:],(self.map_size,self.map_size))
        self.restart()

    def save(self, file):
        open(file,'w').write(' '.join([str(x) for x in [self.map_size]+self.map.ravel().tolist()]))

    # move dir (0,1,2,3) => (up, down, left, right)
    def move(self, movdir):
        r,c = self.map.shape[:2]
        invalid_move = False
        if movdir == 0: # up
            if self.posx>0 and self.map[self.posx-1,self.posy]>=0:
                self.posx -= 1
            else:
                invalid_move = True
        elif movdir == 1: #down
            if self.posx<r-1 and self.map[self.posx+1,self.posy]>=0:
                self.posx += 1
            else:
                invalid_move = True
        elif movdir == 2: # left
            if self.posy>0 and self.map[self.posx,self.posy-1]>=0:
                self.posy -= 1
            else:
                invalid_move = True
        elif movdir == 3: #right
            if self.posy<c-1 and self.map[self.posx,self.posy+1]>=0:
                self.posy += 1
            else:
                invalid_move = True

        self.step += 1
        cur_score = self.map[self.posx,self.posy]
        if cur_score > 0:
            self.score += cur_score
            self.map[self.posx,self.posy] = 0

        #update map scores
        self.map[self.map>0] -= 1

        #add package to make sure package count always the same
        for i in range(self.pkg_cnt - np.sum(self.map>0)):
            if np.random.random()<0.5:
                self.__add_package(2)

        return invalid_move, cur_score

    def __add_package(self, pkg_cnt = 1):
        r,c = self.map.shape[:2]
        cnt = pkg_cnt
        while cnt>0:
            i = np.random.randint(r)
            j = np.random.randint(c)
            if self.map[i,j] == 0 and i!=self.posx and j!=self.posy:
                self.map[i,j] = np.random.randint(18,49)
                cnt-=1

    def __check_connect_domain(self,obs_map,x,y,foo_obs,max_search=9999):
        check_map = np.zeros_like(obs_map)
        r,c = check_map.shape[:2]
        ptlist = [(x,y)]
        check_map[x,y] = 1
        cnt = 0
        for pt in ptlist:
            cnt += 1
            if cnt > max_search:
                return cnt
            for dx,dy in [[1,0],[-1,0],[0,1],[0,-1]]:
                p = (pt[0]+dx,pt[1]+dy)
                if p[0]<0 or p[1]<0 or p[0]>=r or p[1]>=c or foo_obs(obs_map[p]) or check_map[p]>0:
                    continue
                check_map[p] = 1
                ptlist.append(p)
        return cnt

    def get_show_map(self, K=20):
        r,c = self.map.shape[:2]
        drawimg = cv2.cvtColor(np.ones((r*K,c*K),'uint8')+254, cv2.COLOR_GRAY2BGR)
        for i in range(c):
            drawimg = cv2.line(drawimg, (0,i*K), (r*K,i*K), (250,0,0), 1)
        for i in range(r):
            drawimg = cv2.line(drawimg, (i*K,0), (i*K,c*K), (250,0,0), 1)

        for i in range(r):
            for j in range(c):
                if self.map[i,j]>0:
                    drawimg[i*K:i*K+K, j*K:j*K+K] = (0, min(170*self.map[i,j]/20 + 80,255), 0)
                    drawimg = cv2.putText(drawimg, '%d'%self.map[i,j], (int((j+0.1)*K),int((i+0.8)*K)), cv2.FONT_HERSHEY_SIMPLEX, K*0.02, (255,0,0), 2)
                elif self.map[i,j]<0:
                    drawimg[i*K:i*K+K, j*K:j*K+K] = (50, 50, 50)
        
        drawimg = cv2.circle(drawimg, (int((self.posy+0.5)*K),int((self.posx+0.5)*K)), int(K/3), (0,0,255), -1)
        
        status_bar = np.ones((30,c*K,3),'uint8')+128
        status_bar = cv2.putText(status_bar, 'step:%d score:%d'%(self.step,self.score), (0,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        return np.vstack([drawimg,status_bar])

    def watch(self):
        cv2.imshow('map', self.get_show_map())

if __name__ == '__main__':
    for epoch in range(100):
        map_size = 20
        env = Environment()
        while env.step<=500:
            env.reset()
            env.watch()
            key = cv2.waitKey()
            if key == 27:
                exit()
            env.move(np.random.choice([0,1,2,3]))
