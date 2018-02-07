import pdb
import numpy as np
import cv2
import requests
from flask import json

class CompetitionEnvironment:
    def __init__(self, map_size = 12):
        self.team_id = 'KOPB'
        # self.url = 'http://10.2.5.64/test'
        self.url = 'http://10.2.5.64/competition'
        self.map_size = map_size
        self.reset()

    def reset(self):
        # set the vars below
        self.step = 0
        self.score = 0

        while True:
            data = requests.post(self.url, data=json.dumps({'name':self.team_id})).json()
            if(data['msg']=='OK'):
                break
            else:
                print('connect err')
        self.env_id = data['id']
        self.parse_state(data['state'])

    def parse_state(self, state_data):
        self.posx = state_data['ai']['x']
        self.posy = state_data['ai']['y']
        self.map = np.zeros((self.map_size,self.map_size),'int')

        self.obs_cnt = len(state_data['walls'])
        for obs in state_data['walls']:
            self.map[obs['x'],obs['y']] = -1

        self.pkg_cnt = len(state_data['jobs'])
        for pkg in state_data['jobs']:
            self.map[pkg['x'],pkg['y']] = pkg['value']

    def restart(self):
        # set the vars below
        self.reset()

    def save(self, file):
        open(file,'w').write(' '.join([str(x) for x in [self.map_size]+self.map.ravel().tolist()]))

    # move dir (0,1,2,3) => (up, down, left, right)
    def move(self, movdir):
        direction = list('UDLR')
        while True:
            data = requests.post(self.url+'/'+self.env_id+'/move', data=json.dumps({'direction':direction[movdir]})).json()
            if data['msg'] == 'OK':
                break
            else:
                print('mov err')
        self.env_id = data['id']
        self.parse_state(data['state'])
        reward = data['reward']
        done = data['done']
        self.step += 1
        self.score+=reward
        return done, reward

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
        env = CompetitionEnvironment()
        while env.step<=500:
            env.watch()
            key = cv2.waitKey(100)
            if key == 27:
                exit()
            env.move(np.random.choice([0,1,2,3]))
