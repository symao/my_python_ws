#!/usr/bin/python3.4
import pdb
import tensorflow as tf
import numpy as np
import threading
import gym
import time
import os
import matplotlib.pyplot as plt
from collections import deque
import random
import cv2
import sys
from map_env import Environment

class DQN:
    def __init__(self, params):
        self.params = params
        self.network_name = 'qnet'
        self.sess = tf.Session()
        self.x = tf.placeholder('float', [None, params['width'],params['height'], 6],name=self.network_name + '_x')
        self.q_t = tf.placeholder('float', [None], name=self.network_name + '_q_t')
        self.actions = tf.placeholder("float", [None, 4], name=self.network_name + '_actions')
        self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')

        # Layer 1 (Convolutional)
        layer_name = 'conv1' ; size = 3 ; channels = 6 ; filters = 16 ; stride = 1
        self.w1 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o1 = tf.nn.relu(tf.add(self.c1,self.b1),name=self.network_name + '_'+layer_name+'_activations')

        # Layer 2 (Convolutional)
        layer_name = 'conv2' ; size = 3 ; channels = 16 ; filters = 32 ; stride = 1
        self.w2 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o2 = tf.nn.relu(tf.add(self.c2,self.b2),name=self.network_name + '_'+layer_name+'_activations')
        
        o2_shape = self.o2.get_shape().as_list()        

        # Layer 3 (Fully connected)
        layer_name = 'fc3' ; hiddens = 256 ; dim = o2_shape[1]*o2_shape[2]*o2_shape[3]
        self.o2_flat = tf.reshape(self.o2, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')
        self.w3 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat,self.w3),self.b3,name=self.network_name + '_'+layer_name+'_ips')
        self.o3 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations')

        # Layer 4
        layer_name = 'fc4' ; hiddens = 4 ; dim = 256
        self.w4 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.y = tf.add(tf.matmul(self.o3,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')

        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))
        
        # if self.params['load_file'] is not None:
            # self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # else:
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # self.optim = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost,global_step=self.global_step)
        self.optim = tf.train.AdamOptimizer(self.params['lr']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep = 3)

        self.sess.run(tf.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('Loading checkpoint...%s' % self.params['load_file'])
            self.saver.restore(self.sess,self.params['load_file'])

        
    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)
        return cnt, cost

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)


MAP_SIZE = 12
MAX_STEP = 400

def env2input(env):
    r,c = env.map.shape[:2]
    obs_map = np.zeros((r,c),'float')
    obs_map[env.map<0] = 1
    score_map = env.map.astype('float')
    pos_map = np.zeros((r,c),'float')
    pos_map[int(env.posx),int(env.posy)] = 1
    score_map2 = np.copy(score_map)
    score_map2[score_map2<0] = 0

    # cv2.imshow('obs_map',cv2.resize(obs_map,(500,500)))
    # cv2.imshow('score_map',cv2.resize(score_map,(500,500)))
    # cv2.imshow('pos_map',cv2.resize(pos_map,(500,500)))
    # cv2.imshow('score_map2',cv2.resize(score_map2,(500,500)))
    # cv2.imshow('score_map3',cv2.resize(score_map3,(500,500)))
    # cv2.waitKey()
    # exit()

    return np.stack((obs_map, obs_map, obs_map, score_map, score_map2, pos_map), axis = 2)

params = {
    'width': MAP_SIZE,
    'height': MAP_SIZE,
    'num_training': 1,
    # Model backups
    'load_file': 'model/pacman_dqn-4980000_4040.0',#'model/pacman_dqn-2600000_195.0',
    'save_file': 'model/pacman_dqn',
    'save_interval' : 10000, 

    # Training parameters
    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 200000,     # Replay memory size

    'discount': 0.9,        # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
    # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

    # Epsilon value (epsilon-greedy)
    'eps': 1,             # Epsilon start value
    'eps_step': 0.0001,      # Epsilon decrease step
    'eps_final': 0.01,       # Epsilon end value

    'train_result': 'train_scores_dqn.txt'
}

def train_DQN():
    # Start Tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    qnet = DQN(params)

    Q_global = []
    cost_disp = 0     

    # Stats
    cnt = qnet.sess.run(qnet.global_step)
    local_cnt = 0
    rount_cnt = 0

    replay_mem = deque()


    list_score = []
    list_avgscore = []
    list_eps = []

    cur_eps = params['eps']

    # if params['train_result'] is not None:
    #     open(params['train_result'],'w').write('')

    time_start = time.time()

    env = Environment(MAP_SIZE)
    while True:
        env.reset()
        for istep in range(MAX_STEP):
            # get current state
            input_image = env2input(env)
            # make decision, get action
            if np.random.rand() > cur_eps:
                Q_pred = qnet.sess.run(qnet.y, feed_dict = {qnet.x: [input_image], qnet.q_t: np.zeros(1),
                                                qnet.actions: np.zeros((1, 4)),
                                                qnet.terminals: np.zeros(1),
                                                qnet.rewards: np.zeros(1)})[0]
                Q_global.append(max(Q_pred))
                a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
                action = a_winner[0][0] if len(a_winner)==1 else a_winner[np.random.randint(0, len(a_winner))][0]
            else:
                action = np.random.randint(0, 4)
            # do action, get reward
            _, reward = env.move(action)
            next_state = env2input(env)
            is_terminal = 1 if istep==MAX_STEP-1 else 0

            # push train samples
            replay_mem.append((input_image, reward, action, next_state, is_terminal))
            if len(replay_mem) > params['mem_size']:
                replay_mem.popleft()

            # train
            if (local_cnt > params['train_start']):
                batch = random.sample(list(replay_mem), params['batch_size'])
                batch_s = [] # States (s)
                batch_r = [] # Rewards (r)
                batch_a = [] # Actions (a)
                batch_n = [] # Next states (s')
                batch_t = [] # Terminal state (t)

                for i in batch:
                    batch_s.append(i[0])
                    batch_r.append(i[1])
                    batch_a.append(i[2])
                    batch_n.append(i[3])
                    batch_t.append(i[4])
                batch_s = np.array(batch_s)
                batch_r = np.array(batch_r)
                batch_a_onehot = np.zeros((params['batch_size'], 4))
                for i in range(len(batch_a)):                                           
                    batch_a_onehot[i][int(batch_a[i])] = 1      
                batch_a = np.array(batch_a_onehot)
                batch_n = np.array(batch_n)
                batch_t = np.array(batch_t)
                cnt, cost_disp = qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)
                if params['save_file'] is not None and cnt%10000 == 0:
                    save_file = params['save_file']+'-%d_%.1f'%(cnt,list_score[-1] if len(list_score)>0 else 0)
                    qnet.save_ckpt(save_file)
                    # print('train cnt:%d save_model:%s'%(cnt,save_file))
            local_cnt += 1

        list_score.append(env.score)
        list_avgscore.append(np.mean(list_score[-100:]))
        list_eps.append(cur_eps)

        open(params['train_result'],'a').write('%d %d\n'%(rount_cnt, env.score))

        if rount_cnt%100==0:
            print('%.2fh: round %d, avg:%d, eps:%.3f cnt:%d cost:%.2f'%((time.time() - time_start)/3600,
                rount_cnt, list_avgscore[-1], cur_eps, cnt, cost_disp))
        
        rount_cnt += 1
        cur_eps = max(params['eps_final'], cur_eps - params['eps_step'])


def test_DQN():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    qnet = DQN(params)
    env = Environment()

    for i in range(MAX_STEP):
        input_image = env2input(env)
        Q_pred = qnet.sess.run(qnet.y, feed_dict = {qnet.x: [input_image], qnet.q_t: np.zeros(1),
                                        qnet.actions: np.zeros((1, 4)),
                                        qnet.terminals: np.zeros(1),
                                        qnet.rewards: np.zeros(1)})[0]
        a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
        action = a_winner[0][0] if len(a_winner)==1 else a_winner[np.random.randint(0, len(a_winner))][0]
        env.move(action)
        env.watch()
        key = cv2.waitKey(100)
        if key==27:
            exit()
    cv2.waitKey()

class DQNInference:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.qnet = DQN(params)

def strategy_dqn(env, inference_obj):
    input_image = env2input(env)
    Q_pred = inference_obj.qnet.sess.run(inference_obj.qnet.y, feed_dict = {inference_obj.qnet.x: [input_image], inference_obj.qnet.q_t: np.zeros(1),
                                        inference_obj.qnet.actions: np.zeros((1, 4)),
                                        inference_obj.qnet.terminals: np.zeros(1),
                                        inference_obj.qnet.rewards: np.zeros(1)})[0]
    a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
    action = a_winner[0][0] if len(a_winner)==1 else a_winner[np.random.randint(0, len(a_winner))][0]
    return action

if __name__ == '__main__':
    # max score: 4300
    train_DQN()
    # test_DQN()

