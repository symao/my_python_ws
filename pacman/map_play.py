from __future__ import print_function
import pdb
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import functools
import time
from map_env import Environment
from pathplan import pathplan_astar
from dijkstra_offline import GridmapDijkstraOffline
from compitition_env import CompetitionEnvironment

MAP_SIZE = 12
MAX_STEP = 400
OBS_CNT = 20
PKG_CNT = 24
GLOBAL_PLANER = GridmapDijkstraOffline(Environment().map)

def strategy_random_walk(env):
    # information you can get. NOTE:do not change these values
    cx = env.posx
    cy = env.posy
    cur_map = env.map
    cur_step = env.step
    cur_score = env.score

    # run your strategy

    # make your decision
    movedir = np.random.choice([0,1,2,3])
    return movedir

def path2dir(cur_pos, next_pos):
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

def strategy_greedy(env):
    # information you can get. NOTE:do not change these values
    cx = env.posx
    cy = env.posy
    cur_map = env.map
    cur_step = env.step
    cur_score = env.score

    # run your strategy
    targets = [(tx,ty) for tx,ty in np.reshape(np.where(cur_map>0),(2,-1)).T]
    max_value = -1
    best_dir = -1
    for tx,ty in targets:
        tar_score = cur_map[tx,ty]
        if abs(tx-cx)+abs(ty-cy)>tar_score:
            continue
        # path = pathplan_astar(cur_map, (cx,cy), (tx,ty), tar_score)
        path = GLOBAL_PLANER.pathplan((cx,cy), (tx,ty))
        path_len = len(path)
        if path_len==0:
            continue
        tar_value = tar_score-path_len
        tar_value /= path_len
        # print('tar (%d,%d) path:%d tar_value:%d'%(tx,ty,path_len,tar_value))
        if path_len>0 and tar_value>0 and tar_value>max_value:
            max_value = tar_value
            best_dir = path2dir((cx,cy),path[0])
    return best_dir if best_dir>=0 else np.random.choice([0,1,2,3])

def strategy_two_greedy_whole_score(env):
    # information you can get. NOTE:do not change these values
    cx = env.posx
    cy = env.posy
    cur_map = env.map
    cur_step = env.step
    cur_score = env.score

    # run your strategy
    targets = [(tx,ty) for tx,ty in np.reshape(np.where(cur_map>0),(2,-1)).T]
    targets = [(tx,ty) for tx,ty in targets if abs(tx-cx)+abs(ty-cy)<cur_map[tx,ty]]

    max_value = -1
    best_dir = -1
    for tx,ty in targets:
        tar_score = cur_map[tx,ty]
        # path = pathplan_astar(cur_map, (cx,cy), (tx,ty), tar_score)
        path = GLOBAL_PLANER.pathplan((cx,cy), (tx,ty))
        path_len = len(path)
        if path_len==0 or path_len>tar_score:
            continue
        heading_list = [(tx,ty)]
        tar_value = tar_score-path_len
        # tar_value /= path_len
        # print('tar (%d,%d) path:%d tar_value:%d'%(tx,ty,path_len,tar_value))
        if tar_value>max_value:
            max_value = tar_value
            best_dir = path2dir((cx,cy),path[0])

        for tx2,ty2 in targets:
            if (tx2,ty2) in heading_list:
                continue
            tar2_score = cur_map[tx2,ty2]
            # path2 = pathplan_astar(cur_map, (tx,ty), (tx2,ty2), tar2_score-path_len)
            path2 = GLOBAL_PLANER.pathplan((tx,ty), (tx2,ty2))
            path2_len = len(path2)
            if path2_len==0 or path_len+path2_len>tar2_score:
                continue
            heading_list2 = heading_list+[(tx2,ty2)]
            tar_value2 = tar_score - path_len + tar2_score - path2_len
            if tar_value2>max_value:
                max_value = tar_value2
                best_dir = path2dir((cx,cy),path[0])

    return best_dir if best_dir>=0 else np.random.choice([0,1,2,3])

# two step greedy with step_value
def strategy_two_greedy_avg_score(env):
    # information you can get. NOTE:do not change these values
    cx = env.posx
    cy = env.posy
    cur_map = env.map
    cur_step = env.step
    cur_score = env.score

    # run your strategy
    targets = [(tx,ty) for tx,ty in np.reshape(np.where(cur_map>0),(2,-1)).T]
    targets = [(tx,ty) for tx,ty in targets if abs(tx-cx)+abs(ty-cy)<cur_map[tx,ty]]

    max_value = -1
    best_dir = -1
    for tx,ty in targets:
        tar_score = cur_map[tx,ty]
        # path = pathplan_astar(cur_map, (cx,cy), (tx,ty), tar_score)
        path = GLOBAL_PLANER.pathplan((cx,cy), (tx,ty))
        path_len = len(path)
        if path_len==0 or path_len>tar_score:
            continue
        heading_list = [(tx,ty)]
        tar_value = float(tar_score-path_len)/path_len
        # tar_value /= path_len
        # print('tar (%d,%d) path:%d tar_value:%d'%(tx,ty,path_len,tar_value))
        if tar_value>max_value:
            max_value = tar_value
            best_dir = path2dir((cx,cy),path[0])

        for tx2,ty2 in targets:
            if (tx2,ty2) in heading_list:
                continue
            tar2_score = cur_map[tx2,ty2]
            # path2 = pathplan_astar(cur_map, (tx,ty), (tx2,ty2), tar2_score-path_len)
            path2 = GLOBAL_PLANER.pathplan((tx,ty), (tx2,ty2))
            path2_len = len(path2)
            if path2_len==0 or path_len+path2_len>tar2_score:
                continue
            heading_list2 = heading_list+[(tx2,ty2)]
            tar_value2 = float(tar_score - path_len + tar2_score - path2_len)/(path_len + path2_len)
            if tar_value2>max_value:
                max_value = tar_value2
                best_dir = path2dir((cx,cy),path[0])

    return best_dir if best_dir>=0 else np.random.choice([0,1,2,3])

def strategy_two_greedy_avg_score_enh1(env):
    # information you can get. NOTE:do not change these values
    cx = env.posx
    cy = env.posy
    cur_pos = (cx,cy)
    cur_map = env.map
    cur_step = env.step
    cur_score = env.score

    # run your strategy
    targets = [(tx,ty) for tx,ty in np.reshape(np.where(cur_map>0),(2,-1)).T]
    targets = [(tx,ty) for tx,ty in targets if abs(tx-cx)+abs(ty-cy)<cur_map[tx,ty]]

    targets = sorted(targets, key = lambda t:len(GLOBAL_PLANER.pathplan(cur_pos, t)))

    max_len = MAX_STEP-cur_step
    max_value = -1
    best_dir = -1
    for tx,ty in targets:
        tar_score = cur_map[tx,ty]
        # path = pathplan_astar(cur_map, (cx,cy), (tx,ty), tar_score)
        path = GLOBAL_PLANER.pathplan((cx,cy), (tx,ty))
        path_len = len(path)
        if path_len==0 or path_len>tar_score or path_len>max_len:
            continue
        heading_list = [(tx,ty)]
        tar_value = float(tar_score-path_len)/path_len
        # tar_value /= path_len
        # print('tar (%d,%d) path:%d tar_value:%d'%(tx,ty,path_len,tar_value))
        if tar_value>max_value:
            max_value = tar_value
            best_dir = path2dir((cx,cy),path[0])

        for tx2,ty2 in targets:
            if (tx2,ty2) in heading_list:
                continue
            tar2_score = cur_map[tx2,ty2]
            # path2 = pathplan_astar(cur_map, (tx,ty), (tx2,ty2), tar2_score-path_len)
            path2 = GLOBAL_PLANER.pathplan((tx,ty), (tx2,ty2))
            path2_len = len(path2)
            if path2_len==0 or path_len+path2_len>tar2_score or path_len+path2_len>max_len:
                continue
            heading_list2 = heading_list+[(tx2,ty2)]
            tar_value2 = float(tar_score - path_len + tar2_score - path2_len)/(path_len + path2_len)
            if tar_value2>max_value:
                max_value = tar_value2
                best_dir = path2dir((cx,cy),path[0])

    return best_dir if best_dir>=0 else np.random.choice([0,1,2,3])

# two step greedy with step_value
def strategy_two_greedy_avg_score_enh2(env):
    # information you can get. NOTE:do not change these values
    cx = env.posx
    cy = env.posy
    cur_pos = (cx,cy)
    cur_map = env.map
    cur_step = env.step
    cur_score = env.score

    # run your strategy
    targets = [(tx,ty) for tx,ty in np.reshape(np.where(cur_map>0),(2,-1)).T]
    targets = [(tx,ty) for tx,ty in targets if abs(tx-cx)+abs(ty-cy)<cur_map[tx,ty]]
    targets = sorted(targets, key = lambda t:len(GLOBAL_PLANER.pathplan(cur_pos, t)))

    def foo_len(plen):
        a = 5
        b = 1
        y = 50/(a-b)
        x = y*a
        return x/(y+plen)
    min_delta = 0.001 #0.5

    max_value = -1000
    best_dir = -1
    max_len = MAX_STEP-cur_step
    for tx,ty in targets:
        tar_score = cur_map[tx,ty]
        # path = pathplan_astar(cur_map, (cx,cy), (tx,ty), tar_score)
        path = GLOBAL_PLANER.pathplan((cx,cy), (tx,ty))
        path_len = len(path)
        if path_len==0 or path_len>tar_score or path_len>max_len:
            continue
        heading_list = [(tx,ty)]
        tar_value = float(tar_score-path_len)/path_len+foo_len(path_len)
        # tar_value /= path_len
        # print('tar (%d,%d) path:%d tar_value:%d'%(tx,ty,path_len,tar_value))
        if tar_value>max_value + min_delta:
            max_value = tar_value
            best_dir = path2dir((cx,cy),path[0])

        for tx2,ty2 in targets:
            if (tx2,ty2) in heading_list:
                continue
            tar2_score = cur_map[tx2,ty2]
            # path2 = pathplan_astar(cur_map, (tx,ty), (tx2,ty2), tar2_score-path_len)
            path2 = GLOBAL_PLANER.pathplan((tx,ty), (tx2,ty2))
            path2_len = len(path2)
            if path2_len==0 or path_len+path2_len>tar2_score or path_len+path2_len>max_len:
                continue
            heading_list2 = heading_list+[(tx2,ty2)]
            tar_value2 = float(tar_score - path_len + tar2_score - path2_len)/(path_len + path2_len) + foo_len(path_len + path2_len)
            if tar_value2>max_value + min_delta:
                max_value = tar_value2
                best_dir = path2dir((cx,cy),path[0])

    return best_dir if best_dir>=0 else np.random.choice([0,1,2,3])

def strategy_multi_step(env, discount = 0.9, foo_reward = lambda score,len: score, foo_final_reward = lambda sum_score,sum_cnt: sum_score):
    # information you can get. NOTE:do not change these values
    cx = env.posx
    cy = env.posy
    cur_pos = (cx,cy)
    cur_map = env.map
    cur_step = env.step
    cur_score = env.score

    # run your strategy
    targets = [(tx,ty) for tx,ty in np.reshape(np.where(cur_map>0),(2,-1)).T]
    targets = [(tx,ty) for tx,ty in targets if abs(tx-cx)+abs(ty-cy)<cur_map[tx,ty]]

    N = len(targets)
    dist_table = np.zeros((len(targets),len(targets)),'float')
    for i in range(N):
        for j in range(i+1, N):
            len_path = len(GLOBAL_PLANER.pathplan(targets[i], targets[j]))
            dist_table[i,j] = 10000 if len_path<1 else len_path
    dist_table = dist_table+dist_table.T
    value_i = np.zeros((N),'float')
    for i in range(N):
        len_i = len(GLOBAL_PLANER.pathplan(cur_pos,targets[i]))
        ti = targets[i]
        if cur_map[ti]<len_i:
            continue
        value_j = np.zeros((N),'float')
        for j in range(N):
            if j == i:
                continue
            len_j = len_i + dist_table[i,j]
            tj = targets[j]
            if len_j > len(GLOBAL_PLANER.pathplan(cur_pos,tj))+dist_table[i,j]:
                continue
            if cur_map[tj]<len_j:
                continue
            value_k = np.zeros((N),'float')
            for k in range(N):
                if k==i or k==j:
                    continue
                len_k = len_j+dist_table[j,k]
                tk = targets[k]
                if cur_map[tk]<len_k or len_k>len_i+dist_table[i,k]+dist_table[k,j]:
                    continue
                ## cal expect future reward for k
                consider_len = 5
                if False:
                    sum_score = 0
                    sum_cnt = 0
                    for h in range(N):
                        if h==i or h==j or h==k or dist_table[h,k]>consider_len:
                            continue
                        rs = cur_map[targets[h]]-len_k
                        if rs > 0:
                            sum_cnt+=1
                            sum_score+=rs
                    exp_future_reward_k = foo_final_reward(sum_score,sum_cnt)
                else:
                    tmax_score = 0
                    for h in range(N):
                        if h==i or h==j or h==k or dist_table[h,k]>consider_len:
                            continue
                        rs = cur_map[targets[h]]-len_k
                        if rs > 0:
                            temp = float(rs)/dist_table[h,k]
                            if temp>tmax_score:
                                tmax_score = temp
                    exp_future_reward_k = tmax_score
                reward_k = foo_reward(cur_map[tk]-len_k, dist_table[j,k])
                value_k[k] = reward_k + discount*exp_future_reward_k
            value_j[j] = foo_reward(cur_map[tj]-len_j, dist_table[i,j]) + np.max(value_k)*discount
        value_i[i] = foo_reward(cur_map[ti]-len_i, len_i) + np.max(value_j)*discount
    best_path = GLOBAL_PLANER.pathplan(cur_pos,targets[np.argmax(value_i)])
    best_dir = path2dir(cur_pos, best_path[0])
    return best_dir if best_dir>=0 else np.random.choice([0,1,2,3])

class MapPlayer:
    def __init__(self, env, strategy_foo, max_step=MAX_STEP):
        self.env = env
        self.strategy_foo = strategy_foo
        self.max_step = max_step

    def play_episode(self, show_step = False, wait_ms = 100):
        self.env.restart()
        for i in range(self.max_step):
            time_start = time.time()
            movedir = self.strategy_foo(self.env)
            time_cost = time.time() - time_start
            if time_cost > 1:
                print(time_cost,i,self.env.step)
            self.env.move(movedir)
            if show_step:
                self.env.watch()
                key = cv2.waitKey(wait_ms)
                if key==27:
                    break
        return self.env.score

    def play_rounds(self, round_cnt, verbose = False):
        scores = []
        for i in range(round_cnt):
            self.env.restart()
            scores.append(self.play_episode())
            if verbose:
                print('%d/%d'%(i+1,round_cnt),end='\r')
        print('%d/%d'%(i+1,round_cnt))

        scores = np.array(scores)
        mean_score = np.mean(scores)
        if verbose:
            max_score = np.max(scores)
            min_score = np.min(scores)
            var_score = np.var(scores)
            print('Play %d rounds, Average score:%.3f (min:%.3f max:%.3f), var:%.3f'%(round_cnt, mean_score,
                float(min_score),float(max_score),var_score))

        return mean_score

def evaluate_mult_strategys2(strategy_table, rounds = 100):
    strategy_list = sorted(strategy_table.keys())
    scores_table = {}
    times_table = {}
    for s in strategy_list:
        scores_table[s] = []

    env = Environment()
    for s in strategy_list:
        play = MapPlayer(env, strategy_table[s])
        print('test %s...'%s)
        tlist = []
        scores = []
        for i in range(rounds):
            time_start = time.time()
            score = play.play_episode()
            tlist.append(time.time()-time_start)
            scores.append(score)
            print("test %d/%d"%(i+1,rounds), end='\r')
        times_table[s] = tlist
        scores_table[s] = scores
        print('%s, avg:%.1f(%.1f,%.1f),var:%.1f  cost:%.2f(%.2f,%.2f)s'%
            (s,np.mean(scores),np.min(scores),np.max(scores),np.sqrt(np.var(scores)),np.mean(tlist),np.min(tlist),np.max(tlist)))

    evaluates = []
    for s in strategy_list:
        scores = scores_table[s]
        print('%20s: avg:%.1f min:%.1f max:%.1f var:%.1f'%(s,np.mean(scores),np.min(scores),np.max(scores),np.sqrt(np.var(scores))))
        evaluates.append([np.mean(scores),np.min(scores),np.max(scores),np.sqrt(np.var(scores)),np.mean(times_table[s])*1000])
        plt.plot(scores, label = s)
    plt.xlabel("round")
    plt.ylabel("score")
    a = plt.subplot(1,1,1)
    handles, labels = a.get_legend_handles_labels()
    a.legend(handles[::-1], labels[::-1])

    df = pd.DataFrame(np.array(evaluates).T,index=['mean','min','max','variance','time(ms)'], columns = strategy_list)
    df.plot(kind='bar')

    plt.show()

if __name__ == '__main__':
    ########### select strategy  ################
    strategy_table = {
        # '1-random_walk':strategy_random_walk,
        # '0-one_step_greedy':strategy_greedy, #2679~185
        # '1-strategy_two_greedy_whole_score':strategy_two_greedy_whole_score, #2956~210
        # '2-strategy_two_greedy_avg_score':strategy_two_greedy_avg_score, #4194~181
        # '3-strategy_two_greedy_avg_score_enh1':strategy_two_greedy_avg_score_enh1, #4196~203
        # '4-strategy_two_greedy_avg_score_enh2':strategy_two_greedy_avg_score_enh2, #4196~203
        '5-strategy_multi_step': functools.partial(strategy_multi_step, foo_reward = lambda score,plen: float(score)/plen),
        # append your strategy here
    }

    # strategy = strategy_greedy2_2

    # ########### watch single round ##############
    print('start compitition')
    # play = MapPlayer(CompetitionEnvironment(), strategy)
    # # play = MapPlayer(Environment(), strategy)
    # play.play_episode(True, 100)
    # cv2.waitKey()
    # play.play_rounds(100, True)
    # exit()

    ########### test multi strategies ################
    evaluate_mult_strategys2(strategy_table, 20)