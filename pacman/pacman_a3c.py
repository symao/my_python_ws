#!/usr/bin/python3.4
import gym
import time
import random
import threading
import numpy as np
import tensorflow as tf
import os
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K
import copy
from map_env import Environment

# global variables for A3C
global episode
episode = 0
EPISODES = 80000

MAP_SIZE = 12
MAX_STEP = 288
INPUT_DIM = 6
ACTION_SIZE = 4
TRAIN_RES = 'train_scores_a3c.txt'
SAVE_MODEL = "./model/pacman_a3c"

def env2input(env):
    r,c = env.map.shape[:2]
    obs_map = np.zeros((r,c),'float')
    obs_map[env.map<0] = 1
    score_map = env.map.astype('float')
    pos_map = np.zeros((r,c),'float')
    pos_map[int(env.posx),int(env.posy)] = 1
    score_map2 = np.copy(score_map)
    score_map2[score_map2<0] = 0
    return np.stack((obs_map, obs_map, obs_map, score_map, score_map2, pos_map), axis = 2)

# This is A3C(Asynchronous Advantage Actor Critic) agent(global) for the Cartpole
# In this example, we use A3C algorithm
class A3CAgent:
    def __init__(self, action_size=ACTION_SIZE):
        # environment settings
        self.state_size = (MAP_SIZE, MAP_SIZE, INPUT_DIM)
        self.action_size = action_size

        self.discount_factor = 0.85
        self.no_op_steps = 30

        # optimizer parameters
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        self.threads = 8

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)

    def train(self):
        # self.load_model(SAVE_MODEL)
        agents = [Agent(self.action_size, self.state_size, [self.actor, self.critic], self.sess, self.optimizer,
                        self.discount_factor, [self.summary_op, self.summary_placeholders,
                        self.update_ops, self.summary_writer]) for _ in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()

        while True:
            time.sleep(60*20) #20 minutes
            self.save_model(SAVE_MODEL)

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(input)
        conv = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = actor_loss + 0.01*entropy
        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + '_critic.h5')

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

# make agents(local) and start training
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess, optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer = summary_ops

        self.states, self.actions, self.rewards = [],[],[]

        self.local_actor, self.local_critic = self.build_localmodel()

        self.avg_p_max = 0
        self.avg_loss = 0

        # t_max -> max batch size for training
        self.t_max = 20
        self.t = 0

    # Thread interactive with environment
    def run(self):
        # self.load_model(SAVE_MODEL)
        global episode

        env = Environment(MAP_SIZE)

        while episode < EPISODES:
            done = False
            # 1 episode = 5 lives
            env.reset(np.random.randint(10,30), np.random.randint(3,7))
            
            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            # for _ in range(random.randint(1, 10)):
            #     env.move(np.random.randint(4))

            for step in range(MAX_STEP):
                self.t += 1
                observe = env2input(env)
                action, policy = self.get_action(observe)
                _, reward = env.move(action)
                self.avg_p_max += np.amax(self.actor.predict(np.array([observe])))
                # save the sample <s, a, r, s'> to the replay memory
                self.memory(observe, action, reward)

                if self.t >= self.t_max or step == MAX_STEP-1:
                    self.train_model(done)
                    self.update_localmodel()
                    self.t = 0

            episode += 1
            if episode%1000==0:
                print("episode:", episode, "  score:", env.score)
            open(TRAIN_RES,'a').write('%d %d\n'%(episode, env.score))

            stats = [env.score, self.avg_p_max / float(step), step]
            for i in range(len(stats)):
                self.sess.run(self.update_ops[i], feed_dict={
                    self.summary_placeholders[i]: float(stats[i])
                })
            summary_str = self.sess.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, episode + 1)
            self.avg_p_max = 0
            self.avg_loss = 0

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0 if done else self.critic.predict(np.array(self.states[-1:]))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_model(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        states = np.array(self.states)

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def build_localmodel(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(input)
        conv = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.set_weights(self.actor.get_weights())
        critic.set_weights(self.critic.get_weights())

        actor.summary()
        critic.summary()

        return actor, critic

    def update_localmodel(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, state):
        policy = self.local_actor.predict(np.array([state]))[0] # softmax output which sum to 1
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


if __name__ == "__main__":
    if os.path.exists(TRAIN_RES):
        open(TRAIN_RES,'w').write('')

    global_agent = A3CAgent()
    global_agent.train()
