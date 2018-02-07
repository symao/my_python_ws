# This use targets position and current position as an input state vector to train a DQN
# and can reach average score of 3700(while two step greedy 4500)
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.layers import Input, Dense, Flatten, Lambda, merge
from keras.layers.convolutional import Conv2D
from keras import backend as K
from map_env import Environment


EPISODES = 50000
MAP_SIZE = 12
INPUT_DEPTH = 6
MAX_STEP = 400
OBS_CNT = 20
PKG_CNT = 24
MODEL_FILE = "./model/pacman_duel_dqn.h5"

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

class DQNAgent:
    def __init__(self, input_size=(MAP_SIZE,MAP_SIZE,INPUT_DEPTH), action_size=4):
        self.render = False
        self.load_model = False
        # environment settings
        self.state_size = input_size
        self.action_size = action_size
        # parameters about epsilon
        self.epsilon = 1
        self.epsilon_start, self.epsilon_end = 1, 0.05
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.8
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # build model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.ddqn = True

        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights(MODEL_FILE)

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        input = Input(shape=self.state_size)
        shared = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(input)
        shared = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(shared)
        flatten = Flatten()(shared)

        # network separate state value and advantages
        advantage_fc = Dense(256, activation='relu')(flatten)
        advantage = Dense(self.action_size)(advantage_fc)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                           output_shape=(self.action_size,))(advantage)

        value_fc = Dense(256, activation='relu')(flatten)
        value =  Dense(1)(value_fc)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)

        # network merged and make Q Value
        q_value = merge([value, advantage], mode='sum')
        model = Model(inputs=input, outputs=q_value)
        model.summary()

        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(list(self.memory), self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0])
            next_history[i] = np.float32(mini_batch[i][3])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        value = self.model.predict(history)
        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                if self.ddqn:
                    # the key point of Double DQN
                    # selection of action is from model
                    # update is from target model
                    target[i] = reward[i] + self.discount_factor * \
                                            target_value[i][np.argmax(value[i])]
                else:
                    target[i] = reward[i] + self.discount_factor * \
                                            np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def save_model(self, name):
        self.model.save_weights(name)

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


if __name__ == "__main__":
    env = Environment()
    agent = DQNAgent()

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        # for _ in range(random.randint(1, agent.no_op_steps)):
        #     observe, _, _, _ = env.step(1)

        history = np.array([env2input(env)])

        while not done:
            global_step += 1
            step += 1

            # get action for the current history and go one step in environment
            action = agent.get_action(history)
            _, reward = env.move(action)
            next_history = np.array([env2input(env)])

            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history))[0])

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(history, action, reward, next_history, False)
            # every some time interval, train model
            agent.train_replay()
            # update the target model with model
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            # if agent is dead, then reset the history
            if dead:
                dead = False
            else:
                history = next_history

            done = step>=MAX_STEP

            # if done, plot the score over episodes
            if done:
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                scores.append(score)
                if e%100==0:
                    print('#%d score:%.1f queue:%d eps:%.2f glbstep:%d avg_q:%.2f avg_loss:%.2f'
                            %(e, np.mean(scores[-100:]), len(agent.memory), agent.epsilon, global_step,
                            agent.avg_q_max / float(step), agent.avg_loss / float(step)))

                agent.avg_q_max, agent.avg_loss = 0, 0

        if e>0 and e % 1000 == 0:
            agent.model.save_weights(MODEL_FILE)
