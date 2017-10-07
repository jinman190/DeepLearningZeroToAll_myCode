# DQN 2015 - 막대기 균형 잡기 (학습 네트웍과 메인 네트웍 두개로 나눠서 학습)
# 이 게임 받은 곳 https://github.com/ppaquette/gym-super-mario

import gym
import gym_pull
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.envs.registration import register
from gym.scoreboard.registration import add_group
from gym.scoreboard.registration import add_task



class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self._build_network()

    def _build_network(self, h_size=100, l_rate=1e-1):
        with tf.variable_scope(self.net_name):  #net_name(main, target)에 따라 다른 변수를 쓰는 부분
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x") 

            W1 = tf.get_variable("W1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            W2 = tf.get_variable("W2", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.matmul(layer1, W2)

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)      #현재까지 알고 있는 정답을 Y로 받아
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))                   #loss를 계산
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})



def simple_replay_train(DQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        if state is None:
            print("None State, ", action, " , ", reward, " , ", next_state, " , ", done)
        else:
            stateR = state.reshape(-1, DQN.input_size) ##################입력 형태를 슈퍼마리오에 맞춰줌
            Q = DQN.predict(stateR)

            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state)) #예상 Y 값은 target에서 가져오고

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, stateR])

    return DQN.update(x_stack, y_stack) #업데이트는 메인을 업데이트


def bot_play(DQN):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(DQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder









dis = .5
learning_rate = 0.1
REPLAY_MEMORY = 50000




#gym_pull.pull('github.com/ppaquette/gym-super-mario') #슈퍼마리오 다운로드 설치 - 처음 한번은 실행해야함

#내가한 방법
#github 소스를 직접 받아서
#cp -r 받은 폴더 ~/gym/gym/envs (복사)

#아래 내용은 https://github.com/openai/gym/wiki/Environments 참고 (새 게임 gym에 추가하는 법)

register(
     id='SuperMarioBros-1-1-v0',
     entry_point='gym.envs.ppaquette_gym_super_mario:MetaSuperMarioBrosEnv',
)

add_group(
     id='ppaquette_gym_super_mario',
     name='ppaquette_gym_super_mario',
     description='super_mario'
)

add_task(
    id='SuperMarioBros-1-1-v0',
    group='ppaquette_gym_super_mario',
    summary="SuperMarioBros-1-1-v0"
)

env = gym.make('SuperMarioBros-1-1-v0')

input_size = env.observation_space.shape[0]*env.observation_space.shape[1]*3 ##################입력 개수를 슈퍼마리오에 맞춰줌
output_size = 6                                                              ##################출력 개수를 슈퍼마리오에 맞춰줌



def main():
    max_episodes = 5000

    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")   #DQN 하나 더

        tf.global_variables_initializer().run()

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")  #학습한 값 복사 하는 기능
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / ((episode // 100)+1)  # 플레이 할 수록 랜덤이 줄어듬
            done = False
            step_count = 0 #플레이한 프레임 수

            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()  #다른데도 좀 가보자
                else:
                    action = np.argmax(mainDQN.predict(state))       #제일 좋은데로 가자. 랜덤 조금만 섞고        

                next_state, reward, done, info = env.step(action)
#                print(next_state.shape);
                if done:
                    reward = -100        #실패하면 최악의 점수!

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
#                if step_count > 9999:  #영원히 플레이 할거 아니잖여?
#                    break

            print("Episode: {} steps: {}".format(episode, step_count))
#            if step_count > 9999:
                #pass
#                break                    #한번 10000프레임까지 했으면 됐으면 관두자          

            if True: #episode % 10 == 1:   #10번 죽을 때마다 한번씩 학습 하자
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)

                sess.run(copy_ops)  #이부분이 추가 됨 - 그동안 학습한거 복사

        bot_play(mainDQN)

if __name__ == "__main__":
    main()







