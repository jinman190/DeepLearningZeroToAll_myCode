# 딥러닝을 이용한 강화학습 게임 - 막대기 균형 잡기 (잘 안되는 예)

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



dis = .99
learning_rate = 0.1
num_episodes = 2000





env = gym.make("CartPole-v0")

input_size = env.observation_space.shape[0] #이게 뭔지 찾아보자
output_size = env.action_space.n

X = tf.placeholder(shape=[None,input_size], dtype=tf.float32)

#W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))
W = tf.get_variable("W1", shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer()) #더 좋은 초기 값 설정

Qpred = tf.matmul(X, W)
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)




rList = []
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(num_episodes):
    state = env.reset()
    done = False
    e = 1. / ((i // 100)+1)  # 플레이 할 수록 랜덤이 줄어듬
    local_loss = []
    step_count = 0 #플레이한 프레임 수

    while not done:
        step_count += 1
        stateR = np.reshape(state, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X:stateR})
        if np.random.rand(1) < e:
            action = env.action_space.sample()  #다른데도 좀 가보자
        else:
            action = np.argmax(Qs)       #제일 좋은데로 가자. 랜덤 조금만 섞고        
            
        new_state, reward, done, info = env.step(action)

        if done:
            Qs[0, action] = -100        #실패하면 최악의 점수!
        else:
            new_stateR = np.reshape(new_state, [1, input_size]) #결과로 나온 스테이트(현재 위치?)를 그냥 다음 입력으로 쓰자
            Qs1 = sess.run(Qpred, feed_dict={X:new_stateR})
            Qs[0, action] = reward + dis*np.max(Qs1)

        sess.run(train, feed_dict={X:stateR, Y:Qs})

        state = new_state

    rList.append(step_count)

    print("Episode: {}  steps: {}".format(i, step_count))
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
       break





#학습 잘 되었는지 플레이 시켜보자
observation = env.reset()
reward_sum = 0
env.render()
done = False
while True:
    state = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict={X:state})
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    env.render()

    if done:
        print("Total score: {}".format(reward_sum))
        break



#print("Success rate: " + str(sum(rList)/num_episodes))
#print("Final Q-Table Values")
#plt.bar(range(len(rList)), rList, color="blue")
#plt.show()

#        env.render()
