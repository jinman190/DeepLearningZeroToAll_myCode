# 딥러닝을 이용한 강화학습 시작

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



dis = .99
learning_rate = 0.1
num_episodes = 2000



def one_hot(x):
    return np.identity(16)[x:x+1]




env = gym.make("FrozenLake-v0")

input_size = env.observation_space.n
output_size = env.action_space.n

X = tf.placeholder(shape=[1,input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

Qpred = tf.matmul(X, W)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)




rList = []
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    e = 1. / ((i // 100)+1)  # 플레이 할 수록 랜덤이 줄어듬
    local_loss = []

    while not done:
        Qs = sess.run(Qpred, feed_dict={X:one_hot(state)})
        if np.random.rand(1) < e:
            action = env.action_space.sample()  #다른데도 좀 가보자
        else:
            action = np.argmax(Qs)       #제일 좋은데로 가자. 랜덤 조금만 섞고        
            
        new_state, reward, done, info = env.step(action)

        if done:
            Qs[0, action] = reward
        else:
            Qs1 = sess.run(Qpred, feed_dict={X:one_hot(new_state)})
            Qs[0, action] = reward + dis*np.max(Qs1)

        sess.run(train, feed_dict={X:one_hot(state), Y:Qs})

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

#        env.render()
