# nondeterministic - 미끄러지는 frozen lake

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


env = gym.make("FrozenLake-v0")

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    dis = .99
    learning_rate = .85

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
        new_state, reward, done, info = env.step(action)

#        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        Q[state, action] = (1-learning_rate)*Q[state, action] + learning_rate*(reward + dis * np.max(Q[new_state, :])) #얻은 Q값을 다 쓰는게 아니라 조금씩만 대입
        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

#        env.render()
