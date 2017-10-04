# 첫 Q-러닝 해보기 - 게임 실행 후 결과만 출력 + 랜덤으로 여기저기 가보기

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)


env = gym.make("FrozenLake-v3")

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    dis = .99
    e = 1. / ((i // 100)+1)  # 플레이 할 수록 랜덤이 줄어듬

    while not done:
        if np.random.rand(1) < e:
            action = env.action_space.sample()  #다른데도 좀 가보자
        else:
            action = rargmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))       #제일 좋은데로 가자. 랜덤 조금만 섞고
        new_state, reward, done, info = env.step(action)
        Q[state, action] = reward + dis * np.max(Q[new_state, :])   #옆에서 넘어온 값에 디스카운트 좀 주자 - 최적 경로 찾기 위해
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
