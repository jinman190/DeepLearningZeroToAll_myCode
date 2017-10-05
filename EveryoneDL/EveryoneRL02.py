# 강화학습 RL 첫 게임 실행해보기 (Frozen Lake)
'''
우분투에서 먼저 실행할 것
source ~/tensorflow/bin/activate
export DISPLAY=:0
cd /mnt/c/Users/jinma/Desktop/GitHub/DeepLearningZeroToAll_myCode/EveryoneDL
윈도우에서 xming 실행
python3 EveryoneRL02.py
'''
import gym
env = gym.make("Taxi-v2")
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)