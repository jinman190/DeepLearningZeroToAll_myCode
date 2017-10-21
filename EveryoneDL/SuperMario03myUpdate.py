# original DQN 2015 source from  https://github.com/nalsil/kimhun_rl_windows/blob/master/07_3_dqn_2015_cartpole.py
# The code is updated to play super mario by Jinman Chang
# super mario game can be downloaded at https://github.com/ppaquette/gym-super-mario

# ##### is marked where is updated
# explanation for this code is at http://jinman190.blogspot.ca/2017/10/rl.html


###############################################################################super mario initialized
import gym
from gym.envs.registration import register
from gym.scoreboard.registration import add_group
from gym.scoreboard.registration import add_task
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
#################################################################################




import numpy as np
import tensorflow as tf
import random
from collections import deque

from gym import wrappers

env = gym.make('SuperMarioBros-1-1-v0')                                             #####update game title

# Constants defining our neural network
input_size = env.observation_space.shape[0]*env.observation_space.shape[1]*3        #####change input_size - 224*256*3 acquired from ppaquette_gym_super_mario/nes_env.py
output_size = 6                                                                     #####meaning of output can be found at ppaquette_gym_super_mario/wrappers/action_space.py

dis = 0.9
REPLAY_MEMORY = 50000

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=10, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            # First layer of weights
#            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
#                                 initializer=tf.contrib.layers.xavier_initializer())
#            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            # Second layer of Weights
#            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
#                                 initializer=tf.contrib.layers.xavier_initializer())

            # Q prediction
#            self._Qpred = tf.matmul(layer1, W2)




            filters3 = 16

#            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            X_img = tf.reshape(self._X, [-1, 224, 256, 3])
            print("shape X_img ")
            print(X_img.get_shape())
            L1 = tf.layers.conv2d(inputs=X_img, filters=8, kernel_size=[3,3], padding="SAME", activation=tf.nn.tanh)
            print("shape L1 ")
            print(L1.get_shape())
            L1 = tf.layers.max_pooling2d(inputs=L1, pool_size=[2,2], padding="SAME", strides=4)
            print("shape L1 ")
            print(L1.get_shape())
            L2 = tf.layers.conv2d(inputs=L1, filters=32, kernel_size=[3,3], padding="SAME", activation=tf.nn.tanh)
            print("shape L2 ")
            print(L2.get_shape())
            L2 = tf.layers.max_pooling2d(inputs=L2, pool_size=[2,2], padding="SAME", strides=4)
            print("shape L2 ")
            print(L2.get_shape())
            L2 = tf.reshape(L1, [-1, 14 * 16 * 32])
#            L3 = tf.layers.conv2d(inputs=L2, filters=filters3, kernel_size=[3,3], padding="SAME", activation=tf.nn.tanh)
#            print("shape L3 ")
#            print(L3.get_shape())
#            L3 = tf.layers.max_pooling2d(inputs=L3, pool_size=[2,2], padding="SAME", strides=2)
#            print("shape L3 ")
#            print(L3.get_shape())
#            L3 = tf.reshape(L3, [-1, 28 * 32 * filters3])

            L4 = tf.layers.dense(inputs=L2, units=128, activation=tf.nn.relu)
            print("shape L4 ")
            print(L4.get_shape())
            L4 = tf.nn.dropout(L4, 0.7) #dropout 확률
            print("shape L4 ")
            print(L4.get_shape())
#            L4 = tf.layers.dropout(inputs=L4, rate=self.keep_prob, training=self.training) #결과가 다름????

            W5 = tf.get_variable("W5", shape=[128, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([self.output_size]))

        self._Qpred = tf.sigmoid(tf.matmul(L4, W5) + b) #hypothesis
        print("shape hypothesis ")
        print(self._Qpred.get_shape())

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        # Loss function
#        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
#        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._Qpred, labels=self._Y))
        self._loss = -tf.reduce_mean(self._Y*tf.log(self._Qpred) + (1-self._Y)*(tf.log(1-self._Qpred)))#for sigmoid

        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # get target from target DQN (Q')
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack( [x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        if state is None:                                                               #####why does this happen?
            print("None State, ", action, " , ", reward, " , ", next_state, " , ", done)
        else:
            Q = mainDQN.predict(state)

            # terminal?
            if done:
                Q[0, action] = reward
            else:
                # Double DQN: y = r + gamma * targetDQN(s')[a] where
                # a = argmax(mainDQN(s'))
                # Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]
                Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))     #####use normal one for now

            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size)])       #####change shape to fit to super mario

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN, env=env):
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def main():
    max_episodes = 2000
    # store the previous observations in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter('./SuperMario03myUpdate')                        ##3##draw graph to tensorboard (from DeepLearningZeroToAll lab 09-4)
        writer.add_graph(sess.graph)                                                    ##3##draw graph to tensorboard

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / ((episode / float(max_episodes)) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if step_count % 5 == 0:                                          #입력 값을 5프레임에 한번만 바꾸자
                    if np.random.rand(1) < e or state is None or state.size == 1:       #####why does this happen?
                        actionList = env.action_space.sample()
                    else:
                        # Choose an action by greedily from the Q-network
                        #action = np.argmax(mainDQN.predict(state))
                        action = mainDQN.predict(state).flatten()                       #####flatten it and change it as a list
                        actionList = []
                        for i in range(output_size):                                    #####the action list has to have only integer 1 or 0
                            if action[i] > 0.3 :
                                actionList.append(1)                                    #####integer 1 only, no 1.0
                            else:
                                actionList.append(0)                                    #####integer 0 only, no 0.0

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(actionList)

                if done: # Penalty                                                     ##3## remove temporary penalty
                    reward = -100

                # Save the experience to our buffer
                replay_buffer.append((state, actionList, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                      replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:   # Good enough. Let's move on
                    break

            print("Episode: {} steps: {}".format(episode, step_count))
            if step_count > 10000:
                pass
                # break

            if episode % 10 == 9: # train every 10 episode
                # Get a random batch of experiences
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)

                print("Loss: ", loss)
                # copy q_net -> target_net
                sess.run(copy_ops)

        # See our trained bot in action
        env2 = wrappers.Monitor(env, 'gym-results', force=True)

        for i in range(200):
            bot_play(mainDQN, env=env2)

        env2.close()
        # gym.upload("gym-results", api_key="sk_VT2wPcSSOylnlPORltmQ")

if __name__ == "__main__":
    main()




