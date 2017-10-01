# RNN + Time Series Data - 다음날 주가 예측

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)







def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)







input_dimension = 5 #하루 데이터 갯수
output_dimension = num_classes = 1     #필요한 출력이 한개(종가)
hidden_dimension = 5 #2층짜리 RNN을 쓰려고 출력을 5로 맞춤
batch_size = 1
timesteps = sequence_length = 7 #7일치를 학습


X = tf.placeholder(tf.float32, [None, sequence_length, input_dimension])
Y = tf.placeholder(tf.float32, [None, output_dimension])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dimension, state_is_tuple=True, activation=tf.tanh)
cell = tf.contrib.rnn.MultiRNNCell([cell]*2, state_is_tuple=True)               #2층짜리 RNN
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_prediction = tf.contrib.layers.fully_connected(outputs[:, -1], output_dimension, activation_fn=None)





loss = tf.reduce_sum(tf.square(Y_prediction - Y))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)












xy = np.loadtxt('EveryoneDL12_6.csv', delimiter=',')
xy = xy[::-1]               #reverse order
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]        #마지막 값(-1)들을 [] 사이에 넣어서 리턴

x_data = []
y_data = []
for i in range(0, len(y) - sequence_length):
    _x = x[i:i+sequence_length]
    _y = y[i+sequence_length]   #Next close price
    x_data.append(_x)
    y_data.append(_y)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})

    testPredict = sess.run(Y_prediction, feed_dict={X: x_data})
    plt.plot(y_data)        #실제 주가 출력
    plt.plot(testPredict)   #예상된 다음날 주가 출력
    plt.show()
