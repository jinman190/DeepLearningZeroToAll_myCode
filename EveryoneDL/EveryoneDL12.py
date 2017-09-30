# RNN의 기초

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)

h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

hidden_size = 2
batch_size = 3
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[h,e,l,l,o],[e,l,l,o,h],[l,l,o,e,h]], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())