# RNN - hihello 학습

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)





input_dimension = 5 #글자 종류 수
hidden_size = 5 #같은 글자들에서 글자들을 찾아내니까 input_dimension과 같을 수 밖에
batch_size = 1
sequence_length = 6 #글자 갯수

X = tf.placeholder(tf.float32, [None, sequence_length, input_dimension])
Y = tf.placeholder(tf.int32, [None, sequence_length])

weights = tf.ones([batch_size, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, 2)








h = [1,0,0,0,0]
i = [0,1,0,0,0]
e = [0,0,1,0,0]
l = [0,0,0,1,0]
o = [0,0,0,0,1]

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[h,i,h,e,l,l]]
y_data = [[1, 0, 2, 3, 3, 4]]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X:x_one_hot, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_one_hot})
        print(i, " loss: ", l, " prediction: ", result, " true Y: ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(" prediction str: ", ''.join(result_str))
