# RNN - 아주 긴 문자열 잘라서 학습하기 (제대로 안되는 예)

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)

sample = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}
sample_idx = [char2idx[c] for c in sample]  #샘플 글자열->숫자 배열


sequence_length = 10 #10자씩 잘라서 학습하자

x_data = []
y_data = []
for i in range(0, len(sample)-sequence_length):
    x_str = sample[i:i+sequence_length]
    y_str = sample[i+1:i+sequence_length+1]

    x_num = [char2idx[c] for c in x_str]
    y_num = [char2idx[c] for c in y_str]

    x_data.append(x_num)
    y_data.append(y_num)




input_dimension = len(idx2char) #글자 종류 수
hidden_size = len(idx2char) #같은 글자들에서 글자들을 찾아내니까 input_dimension과 같을 수 밖에
batch_size = len(x_data)
num_classes = len(idx2char)


X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
x_one_hot = tf.one_hot(X, num_classes)

weights = tf.ones([batch_size, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, 2)














with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
#        print(i, " loss: ", l, " prediction: ", result, " true Y: ", y_data)
        print(i, " loss: ", l, " prediction: ", result[0], " true Y: ", y_data[0]) #첫번째 결과만 출력해보자

        result_str = [idx2char[c] for c in np.squeeze(result[0])] #첫번째 결과만 출력해보자
        print(" prediction str: ", ''.join(result_str))
