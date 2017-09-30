#파일이 너무 커서 여러개로 나눠져 있을 때 데이터 한번에 가져다 쓰기

import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np


'''
xy = np.loadtxt('EveryoneDL04_3_data.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
'''
filename_queue = tf.train.string_input_producer(['EveryoneDL04_3_data.csv'], shuffle=False, name='filename_queue');
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)


#기존과 똑같은 부분
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(X, W) + b #x1 * w1 + x2  * w2 + x3 * w3 + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


#항상 쓰는 부분
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


for step in range(2001):
    x_data, y_data = sess.run([train_x_batch, train_y_batch]) #파일 내용 한줄씩 읽어서 변수에 대입
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "Prediction:", hy_val)

print("My score will be ", sess.run(hypothesis, feed_dict={X:[[100, 70, 101]]}))  #학습 후 실제 내 스코어 예상 출력      


#항상 쓰는 부분
coord.request_stop()
coord.join(threads)