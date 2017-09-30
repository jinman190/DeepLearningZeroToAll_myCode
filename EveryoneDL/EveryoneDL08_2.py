#텐서보드로 그래프 그리기
#아래를 실행 후에 로그 파일은 파이썬 창에서 그 디렉토리로 가서 tensorboard --logdir=.
#인터넷 창에서 http://192.168.0.19:6006

import tensorflow as tf
import numpy as np

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope("layer1") as scope:                                      #
    W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    h1 = tf.sigmoid(tf.matmul(X,W1)+b1)

with tf.name_scope("layer2") as scope:                                      #
    W = tf.Variable(tf.random_normal([2,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    hypothesis = tf.sigmoid(tf.matmul(h1,W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

w2_hist = tf.summary.histogram("weight", W)                                 #
cost_summ = tf.summary.scalar("cost", cost)                                 #
summary = tf.summary.merge_all()                                            #

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./EveryoneDL_08_2')                     #
    writer.add_graph(sess.graph)                                            #

    for step in range(10001):
        s, _ = sess.run([summary, train], feed_dict={X:x_data, Y:y_data})   #
        writer.add_summary(s, global_step=step)                             #

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print(h, " ", c, " ", a);