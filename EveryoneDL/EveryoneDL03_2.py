#GradientDescentOptimizer가 하는 일을 코드로 풀어 쓰기 - 직접 미분

import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train = optimizer.minimize(cost) 아래 내용이 이 것의 구현

descent = W - 0.1 * (tf.reduce_mean((W*X-Y)*X)) #미분값
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_train, Y: y_train})
    print(step, sess.run(cost, feed_dict={X: x_train, Y:y_train}), sess.run(W))