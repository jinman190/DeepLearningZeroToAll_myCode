#데이터를 행렬로 넣기

import tensorflow as tf
#import matplotlib.pyplot as plt

x_data = [[73.0, 80.0, 75.0], [93.0, 88.0, 93.0], [89.0, 91.0, 90.0], [96.0, 98.0, 100.0], [73.0, 66.0, 70.0]]
y_data = [[152.0], [185.], [180.], [196.], [142.]]

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

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "Prediction:", hy_val)