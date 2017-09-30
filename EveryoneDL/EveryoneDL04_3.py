#데이터를 직접 넣는 대신 파일에서 읽어오기

import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np

xy = np.loadtxt('EveryoneDL04_3_data.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data)) #데이터 형 맞는지 확인
print(y_data.shape, y_data)

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

print("My score will be ", sess.run(hypothesis, feed_dict={X:[[100, 70, 101]]}))  #학습 후 실제 내 스코어 예상 출력      