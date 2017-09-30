# 실제 mnist 숫자 이미지로 cnn 시스템 개발2, 깊이를 늘려서 정확도 99%
# convolutional layer 1 -> layer 2 -> layer 3 -> dropout layer -> fullyconnected layer

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME') #output img shape (3,3) #합성곱 convolution
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #풀링 subsampling



W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME') #output img shape (3,3) #합성곱 convolution
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #풀링 subsampling


#convolution 레이어 하나 더!!!
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME') #output img shape (3,3) #합성곱 convolution
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #풀링 subsampling


L3 = tf.reshape(L3, [-1, 4 * 4 * 128])
learning_rate=0.01


#dropout 레이어 하나 추가
W4 = tf.get_variable("W4", shape=[4*4*128, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob) #dropout 확률


W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

training_epochs = 1 #15 느림
batch_size = 100

for epoch in range(training_epochs):
    avg_cost=0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.7}) #학습은 drop out 70%로
        avg_cost += c / total_batch

    print('Epoch: ', (epoch+1), ' cost: ', avg_cost)


prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acurracy: ', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1})) #테스트는 drop out 없이
