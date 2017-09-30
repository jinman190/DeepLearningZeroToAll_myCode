# 파이썬 클래스 사용 

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32)

            self.X = tf.placeholder(tf.float32, [None, 28*28])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

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
            L4 = tf.nn.dropout(L4, self.keep_prob) #dropout 확률


            W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]))
            self.hypothesis = tf.matmul(L4, W5) + b
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

            self.prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

    def train(self, x_data, y_data, keep_prob=0.7): #학습은 drop out 70%로
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X:x_data, self.Y:y_data, self.keep_prob:keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0): #테스트는 drop out 없이
        return self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test, self.keep_prob:keep_prob})

    def get_predict(self, x_test, y_test, keep_prob=1.0): #테스트는 drop out 없이
        return self.sess.run(self.hypothesis, feed_dict={self.X:x_test, self.Y:y_test, self.keep_prob:keep_prob})



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())

training_epochs = 1 #15 느림
batch_size = 100

for epoch in range(training_epochs):
    avg_cost=0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch: ', (epoch+1), ' cost: ', avg_cost)

print('acurracy: ', m1.get_accuracy(mnist.test.images, mnist.test.labels))
