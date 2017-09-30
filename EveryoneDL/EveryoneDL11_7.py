# ensenble 앙상블 트레이닝

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
            self.training = tf.placeholder(tf.bool)
            learning_rate=0.01


            self.X = tf.placeholder(tf.float32, [None, 28*28])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            L1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
            L1 = tf.layers.max_pooling2d(inputs=L1, pool_size=[2,2], padding="SAME", strides=2)
            L2 = tf.layers.conv2d(inputs=L1, filters=64, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
            L2 = tf.layers.max_pooling2d(inputs=L2, pool_size=[2,2], padding="SAME", strides=2)
            L3 = tf.layers.conv2d(inputs=L2, filters=128, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
            L3 = tf.layers.max_pooling2d(inputs=L3, pool_size=[2,2], padding="SAME", strides=2)
            L3 = tf.reshape(L3, [-1, 4 * 4 * 128])
            L4 = tf.layers.dense(inputs=L3, units=625, activation=tf.nn.relu)

            L4 = tf.nn.dropout(L4, self.keep_prob) #dropout 확률
#            L4 = tf.layers.dropout(inputs=L4, rate=self.keep_prob, training=self.training) #결과가 다름????

            W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]))
            self.hypothesis = tf.matmul(L4, W5) + b
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

            self.prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

    def train(self, x_data, y_data, keep_prob=0.7): #학습은 drop out 70%로
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X:x_data, self.Y:y_data, self.keep_prob:keep_prob, self.training:True})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0): #테스트는 drop out 없이
        return self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test, self.keep_prob:keep_prob, self.training:False})

    def get_predict(self, x_test, y_test, keep_prob=1.0): #테스트는 drop out 없이
        return self.sess.run(self.hypothesis, feed_dict={self.X:x_test, self.Y:y_test, self.keep_prob:keep_prob, self.training:False})



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

models = []

sess = tf.InteractiveSession()
for i in range(3):  #모델수는 3개로 하자
    models.append(Model(sess, "m"+str(i)))

sess.run(tf.global_variables_initializer())

training_epochs = 1 #15 느림
batch_size = 100

for epoch in range(training_epochs):
    avg_costs= np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        for m_idx, m in enumerate(models):                         #model 수 만큼 반복
            c, _ = m.train(batch_xs, batch_ys)
            avg_costs[m_idx] += c / total_batch
            print('Epoch: ', (epoch+1), ' cost: ', avg_costs[m_idx])

predictions = np.zeros([len(mnist.test.labels), 10])
for m_idx, m in enumerate(models):
    print(m_idx, 'accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.get_predict(mnist.test.images, mnist.test.labels)
    predictions += p

result_predict = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
result_accuracy = tf.reduce_mean(tf.cast(result_predict, tf.float32))
print('result accuracy: ', sess.run(result_accuracy))