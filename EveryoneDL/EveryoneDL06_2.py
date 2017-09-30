#multi classification 파일에서 읽어와서

import tensorflow as tf
import numpy as np

xy = np.loadtxt('EveryoneDL06_2.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7 #답의 가짓수

X = tf.placeholder(tf.float32, [None,16])
Y = tf.placeholder(tf.int32, [None,1]) #shape=(?,1)
Y_one_hot = tf.one_hot(Y, nb_classes) #shape=(?,1,7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) #shape=(?,7)

W = tf.Variable(tf.random_normal([16, nb_classes], name='weight'))
b = tf.Variable(tf.random_normal([nb_classes], name='bias'))

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

#cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#중간 결과 출력을 위해 계산
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            print("Step: {:5}, Loss:{:.3f}, Acc: {:.2%}".format(step, loss, acc))





