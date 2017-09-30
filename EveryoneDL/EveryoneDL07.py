#training dataset과 test dataset 나누기 6에서 가져옴

import tensorflow as tf

x_data = [[1,2,1],[1,3,2],[1,3,4], [1,5,5], [1,7,5], [1,2,5], [1,6,6], [1,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]#fail 0 pass 1

X = tf.placeholder("float", [None,3])
Y = tf.placeholder("float", [None,3])

nb_classes = 3 #답의 가짓수

W = tf.Variable(tf.random_normal([3, nb_classes], name='weight'))
b = tf.Variable(tf.random_normal([nb_classes], name='bias'))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(201):
    sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
    cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X:x_data, Y:y_data})
    print(step, cost_val, W_val)


#test----------------------------------------------------------------------------------------
x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]
print("prediction: ", sess.run(prediction, feed_dict={X:x_test}))
print("accuracy: ", sess.run(accuracy, feed_dict={X:x_test, Y:y_test}))
