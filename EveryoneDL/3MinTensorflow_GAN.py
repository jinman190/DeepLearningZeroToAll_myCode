# 3분 딥러닝 텐서플로 GAN

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28 * 28
n_noise = 128

X = tf.placeholder(tf.float32, [None, n_input]) #actual image as input
Z = tf.placeholder(tf.float32, [None, n_noise]) #random noise - to create image

#vars used for image generator (random noise -> fake image)
F_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
F_b1 = tf.Variable(tf.zeros([n_hidden]))
F_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
F_b2 = tf.Variable(tf.zeros([n_input]))

#vars used to verify image (is it actual image?)
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))   #output - 1 for actual image, 0 for generated image

def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, F_W1) + F_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, F_W2) + F_b2)
    return output   #generated image

def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output   #0~1

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))


#model
F_image = generator(Z)            #generate fake image
D_fake = discriminator(F_image)   #check fake image -> output should be 0
D_real = discriminator(X)   #check real image -> output should be 1

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_fake))  #loss for cop
loss_F = tf.reduce_mean(tf.log(D_fake))                     #loss for criminal

D_var_list = [D_W1, D_b1, D_W2, D_b2]
F_var_list = [F_W1, F_b1, F_W2, F_b2]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_F = tf.train.AdamOptimizer(learning_rate).minimize(-loss_F, var_list=F_var_list)






#session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D = 0
loss_val_F = 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X:batch_xs, Z:noise})
        _, loss_val_F = sess.run([train_F, loss_F], feed_dict={Z:noise})

    #create image to print
    if(epoch % 10 == 0):
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(F_image, feed_dict={Z:noise})

        fig, ax = plt.subplots(1, sample_size, figsize = (sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close()
    print('epoch end')
















