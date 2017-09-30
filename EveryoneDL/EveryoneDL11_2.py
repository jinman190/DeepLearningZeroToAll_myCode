# 실제 mnist 숫자 이미지 한장으로 cnn 테스트

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img = mnist.train.images[0].reshape(28,28)
#plt.imshow(img, cmap='gray')
#plt.show()

sess = tf.InteractiveSession()

image = img.reshape(-1, 28, 28, 1)

#image = tf.nn.max_pool(image, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME').eval() #풀링 subsampling 은 한줄로 가능

weight = tf.constant([[[[1.,-1.]], [[1.,-1.]]],
                      [[[1.,-1.]], [[1.,-1.]]]]) #weight가 2개씩 이므로 결과 이미지가 두개 나옴

#conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='VALID') #output img shape (2,2)
conv2d = tf.nn.conv2d(image, weight, strides=[1,2,2,1], padding='SAME') #output img shape (3,3) #합성곱 convolution

#conv2d = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #풀링 subsampling

conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,2,i+1)
    plt.imshow(one_img.reshape(14,14), cmap='gray') #stride가 2*2 이므로 이미지가 14*14가 됨

plt.show()

#print(image.shape)
#plt.imshow(image.reshape(3,3), cmap='Greys')
#plt.show()