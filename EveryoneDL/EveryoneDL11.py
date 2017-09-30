# CNN 처음 + max pool, 이미지 출력

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.InteractiveSession()

image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)

#image = tf.nn.max_pool(image, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME').eval() #풀링은 한줄로 가능

weight = tf.constant([[[[1.,-1.]], [[1.,-1.]]],
                      [[[1.,-1.]], [[1.,-1.]]]]) #weight가 2개씩 이므로 결과 이미지가 두개 나옴

#conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='VALID') #output img shape (2,2)
conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME') #output img shape (3,3)

conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,2,i+1)
    plt.imshow(one_img.reshape(3,3), cmap='gray')

plt.show()

#print(image.shape)
#plt.imshow(image.reshape(3,3), cmap='Greys')
#plt.show()