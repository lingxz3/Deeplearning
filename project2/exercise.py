import tensorflow as tf

#from PIL import Image
#import glob
import numpy as np
from scipy.misc import imread, imresize
import pandas as pd
import h5py


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


#file = open("train.txt", "r")



#array_label = []
#test_label = []
#test_data = []
#images = []
size = 20

#for line in file:
   # each_path = line[:-3]
    #array_label = np.append(array_label, line[-2:-1])
    #print (each_path)
    #im = imread("/Users/lingxiaozhang/Documents/6000b/project2/project2/data/flower_photos/daisy/" + "8446495985_f72d851482.jpg")
    #im = imread(each_path)
    #im = imresize(im, (size, size))
    #print im
    #print im.shape
    #print(im.shape)
    #images = np.append(images, im).reshape([-1, 20, 20, 3])



#array_label = [int(i) for i in array_label]
#array_label = np.asarray(array_label).reshape(-1, 1)
#array_label = tf.one_hot(array_label, depth=5)

train = pd.read_table('train.txt', sep=' ', header=None)
train.columns = ['path', 'label']
val = pd.read_table('val.txt', sep=' ', header=None)
val.columns = ['path', 'label']



#for i in range()



# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, (size*size)])/255.   # 28x28 -- 784
ys = tf.placeholder(tf.float32, [None, 5])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, size, size, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

# (2569,)
#print (array_label)

#print array_label
#sess = tf.Session()
#a = sess.run(array_label)
#print(a)

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32


## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64



## fc1 layer ##
W_fc1 = weight_variable([13*13*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 13*13*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


## fc2 layer ##
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
#if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#    init = tf.initialize_all_variables()
#else:
init = tf.global_variables_initializer()
sess.run(init)


#images = np.asarray(images)

#print(images.shape)

#for i in range(1000):
 #  batch_xs, batch_ys = images.train.next_batch(100)
 #   sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
  ##     print(compute_accuracy(
    #        mnist.test.images[:1000], mnist.test.labels[:1000]))


