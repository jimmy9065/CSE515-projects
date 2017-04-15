import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pipeline import DC_dataset
import sys
import numpy as np

train_path = '/home/jimmy/WinDisk/data_small/pics/'
train_path = '/home/jimmy/WinDisk/data/pics/'
n_epochs = 60
IMAGE_SIZE = 128

TEST_SIZE = 5000
BATCH_SIZE = 500

generate = True

def weight_variale(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variale(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2_back(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

filter_size = 3

W_conv1 = weight_variale([filter_size, filter_size, 3, 32])
b_conv1 = bias_variale([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
# h_conv1 = tf.nn.relu(conv2d(x, W_conv1))
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variale([filter_size, filter_size, 32, 64])
b_conv2 = bias_variale([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2))
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variale([filter_size, filter_size, 64, 128])
b_conv3 = bias_variale([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3))
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variale([filter_size, filter_size, 128, 64])
b_conv4 = bias_variale([64])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
# h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4))
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variale([filter_size, filter_size, 64, 32])
b_conv5 = bias_variale([32])

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

W_conv6 = weight_variale([filter_size, filter_size, 32, 32])
b_conv6 = bias_variale([32])

h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)

last_n_channel = 32
last_layer = h_pool6
flat_size = int((IMAGE_SIZE/ 64)**2) * last_n_channel

W_fc1 = weight_variale([flat_size, 1024])
b_fc1 = bias_variale([1024])

h_flat = tf.reshape(last_layer, [-1, flat_size])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variale([1024, 2])
b_fc2 = bias_variale([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

y_shape = tf.shape(y_)
y_conv = tf.shape(y_conv)

saver = tf.train.Saver()

with tf.Session() as sess:
    with DC_dataset(train_path, sess, test_size=TEST_SIZE, batch_size=BATCH_SIZE) as data:
        sess.run(tf.global_variables_initializer())
        
        test_image_list = []
        test_label_list = []
        test_label_tmp_list = []
        for it in range(int(TEST_SIZE / BATCH_SIZE)):
            test_images, test_labels_tmp = data.get_test_set()
            test_labels = [[1, 0] if test_label==1 else [0, 1] for test_label in test_labels_tmp]
            test_image_list.append(test_images)
            test_label_list.append(test_labels)
            test_label_tmp_list.append(test_labels_tmp)

        best_acc = 0

        if generate:
            for it in range(n_epochs):
                for ib in range(int((25000-TEST_SIZE) / BATCH_SIZE)):
                    train_images, train_labels_tmp = data.get_train_batch()
                    train_labels = [[1, 0] if train_label==1 else [0, 1] for train_label in train_labels_tmp]
                    [_, loss] = sess.run([train_step, cross_entropy], feed_dict={
                                     x: train_images, y_: train_labels, keep_prob: 0.8})

                acc = 0
                for ib in range(int(TEST_SIZE / BATCH_SIZE)):
                    acc += sess.run(accuracy, feed_dict={
                    x: test_image_list[ib], y_: test_label_list[ib],
                    keep_prob: 1.0})

                acc /= int(TEST_SIZE / BATCH_SIZE)

                if acc>best_acc:
                    best_acc = acc
                    saver.save(sess, './weights/CNN_model.ckpt')

                print("iter%d, loss %g, test accuracy %g" % (it, loss, acc))

        saver.restore(sess, save_path='./weights/CNN_model.ckpt')
        acc = 0
        for ib in range(int(TEST_SIZE / BATCH_SIZE)):
            acc += sess.run(accuracy, feed_dict={
            x: test_image_list[ib], y_: test_label_list[ib],
            keep_prob: 1.0})

        acc /= int(TEST_SIZE / BATCH_SIZE)
        print('best-acc=', acc)

        for ib in range(int((25000-TEST_SIZE) / BATCH_SIZE)):
            train_images, train_labels_tmp = data.get_train_batch()
            tmp_train_image = sess.run(h_flat, feed_dict={x: train_images, keep_prob: 1.0})
            
            if ib==0:
                output_layer = tmp_train_image
                output_label = train_labels_tmp
            else:
                output_layer = np.append(output_layer, tmp_train_image, axis=0)
                output_label = np.append(output_label, train_labels_tmp, axis=0)

        np.savez('./train_CNN_128_cov',X=output_layer,label=output_label)
        
        for ib in range(int(TEST_SIZE / BATCH_SIZE)):
            tmp_test_image = sess.run(h_flat, feed_dict={x: test_image_list[ib], keep_prob: 1.0}) 
            if ib==0:
                output_layer = tmp_test_image
                output_label = test_label_tmp_list[0]
            else:
                output_layer = np.append(output_layer, tmp_test_image, axis=0)
                output_label = np.append(output_label, test_label_tmp_list[ib], axis=0)

        np.savez('./test_CNN_128_cov',X=output_layer,label=output_label)
