import tensorflow as tf
from pipeline import DC_dataset
import matplotlib.pyplot as plt
# import sys


train_path = '/home/jimmy/WinDisk/data/pics/'
# train_path = '/home/jimmy/Dropbox/CSE515-Team/projects/data/pics/'
IMG_SIZE = 128
layer_size = 4
filter_sizes = [5, 5, 5, 5]
channel_sizes = [3, 16, 32, 16, 8]  # the first one is for RGB
strides_sizes = [2, 2, 2, 2]
BATCH_SIZE = 100
TEST_SIZE = 100
iters = 2000
output_shapes = [IMG_SIZE]  # the first one is for img_size
study_rate = 0.01


def Conv_encoders(input_layer):
    for i in range(layer_size):
        conv_shape = [filter_sizes[i], filter_sizes[i],
                      channel_sizes[i], channel_sizes[i+1]]
        conv_filter = tf.Variable(tf.truncated_normal(conv_shape, stddev=0.1))
        strides = [1, strides_sizes[i], strides_sizes[i], 1]
        output_shapes.append(int(output_shapes[i]/strides_sizes[i]))
        output = tf.nn.relu(tf.nn.conv2d(input_layer,
                                         conv_filter,
                                         strides=strides,
                                         padding='SAME'))
        input_layer = output
    return output


def Conv_decoders(code_layer):
    for i in range(layer_size-1, -1, -1):
        conv_shape = [filter_sizes[i], filter_sizes[i],
                      channel_sizes[i], channel_sizes[i+1]]
        conv_filter = tf.Variable(tf.truncated_normal(conv_shape, stddev=0.1))
        strides = [1, strides_sizes[i], strides_sizes[i], 1]
        decode_shape = [BATCH_SIZE,
                        output_shapes[i], output_shapes[i], channel_sizes[i]]
        output = tf.nn.sigmoid(tf.nn.conv2d_transpose(code_layer, conv_filter,
                                                      decode_shape,
                                                      strides=strides,
                                                      padding='SAME'))
        code_layer = output
    return output


def Conv_autoencdoer(display=False, generate=False, save=False,
                     train_size=100, test_size=10):
    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])

    code_layer = Conv_encoders(x)
    xp = Conv_decoders(code_layer)

    code_size = output_shapes[layer_size]
    code_flat = tf.reshape(
            code_layer,
            [-1, code_size**2 * channel_sizes[layer_size-1]])

    cost = tf.reduce_mean(tf.pow(xp-x, 2))
    optimizer = tf.train.AdamOptimizer(study_rate).minimize(cost)

    with tf.Session() as sess:
        print('Train the autoencoder')
        with DC_dataset(train_path, sess, batch_size=BATCH_SIZE) as data:
            init = tf.global_variables_initializer()
            sess.run(init)

            for step in range(iters+1):
                train_images, train_labels = data.get_train_batch()
                c, _ = sess.run(
                        [cost, optimizer],
                        feed_dict={x: train_images})
                if step % (iters / 10) == 0:
                    print('step%d, loss %g' % (step, c))

            if display:
                new_images = sess.run(xp,
                                      feed_dict={x: train_images})
                fig = plt.figure()
                for i in range(3):
                    a = fig.add_subplot(3, 2, 2*i+1)
                    plt.imshow(train_images[i, :, :, :])
                    a.set_title('Origin')
                    a = fig.add_subplot(3, 2, 2*i+2)
                    plt.imshow(new_images[i, :, :, :])
                    a.set_title('Decoded')
                plt.show()

        # Generate feature space
        if generate:
            with DC_dataset(
                    train_path, sess,
                    test_size=TEST_SIZE, batch_size=BATCH_SIZE) as data:
                """
                __init__(path, sess, test_size, batch_size, seed, num_tread)
                """
                init = tf.global_variables_initializer()
                sess.run(init)

                train_images, train_labels = data.get_train_batch()
                train_x = sess.run(code_flat, feed_dict={x: train_images})
                print('shape of train_set', train_x.shape)
                print('num of positive case in train_set', sum(train_labels))

                test_images, test_labels = data.get_test_set()
                test_x = sess.run(code_flat, feed_dict={x: test_images})
                print('shape of test_set', test_x.shape)
                print('num of positive case in test_set', sum(test_labels))

            return ((train_x, train_labels), (test_x, test_labels))


if __name__ == '__main__':
    Conv_autoencdoer(display=True)
    # Conv_autoencdoer(generate=True, train_size=3000, test_size=200)
