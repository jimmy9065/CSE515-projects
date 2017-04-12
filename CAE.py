import tensorflow as tf
import numpy as np
from pipeline import DC_dataset
import matplotlib.pyplot as plt
from progressbar import ProgressBar
import sys


train_path = '/home/jimmy/WinDisk/data/pics/'
out_put_path = '/home/jimmy/WinDisk/data/'
# train_path = '/home/jimmy/Dropbox/CSE515-Team/projects/data/pics/'
IMG_SIZE = 128
layer_size = 3
filter_sizes = [5, 5, 5, 5]
channel_sizes = [3, 16, 32, 8, 8]  # the first one is for RGB
strides_sizes = [1, 2, 2, 2]
BATCH_SIZE = 100
iters = 30000
output_shapes = [IMG_SIZE]  # the first one is for img_size
study_rate = 0.01
Filters_dict = {}


def Display_imgs(image1, image2):
    fig = plt.figure()
    for i in range(3):
        a = fig.add_subplot(3, 2, 2*i+1)
        plt.imshow(image1[i, :, :, :])
        a.set_title('Origin')
        a = fig.add_subplot(3, 2, 2*i+2)
        plt.imshow(image2[i, :, :, :])
        a.set_title('Decoded')
    plt.show()


def Conv_encoders(input_layer):
    for i in range(layer_size):
        conv_shape = [filter_sizes[i], filter_sizes[i],
                      channel_sizes[i], channel_sizes[i+1]]
        conv_filter = tf.Variable(tf.truncated_normal(conv_shape, stddev=0.1),
                                  name='filter'+str(i))

        Filters_dict['filter'+str(i)] = conv_filter

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
        conv_filter = tf.Variable(
                tf.truncated_normal(conv_shape, stddev=0.1),
                name='defilter'+str(i))
        Filters_dict['defilter'+str(i)] = conv_filter
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
                     train_size=100, test_size=100):
    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])

    code_layer = Conv_encoders(x)
    code_shape = tf.shape(code_layer)
    xp = Conv_decoders(code_layer)

    code_size = output_shapes[layer_size]
    code_flat = tf.reshape(
            code_layer,
            [-1, code_size**2 * channel_sizes[layer_size]])

    cost = tf.reduce_mean(tf.pow(xp-x, 2))
    optimizer = tf.train.AdamOptimizer(study_rate).minimize(cost)

    with tf.Session() as sess:
        print('Train the autoencoder')
        with DC_dataset(train_path, sess,
                        batch_size=BATCH_SIZE,
                        test_size=0) as data:
            init = tf.global_variables_initializer()
            sess.run(init)

            for step in range(iters+1):
                train_images, train_labels = data.get_train_batch()
                c, _ = sess.run(
                        [cost, optimizer],
                        feed_dict={x: train_images})
                if step % (iters / 10) == 0:
                    print('step%d, loss %g' % (step, c))

            sample_image = sess.run(code_flat, feed_dict={x: train_images})
            code_shape_np = sess.run(code_shape, feed_dict={x: train_images})
            print('code_shape: ', code_shape_np)
            print('feature sapce dim: ', sample_image.shape[1:])

            if display:
                new_images = sess.run(xp, feed_dict={x: train_images})
                Display_imgs(train_images, new_images)

            if generate:
                saver = tf.train.Saver(Filters_dict)
                saver.save(sess, './CAE_weight.ckpt')
    # Generate feature space
    if generate:
        div = 20
        with tf.Session() as sess:
            # sess.run(Filters, feed_dict=Filters_dict)
            saver.restore(sess, './CAE_weight.ckpt')
            print('Creating Mapped Feature space')
            with DC_dataset(train_path, sess,
                            test_size=test_size,
                            batch_size=int(train_size/div)) as data:
                """
                __init__(path, sess, test_size, batch_size, seed, num_tread)
                """
                init = tf.global_variables_initializer()
                sess.run(init)

                train_xs = []
                train_labelss = []

                bar = ProgressBar(maxval=div)
                bar.start()
                for i in range(div):
                    bar.update(i+1)
                    train_images, train_labels = data.get_train_batch()
                    train_x = sess.run(code_flat, feed_dict={x: train_images})
                    if i == 0:
                        train_xs = train_x
                        train_labelss = train_labels
                    else:
                        train_xs = np.append(train_xs, train_x, axis=0)
                        train_labelss = np.append(train_labelss,
                                                  train_labels, axis=0)

                bar.finish()
                print('shape of part of train_set',
                      train_x.shape,
                      train_labels.shape)
                print('shape of train_set',
                      train_xs.shape,
                      train_labelss.shape)
                print('n_positive case in train_set', sum(train_labelss))

                test_images, test_labels = data.get_test_set()
                test_x = sess.run(code_flat, feed_dict={x: test_images})
                print('shape of test_set', test_x.shape)
                print('n_positive case in test_set', sum(test_labels))

        np.save(out_put_path+'train_set.npy',
                {'train_x': train_xs, 'train_labels': train_labelss})
        np.save(out_put_path+'test_set.npy',
                {'test_x': test_x, 'test_labelss': test_labels})
        # return ((train_xs, train_labelss), (test_x, test_labels))


if __name__ == '__main__':
    # Conv_autoencdoer(display=True)
    Conv_autoencdoer(generate=True, train_size=20000, test_size=1500)
