import os
# import sys
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from numpy import random
import matplotlib.pyplot as plt


class DC_dataset:
    NUM_COLORS = 3
    IMAGE_SIZE = 128

    def __init__(self, f_path, sess,
                 test_size=1729, batch_size=1000, seed=0, num_tread=4):
        """
        input set path, randomly pick 1729(default) instance as test case
        intput sess
        """
        self._sess = sess
        self._test_size = test_size
        self._batch_size = batch_size
        self._seed = seed

        self.test_size = test_size

        files = [f_path+f for f in os.listdir(f_path) if re.search('.jpeg', f)]
        self._len_files = len(files)
        # generate labels
        labels = [1 if re.search('dog[0-9]+.jpeg', f) else 0 for f in files]

        self._all_images = ops.convert_to_tensor(files, dtype=dtypes.string)
        self._all_labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)

        self.generate_test_cases()

        # Produce a queue of the filenames
        train_input_queue = tf.train.slice_input_producer(
                [self.train_images, self.train_labels],
                shuffle=True,  # shuffle the queue when output
                seed=self._seed)

        test_input_queue = tf.train.slice_input_producer(
                [self.test_images, self.test_labels],
                shuffle=False)

        # return single image everytime read the input queue
        # pipeline for train case
        file_content = tf.read_file(train_input_queue[0])

        train_image_label_list = []
        for i in range(num_tread):
            train_image = tf.image.decode_jpeg(
                    file_content,
                    channels=self.NUM_COLORS)
            train_label = train_input_queue[1]
            train_image.set_shape(
                    [self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_COLORS])
            train_image_label_list.append([train_image, train_label])

        # pipeline for test case
        file_content = tf.read_file(test_input_queue[0])

        test_image_label_list = []
        # changed here num_tread
        for i in range(4):
            test_image = tf.image.decode_jpeg(
                    file_content,
                    channels=self.NUM_COLORS)
            test_label = test_input_queue[1]
            test_image.set_shape(
                    [self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_COLORS])
            test_image_label_list.append([test_image, test_label])

        train_image_batch_int, self._train_label_batch = tf.train.batch_join(
                train_image_label_list,
                batch_size=batch_size)

        test_image_batch_int, self._test_labels_batch = tf.train.batch_join(
                test_image_label_list,
                batch_size=batch_size)

        self._train_image_batch = tf.to_float(train_image_batch_int) / 255.0
        self._test_image_batch = tf.to_float(test_image_batch_int) / 255.0

        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(
                coord=self._coord,
                sess=self._sess)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_train()

    def generate_test_cases(self):
        """
        This is only choose some of the image to
        be the test case, not return a test batch
        """
        random.seed(self._seed)
        partitions = [0] * self._len_files
        partitions[:self.test_size] = [1] * self.test_size
        random.shuffle(partitions)

        self.train_images, self.test_images = tf.dynamic_partition(
                self._all_images,
                partitions,
                2)
        self.train_labels, self.test_labels = tf.dynamic_partition(
                self._all_labels,
                partitions,
                2)

    def get_train_batch(self):
        """
        return train_image_batch and train_labels_batch
        the order of example is shuffle in every iteration and epoch
        """
        v = self._sess.run([self._train_image_batch, self._train_label_batch])
        return v

    def get_test_set(self):
        """
        return test_image and test_labels
        """
        v = self._sess.run([self._test_image_batch, self._test_labels_batch])
        return v

    def end_train(self):
        self._coord.request_stop()
        self._coord.join(self._threads)


def Display_imgs(image1, image2, n_image):
    fig = plt.figure()
    for i in range(n_image):
        a = fig.add_subplot(2, n_image, i+1)
        plt.imshow(image1[i, :, :, :])
        a.set_title('test1-'+str(i))
        a = fig.add_subplot(2, n_image, i+n_image+1)
        plt.imshow(image2[i, :, :, :])
        a.set_title('test2-'+str(i))
    plt.show()


if __name__ == '__main__':
    print('Running test main in pipeline.py')
    train_path = '/home/jimmy/WinDisk/data/pics/'
    with tf.Session() as sess:
        Test_Size = 2000
        with DC_dataset(train_path, sess,
                        test_size=Test_Size, batch_size=500) as data:
            # data = DogsvsCats(train_path, sess, batch_size=500)
            for it in range(10):
                train_images, train_labels = data.get_train_batch()
                print('train_images shape in every batch', train_images.shape)
                print('left top pixel RGB values[0,1]:',
                      train_images[0, 0, 0, :])

            for ib in range(int(Test_Size/500)):
                test_images, test_labels = data.get_test_set()
                print('test_lables shape= ', test_labels.shape)
                print('first 10th test_labels:', test_labels[:10])
                print('number of positive cases in test_labels:',
                      sum(test_labels))
            print('new start')

            for ib in range(int(Test_Size/500)):
                test_images2, test_labels2 = data.get_test_set()
                print('test_lables shape= ', test_labels2.shape)
                print('first 10th test_labels:', test_labels2[:10])
                print('number of positive cases in test_labels:',
                      sum(test_labels2))

            n_image = 10
            Display_imgs(test_images[:n_image, :, :, :],
                         test_images2[:n_image, :, :, :], n_image)
