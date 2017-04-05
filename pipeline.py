import os
import sys
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from numpy import random


class DogsvsCats:
    NUM_COLORS = 3
    IMAGE_HEIGHT = 375
    IMAGE_WIDTH = 500

    def __init__(self, f_path, sess, test_size=3647, batch_size=2000, seed=0):
        """
        input set path, randomly pick 3647(default) instance as test case
        intput sess
        """
        self._sess = sess
        self._test_size = test_size
        self._batch_size = batch_size
        self._seed = seed

        self.test_size = test_size

        files = [f_path+f for f in os.listdir(f_path)]
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
                [self.test_images, self.test_labels])

        # return single image everytime read the input queue
        file_content = tf.read_file(train_input_queue[0])
        train_image = tf.image.decode_jpeg(
                file_content,
                channels=self.NUM_COLORS)

        train_labels = train_input_queue[1]

        file_content = tf.read_file(test_input_queue[0])
        test_image = tf.image.decode_jpeg(
                file_content,
                channels=self.NUM_COLORS)

        test_labels = test_input_queue[1]

        # Now train_image and test_image only has one image
        train_image.set_shape(
                [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_COLORS])
        test_image.set_shape(
                [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_COLORS])

        self._train_image_batch, self._train_labels_batch = tf.train.batch(
                [train_image, train_labels],
                batch_size=batch_size)

        self._test_image_batch, self._test_labels_batch = tf.train.batch(
                [test_image, test_labels],
                batch_size=test_size)

        sess.run(tf.global_variables_initializer())
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(
                coord=self._coord,
                sess=self._sess)

    def generate_test_cases(self):
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
        v = self._sess.run([self._train_image_batch, self._train_labels_batch])
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


if __name__ == '__main__':
    print('here***************************************8')
    sess = tf.Session()
    train_path = '/home/jimmy/WinDisk/data/pics/'
    data = DogsvsCats(train_path, sess)

    # for it in range(10):
    #     train_images, train_labels = data.get_train_batch()

    test_images, test_labels = data.get_test_set()
    print(test_labels.shape)

    sys.exit()
