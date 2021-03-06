import tensorflow as tf
# import numpy as np

import sys

from pipeline import DogsvsCats


sess = tf.Session()
train_path = '/home/jimmy/WinDisk/data/pics/'

data = DogsvsCats(train_path, sess)

test_images, test_labels = data.get_test_set()
print(test_labels.shape)
print(sum(test_labels))

cnt1 = sum(test_labels)
cnt2 = len(test_labels)
for i in range(11):
    images, labels = data.get_train_batch()
    if i == 0:
        print(images.shape)
    print(sum(labels))
    cnt1 += sum(labels)
    cnt2 += len(labels)
print(cnt1, '/', cnt2)

cnt1 = sum(test_labels)
cnt2 = len(test_labels)
for i in range(11):
    images, labels = data.get_train_batch()
    if i == 0:
        print(images.shape)
    print(sum(labels))
    cnt1 += sum(labels)
    cnt2 += len(labels)
print(cnt1, '/', cnt2)

data.end_train()

sys.exit()
