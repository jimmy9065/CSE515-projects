import os
import sys
from scipy import misc
import re
import numpy as np
import shutil
from progressbar import ProgressBar, Bar, Percentage
from multiprocessing import Process
import matplotlib.pyplot as plt
from PIL import Image


# path for source images
path_input = '/home/jimmy/temp/VAE/train/'
# path for output
path_output = '/home/jimmy/WinDisk/data/'

img_size = 128


def normalize_image(image):
    img_y = image[:, :, 0].astype(float)

    img_y /= 255.0
    img_y -= img_y.mean()
    img_y /= img_y.std()
    scale = np.max([np.abs(np.percentile(img_y, 1.0)),
                    np.abs(np.percentile(img_y, 99.0))])
    img_y = img_y / scale
    img_y = np.clip(img_y, -1.0, 1.0)
    img_y = (img_y + 1.0) / 2.0

    image[:, :, 0] = (img_y * 255 + 0.5).astype(np.uint8)

    image = misc.toimage(image, mode='YCbCr').convert('RGB')
    image = np.array(image)
    return image

    # return misc.toimage(image, mode='YCbCr')


def padding_image(image, old_x, old_y, new_size):
    image = np.lib.pad(
            image,
            ((0, new_size-old_x), (0, new_size-old_y), (0, 0)),
            'constant',
            constant_values=0)

    # plt.imshow(image)
    # plt.show()
    # sys.exit()

    return image


def processImage(files, key):
    images = []
    img_files = [f for f in files if re.search('^'+key, f)]
    len_files = len(img_files)
    bar = ProgressBar(maxval=len_files, widgets=[
                      key+': ',
                      Percentage(),
                      Bar()])

    print(key, ' len=', len_files)
    bar.start()
    for i in range(len_files):
        img_file = img_files[i]
        bar.update(i+1)
        no = re.findall('\.([0-9]+)\.', img_file)
        image = misc.imread(path_input+img_file, mode='YCbCr')
        shape_x = image.shape[0]
        shape_y = image.shape[1]
        ratio = image.shape[0]/image.shape[1]
        if ratio <= 1.5 and ratio >= 0.66:
            x = min(img_size, shape_x)
            y = min(img_size, shape_y)

            image = misc.imresize(image, (x, y))
            image = normalize_image(image)
            image = padding_image(image, x, y, img_size)

            images.append(image)
            misc.imsave(path_output+'pics/'+key+no[0]+'.jpeg', image)
    bar.finish()
    print('saving '+key + 'imgs')
    # np.save(path_output+key, images)
    return images


def cleanData():
    while True:
        res = input('Do you want to delete the output[y/n]').lower()
        if res == 'y':
            print('cleaning')
            shutil.rmtree(path_output)
            os.mkdir(path_output)
            os.mkdir(path_output+'pics/')
            print('cleaned')
            break
        elif res == 'n':
            print('skip cleaning')
            break

    if not os.path.exists(path_output):
        os.mkdir(path_output)
    if not os.path.exists(path_output+'pics/'):
        os.mkdir(path_output+'pics/')


# clear old data
cleanData()
# cats=0 dogs=1
files = os.listdir(path_input)

procs = dict()
keys = ['dog', 'cat']
procs[1] = Process(target=processImage, args=(files, keys[0]))
procs[1].start()
procs[2] = Process(target=processImage, args=(files, keys[1]))
procs[2].start()

procs[1].join()
procs[2].join()

sys.exit()
# for key in ['dog', 'cat']:
#     images = processImage(files, key)

# images = np.array(images)
# print('info of images')
# print('images.shape= ', images.shape)
# print('data type=', type(images))
# 
# shapes = np.array([image.shape for image in images])
# print('max/min size:')
# print(max(shapes[:, 0]), min(shapes[:, 0]))
# print(max(shapes[:, 1]), min(shapes[:, 1]))
# 
# sys.exit()
