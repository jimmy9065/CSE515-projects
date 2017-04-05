import os
import sys
from scipy import misc
import re
import numpy as np

path_train = '/home/jimmy/temp/VAE/train/'
path_data = '/home/jimmy/WinDisk/data/'

# cats=0 dogs=1
files = os.listdir(path_train)

# only extract cat img
key = 'dog'
cat_files = [f for f in files if re.search('^'+key, f)]
# images = [misc.imread(path_train+f) for f in files if re.search('^'+key, f)]
print(len(cat_files))

images = []
for i in range(len(cat_files)):
    image = misc.imread(path_train+cat_files[i])
    shape_x = image.shape[0]
    shape_y = image.shape[1]
    ratio = image.shape[0]/image.shape[1]
    if ratio <= 1.55 and ratio >= 0.6:
        x = min(375, shape_x)
        y = min(500, shape_y)
        image = misc.imresize(image, (x, y))
        image = np.lib.pad(
                image,
                ((0, 375-x), (0, 500-y), (0, 0)),
                'constant',
                constant_values=0)
        # print(image.shape)
        images.append(image)
        misc.imsave(path_data+'pics/'+key+str(i+1)+'.jpeg', image)

images = np.array(images)
print('info of images')
print(images.shape)
print(type(images))

shapes = np.array([image.shape for image in images])
print('max size:')
print(max(shapes[:, 0]), min(shapes[:, 0]))
print(max(shapes[:, 1]), min(shapes[:, 1]))

print('saving')
np.save(path_data+key, images)

sys.exit()
# split the data into 4 part, otherwise it might be hard to load into RAM
images_parts = np.split(images, 2)

# save those part into npy file
cnt = 1
for images_part in images_parts:
    np.save(path_data+key+str(cnt), images_part)
    cnt += 1
