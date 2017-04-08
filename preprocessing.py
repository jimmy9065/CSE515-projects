import os
import sys
from scipy import misc
import re
import numpy as np
from subprocess import call


# path for source images
path_train = '/home/jimmy/temp/VAE/train/'
# path for output
path_data = '/home/jimmy/WinDisk/data/'

# clear old data
call(["rm", path_data+'*', '-r'])
call(["mkdir", path_data+'pics'])
print("cleaned dest path")

# cats=0 dogs=1
files = os.listdir(path_train)

# only extract cat img
# images = [misc.imread(path_train+f) for f in files if re.search('^'+key, f)]

len_x = 256
len_y = 256

for key in ['dog', 'cat']:
    images = []
    img_files = [f for f in files if re.search('^'+key, f)]
    print(len(img_files))
    for img_file in img_files:
        no = re.findall('\.([0-9]+)\.', img_file)
        image = misc.imread(path_train+img_file)
        shape_x = image.shape[0]
        shape_y = image.shape[1]
        ratio = image.shape[0]/image.shape[1]
        if ratio <= 1.5 and ratio >= 0.66:
            x = min(len_x, shape_x)
            y = min(len_y, shape_y)
            image = misc.imresize(image, (x, y))
            image = np.lib.pad(
                    image,
                    ((0, len_x-x), (0, len_y-y), (0, 0)),
                    'constant',
                    constant_values=0)
            images.append(image)
            misc.imsave(path_data+'pics/'+key+no[0]+'.jpeg', image)
    print('saving '+key + 'imgs')
    np.save(path_data+key, images)

images = np.array(images)
print('info of images')
print('images.shape= ', images.shape)
print('data type=', type(images))

shapes = np.array([image.shape for image in images])
print('max/min size:')
print(max(shapes[:, 0]), min(shapes[:, 0]))
print(max(shapes[:, 1]), min(shapes[:, 1]))

sys.exit()
