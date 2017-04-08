import os
import sys
from scipy import misc
from scipy import stats
import re
import numpy as np

path_train = '/home/jimmy/temp/VAE/train/'
path_data = '/home/jimmy/WinDisk/data/'

# cats=0 dogs=1
files = os.listdir(path_train)

labels = np.array([0 if re.search('^cat', f) else 1 for f in files])
# only extract cat img
key = 'cat'
images = np.array(
        [misc.imread(path_train+f) for f in files])
#        [misc.imread(path_train+f) for f in files if re.search('^'+key, f)])

# here the images is not a 4D ndarray but a array of 3D ndarray,
# because those 3D ndarray's shape is not identical.
# So we set dim[0]=375, dim[1]=500
shapes = np.array([image.shape for image in images])
ratios = np.array([image.shape[0]/image.shape[1] for image in images])

print(max(ratios), min(ratios), stats.mode(ratios))
print(min(shapes[:, 0]), '--', max(shapes[:, 0]), stats.mode(shapes[:, 0]))
print(min(shapes[:, 1]), '--', max(shapes[:, 1]), stats.mode(shapes[:, 1]))

# lower bound is around 0.65
# upper bound is around 1.5
# Then for cats we will throw 463+661=1124 pics
#      for dogs we will throw 681+321=1002 pics
# The loss will be minor

print('comparing ratios:')
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
for i in range(len(ratios)):
    if ratios[i] > 1.5:
        if re.search('^cat', files[i]):
            cnt1 += 1
        else:
            cnt2 += 1
    if ratios[i] < 0.67:
        if re.search('^cat', files[i]):
            cnt3 += 1
        else:
            cnt4 += 1

print('cats: ', cnt1+cnt3)
print('dogs: ', cnt2+cnt4)
cnt = cnt1+cnt2+cnt3+cnt4
print('total: ', cnt1+cnt2, cnt3+cnt4, cnt, 25000-cnt)

sys.exit()

print('comparing length&width')
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
for i in range(len(shapes)):
    if shapes[i][0] < 350:
        if re.search('^cat', files[i]):
            cnt1 += 1
        else:
            cnt2 += 1
    if shapes[i][1] >= 500:
        if re.search('^cat', files[i]):
            cnt3 += 1
        else:
            cnt4 += 1

print('cats: ', cnt1,cnt3)
print('dogs: ', cnt2,cnt4)
print('total: ', cnt1+cnt2, cnt3+cnt4, cnt1+cnt2+cnt3+cnt4)
sys.exit()

# split the data into 4 part, otherwise it might be hard to load into RAM(7GB)
images_parts = np.split(images, 4)

# save those part into npy file
cnt = 1
for images_part in images_parts:
    np.save(path_data+key+str(cnt), images_part)
    cnt += 1
