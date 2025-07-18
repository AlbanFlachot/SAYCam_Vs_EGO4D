import os
import shutil
from os.path import join

path2full = '/raid/katha/datasets/objects_1000/test'
path2save = 'testsets/100_objects'

categories = os.listdir(path2full)

for c, cat in enumerate(categories):
    if (cat != '.DS_Store') & (not cat.endswith('.mat')):
        path = join(path2full, cat)
        imgname = os.listdir(path)[0]
        if imgname=='.DS_store':
            imgname = os.listdir(path)[1]
        dist = os.path.join(path2save, cat + '_' + imgname)
        shutil.copy(join(path2full, cat, imgname), dist)