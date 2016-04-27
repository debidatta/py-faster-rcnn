import glob
import random
from PIL import Image
import numpy as np
import sys
import os
import multiprocessing

def get_annotations(img_f):
    img = Image.open(img_f)
    m = np.array(img.split()[-1])
    s = np.where(m == 255.0)
    if len(s[0]) != 0:
        x1, x2 = min(s[1]), max(s[1])
        y1, y2 = min(s[0]), max(s[0])
        anno_f = img_f[:-3] + 'txt'
        with open(anno_f, 'w') as f:
           f.write('%s %s %s %s'%(x1, y1, x2, y2))
    else:
        print img_f
    return "Done"

img_dir = sys.argv[1]
imgs = glob.glob(img_dir+'*/*.png')
pool = multiprocessing.Pool(processes=60)
for i in pool.imap_unordered(get_annotations, imgs):
    i
