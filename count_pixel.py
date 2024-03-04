


import glob
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib

#INSTANCE voxel size is 0.429 0.429 5.

mask_dir = '/home/data_repo/INSTANCE_ICH/train_2/label/'
mask_dir = '/home/data_repo/physionetICH/masks/'

data = []
for p in glob.glob(mask_dir+'*'):
    im = nib.load(p)
    im = im.get_fdata()
    #data+= [ (p.split('/')[-1] , im.sum() * 0.429* 0.429 * 5 )]
    data+= [ (p.split('/')[-1] , im.sum() * 0.41796875* 0.41796875 * 5 )]



data = pd.DataFrame(data)
data.columns = ['name','volume']
data.query('(volume < 10000000) & (volume >0)')['volume'].plot.hist(bins=50)

plt.title('Volumze size.')

print(data['volume'].describe())

plt.savefig('hist.png')
