import numpy as np
# Test cell, to be deleted
# File paths of the .npy images
root = '../data/VOC12/img_with_margin_0/'
test = root+' test/'
train = root+'train/'
train_aug = root+'train_aug/'
val = root+'val/'

# root_src = '../VOC12/img_with_margin_0/'
# test_src = root_src+' test/'
# train_src = root_src+'train/'
# train_aug_src = root_src+'train_aug/'
# val_src = root_src+'val/'
f = '2007_000033.npy'

arr=np.load(val+f)

arr2 = arr[:,-1]

print(arr2.ndim , arr.ndim)
print(arr)
