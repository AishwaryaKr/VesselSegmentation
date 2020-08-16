# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:08:09 2019

@author: aishwarya
"""

import numpy as np 
from skimage import io 
from skimage.external.tifffile import imsave 
from scipy.ndimage import label
from matplotlib import pyplot as plt 

# Functions 

def image_show(image, nrows=1, ncols=1, cmap='gray',size = 10):
    """ Displays the image
    Parameters:
    image -- the image to be loaded 
    nrows -- rows of subplot grid
    ncols -- column of subplot grid
    cmap -- colormap of the image (ex: greyscale, RGB, etc)
    size -- square image dimension
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size, size))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax 

def vol_filter(image):
    uniq = np.unique(image)

    condensed_rmv = list(set(uniq).intersection(rmv))

    for i in condensed_rmv:
        image[image == i] = 0
    
    image[image > 0] = 255
    #for i in range(12):
    #    img = array[i]
    #    img[img==1] = 0
    
    return image
    
# Main Parameters 

stack_path = r'\Users\aishw\Desktop\ImgProc\zstack\PythonVer\Ver2\Ver2.tif'

# Main Method 

stack = io.imread(stack_path) #tif stack, shape: num_frames, height, width, dtype: uint8

# Labelling Image by describing connectivity 
labelled = label(stack)
array = labelled [0]
# array = array.astype(np.uint8)

# calculating frequency of label numbers 
unique, counts = np.unique(array, return_counts=True)

#threshold frequency(volume) 
volume_thres = 400

# list of objects smaller than threshold 
rmv = []

for i in range(len(counts)):
    if counts[i] < volume_thres:
        rmv.append(i) 
        
for i in range(50):
    image = array[i]
    image = vol_filter(image)
    imsave('00000' + str(i) + '.tif', image)        

        
## single image of stack 
#filtered = np.zeros(stack.shape,dtype = np.uint8)
#for idx, frame in tqdm(enumerate(array)):
#    filtered[idx] = vol_filter(frame)
#save_path = r'\Users\aishw\Desktop\ImgProc\zstack\PythonVer\Ver2\Filtered.tif'
#imsave(save_path, filtered)
