# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.external.tifffile import imsave
from scipy import ndimage

# Functions Used

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


# Main Parameters
for x in range(1000, 1236, 1):
    stack_path = r'\Users\aishw\Desktop\ImgProc\zstack\Images\data_00000' + str(999) + '.tif'
    
    # Main Method 
    
    stack = cv2.imread(stack_path) #tif stack, shape: num_frames, height, width, 
    #                                                              dtype: uint8
    
#    image_show(stack)
    image = cv2.cvtColor(stack,cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    cl1 = clahe.apply(image)
#    image_show(cl1)
    
    ret,image = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
    image = image.astype(np.uint8)
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 5, 7)
        
    #image_show(img)
    
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    new = np.zeros(img.shape,dtype=np.uint8)
    
    area_thres = 10
    
    cv2.drawContours(new, contours, -1, 255, -1)
    
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if (area < area_thres): 
            cv2.drawContours(new, contours, i, 0, thickness=cv2.FILLED)        
    
    new = ndimage.binary_fill_holes(new).astype(np.uint8)
    
#    image_show(new)
    imsave('00000' + str(999) + '.tif', new) 
    


