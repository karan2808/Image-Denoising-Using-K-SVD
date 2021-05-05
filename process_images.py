from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image as image_transforms
import cv2
import numpy as np


def resize_image(image, shape_w, shape_h):
    resized_img = cv2.resize(image, (shape_w, shape_h))
    return resized_img

# def get_patches(image, patch_size):

#     '''get image patches of N, patch_size, patch_size'''
#     patches = image_transforms.extract_patches_2d(image, (patch_size, patch_size))
#     return patches


def get_patches(image, patch_size):
    
    '''get image patches of N, patch_size, patch_size'''
    

    # Compute the selection tensor R
    indices = np.arange(image.size).reshape(image.shape)                                 
    patch_idx = image_transforms.extract_patches_2d(indices, (patch_size, patch_size))  # shape (num patches, patch_size, patch_size)
    patch_idx = patch_idx.reshape(-1, patch_size**2, 1)                                 # shape (num patches, patch_size*patch_size)

    N_p = patch_idx.shape[0]   # number of patches
    N = image.size             # number of pixels in image

    R = np.zeros((N_p, patch_size**2, N))  
    np.put_along_axis(R, patch_idx, 1, axis=-1)
     
    # Compute actual image patches from the selection matrices
    y = image.reshape(1,-1,1).repeat(N_p,axis=0)
    patches = R @ y
    patches = patches.reshape(-1, patch_size, patch_size)

    return R, patches

