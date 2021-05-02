from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image as image_transforms
import cv2


def resize_image(image, shape_w, shape_h):
    resized_img = cv2.resize(image, (shape_w, shape_h))
    return resized_img

def get_patches(image, patch_size):
    '''get image patches of N, patch_size, patch_size'''
    patches = image_transforms.extract_patches_2d(image, (patch_size, patch_size))
    return patches

