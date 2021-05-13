from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image as image_transforms
from sklearn.feature_extraction.image import extract_patches_2d
import cv2
import numpy as np

def get_example_image(image_name=None, image_shape=(512, 512), sigma=None, image_path=None): 
    image = 0
    # Load the image 
    if image_name is not None:
        try:
            image = cv2.imread('example_images/' + str(image_name) + '.png', cv2.IMREAD_GRAYSCALE)
        except:
            print("Example Image Not Found")
            return
    else:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except:
            print("Path Not Found")
            return
    # Reshape the image
    image = cv2.resize(image, image_shape).astype('float32')
    
    # Add noise to the image
    if sigma != None:
        image = image + np.random.normal(scale=sigma,
                                    size=image.shape).astype(image.dtype)
    
    return image

def normalize(image):
    return (image - np.amin(image)) / (np.amax(image) - np.amin(image))

def get_Y(image, patch_size):
    patches = extract_patches_2d(image, (patch_size, patch_size))
    patches = [patch.reshape(patch_size**2) for patch in patches]
    Y       = np.array(patches).T
    return Y