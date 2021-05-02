import numpy as np 
from math import log10

def psnr(original_image, denoised_image):
    mse         = np.linalg.norm(original_image - denoised_image) ** 2
    max_val     = np.amax(original_image)
    psnr_       = 20 * log10(max_val) - 10 * log10(mse)
    return psnr_ 

## TODO: Add more metrics ##