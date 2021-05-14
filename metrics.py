import numpy as np 
from math import log10
from skimage.metrics import structural_similarity as ssim

def rmse(original_image, denoised_image):
    num_pixels = original_image.shape[0] * original_image.shape[1]
    err        = np.sum((original_image - denoised_image) ** 2)
    return (err / num_pixels)**(1/2)

def psnr(original_image, denoised_image):
    # return peak_signal_noise_ratio(original_image, denoised_image)
    mse_        = (rmse(original_image, denoised_image))**2
    max_val     = np.amax(original_image)
    psnr_       = 20 * log10(max_val) - 10 * log10(mse_)
    return psnr_ 

def structural_similarity(original_image, denoised_image):
    return ssim(original_image, denoised_image,
                  data_range=np.amax(original_image) - np.amin(original_image), multichannel=True)