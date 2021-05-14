import cv2
from metrics import psnr

original_image = cv2.imread('original_image.png')
denoised_image = cv2.imread('denoised_image.png')

psnr_ = psnr(original_image, denoised_image)
print("The psnr is " + str(psnr_))