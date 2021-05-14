import cv2
from metrics import psnr, rmse, structural_similarity

original_image = cv2.imread('original_image.png').astype('float')
denoised_image = cv2.imread('noisy_image.png').astype('float')

psnr_ = psnr(original_image, denoised_image)
print("The Peak Signal To Noise Ratio is " + str(psnr_))

mse_ = rmse(original_image, denoised_image)
print("The Root Mean Squared Error is " + str(mse_))

ssim = structural_similarity(original_image, denoised_image)
print("The structural similarity index is " + str(ssim))