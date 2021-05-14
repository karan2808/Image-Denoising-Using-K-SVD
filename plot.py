import matplotlib.pyplot as plt 
import numpy as np 
from metrics import psnr, rmse, structural_similarity
import cv2

psnr_noisy = []
psnr_recon = []

rmse_noisy = []
rmse_recon = []

ssim_noisy = []
ssim_recon = []

original_image = cv2.imread('experiments/original_image_sigma_1.png').astype('float')

for i in range(1, 100, 5):
    denoised_image = cv2.imread('experiments/denoised_image_sigma_' + str(i) + '.png').astype('float')
    noisy_image    = cv2.imread('experiments/noisy_image_sigma_' + str(i) + '.png').astype('float')
    
    psnr_noisy.append(psnr(original_image, noisy_image))
    psnr_recon.append(psnr(original_image, denoised_image))

    rmse_noisy.append(rmse(original_image, noisy_image))
    rmse_recon.append(rmse(original_image, denoised_image))

    ssim_noisy.append(structural_similarity(original_image, noisy_image))
    ssim_recon.append(structural_similarity(original_image, denoised_image))

xx = np.arange(len(psnr_noisy))
plt.subplot(1, 3, 1)
plt.title('PSNR')
plt.plot(psnr_noisy, 'go--')
plt.plot(psnr_recon, 'bo--')
plt.subplot(1, 3, 2)
plt.title('RMSE')
plt.plot(rmse_noisy, 'go--')
plt.plot(rmse_recon, 'bo--')
plt.subplot(1, 3, 3)
plt.title('SSIM')
plt.plot(ssim_noisy, 'go--')
plt.plot(ssim_recon, 'bo--')
plt.show()