# K-SVD Image denoising
from process_images import get_example_image, normalize
import matplotlib.pyplot as plt 
from dictionaries import get_dictionary
from denoiser import KSVDDenoiser
import cv2
from metrics import psnr
import numpy as np 
from metrics import psnr, rmse, structural_similarity

patch_size     = 8
K              = 16
sigma          = 1
iterations     = 3

psnr_vals           = []
rmse_vals           = []
ssim_vals           = []

# original_image = get_example_image(image_path='example_images/CameraMan.png', image_shape=(256, 256))
# noisy_image    = get_example_image(image_path='example_images/CameraMan.png', image_shape=(256, 256), sigma = sigma)

original_image = get_example_image(image_name='Barbara', image_shape=(256, 256))


for sigma in range(1, 100, 5):
    dict_          = get_dictionary(patch_size, K)
    noisy_image    = get_example_image(image_name='Barbara', image_shape=(256, 256), sigma = sigma)
    # dict_          = np.random.normal(size = dict_.shape)
    denoiser       = KSVDDenoiser(patch_size = patch_size, iterations = iterations, lambd = 30/sigma, sigma = sigma, noise_gain = 1.15, viz_dict=True)
    denoised_img   = denoiser.denoise(noisy_image, dict_)
    cv2.imwrite('experiments/original_image_sigma_' + str(sigma) + '.png', normalize(original_image)*255)
    cv2.imwrite('experiments/noisy_image_sigma_' + str(sigma) + '.png', normalize(noisy_image)*255)
    cv2.imwrite('experiments/denoised_image_sigma_' + str(sigma) + '.png', normalize(denoised_img)*255)

plt.subplot(1, 3, 1)
plt.title('original image')
plt.imshow(normalize(original_image), cmap='gray')
plt.subplot(1, 3, 2)
plt.title('noisy image')
plt.imshow(normalize(noisy_image), cmap='gray')
plt.subplot(1, 3, 3)
plt.title('denoised image')
plt.imshow(normalize(denoised_img), cmap='gray')
plt.show()