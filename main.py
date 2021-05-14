# K-SVD Image denoising
from process_images import get_example_image, normalize
import matplotlib.pyplot as plt 
from dictionaries import get_dictionary
from denoiser import KSVDDenoiser
import cv2
from metrics import psnr
import numpy as np 

patch_size     = 8
K              = 16
sigma          = 5
iterations     = 5

# original_image = get_example_image(image_path='example_images/CameraMan.png', image_shape=(256, 256))
# noisy_image    = get_example_image(image_path='example_images/CameraMan.png', image_shape=(256, 256), sigma = sigma)

original_image = get_example_image(image_name='Barbara', image_shape=(512, 512))
noisy_image    = get_example_image(image_name='Barbara', image_shape=(512, 512), sigma = sigma)

dict_          = get_dictionary(patch_size, K)
# dict_          = np.random.normal(size = dict_.shape)

denoiser       = KSVDDenoiser(patch_size = patch_size, iterations = iterations, lambd = 30/sigma, sigma = sigma, noise_gain = 1.15, viz_dict=True)
denoised_img   = denoiser.denoise(noisy_image, dict_)

cv2.imwrite('original_image.png', normalize(original_image)*255)
cv2.imwrite('noisy_image.png', normalize(noisy_image)*255)
cv2.imwrite('denoised_image.png', normalize(denoised_img)*255)

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