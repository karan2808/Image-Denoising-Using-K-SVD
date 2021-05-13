# K-SVD Image denoising
from process_images import get_example_image, normalize
import matplotlib.pyplot as plt 
from dictionaries import get_dictionary
from denoiser import KSVDDenoiser

patch_size     = 8
K              = 11

original_image = get_example_image('Barbara', image_shape=(256, 256))
noisy_image    = get_example_image('Barbara', image_shape=(256, 256), sigma = 20)

dict_          = get_dictionary(patch_size, K)
denoiser       = KSVDDenoiser(patch_size = 8, iterations = 5, lambd = 0.5, sigma = 20, noise_gain = 1.15, viz_dict=False)
denoised_img   = denoiser.denoise(noisy_image, dict_)

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