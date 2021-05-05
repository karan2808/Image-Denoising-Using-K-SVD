from ksvd import *
from process_images import *
import matplotlib.pyplot as plt 

# Hyperparameters
sigma = 0.25    # noise variance
img_size = 32   # size image to resized to 
patch_size = 3  # size of patches extracted from image
k = 16          # dictionary size
lam = 0.5       # noisy image weightage





# Load the image
img = cv2.imread('./examples/original_images/camera_man.png', 0)
img = resize_image(img, img_size, img_size)
img = img/255
cv2.imwrite('noisy_camera_man.png', (255*img).astype(np.uint8))


# Add noise
X = img + np.random.normal(0, sigma, size = img.shape)
X = np.clip(X, 0, 1)


# Get patches
R, patches = get_patches(X, patch_size)


# Run K-SVD Denoising algorithm
ksvd = KSVD()


# n = 4
# k = 5
# N_p = 10
# N = 30

# D = np.random.rand(n,k)
# P = np.random.rand(n, N_p)
# A = np.random.rand(k,N_p)
# R = np.random.rand(N_p, n, N)
# y = np.random.rand(N, 1)

# ksvd = KSVD(rank=None, sparsity=None, max_iterations=None, max_tolerance=None)
# ksvd.denoise(R, D, A, y, lam=1)