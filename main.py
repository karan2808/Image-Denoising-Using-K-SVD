from ksvd import *
from process_images import *
import matplotlib.pyplot as plt 

# Hyperparameters
sigma = 0.25    # noise variance
img_size = 8    # size image to resized to 
patch_size = 3  # size of patches extracted from image
k = 50          # dictionary size
lam = 1       # noisy image weightage





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
num_iters = 50
D = np.random.randn(patch_size**2,k)

ksvd = KSVD()

for i in range(num_iters):
    A = ksvd.omp(D, X = patches, L = 3)
    D, A = ksvd.update_dictionary(D, patches, A)
    X_hat = ksvd.denoise(R, D, A, X.reshape(-1,1), lam)
    R, patches = get_patches(X_hat.reshape(X.shape), patch_size) 

cv2.imwrite('X_hat.png', (255*X_hat.reshape(X.shape)).astype(np.uint8))
# cv2.waitKey(1000)

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