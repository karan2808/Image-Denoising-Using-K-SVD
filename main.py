from ksvd import *
from process_images import *
import matplotlib.pyplot as plt 

# Hyperparameters
sigma = 0.25         # noise variance
img_size = 100       # size image to resized to 
patch_size = 20      # size of patches extracted from image
k = 256              # dictionary size
lam = 1              # noisy image weightage
n_nonzero_coefs = 3  # number of nonzero entries in alphas
num_iters = 1        # iterations

# Load the image
img = cv2.imread('./examples/original_images/camera_man.png', 0)
img = cv2.resize(img, (img_size, img_size))
img = img / 255
cv2.imwrite('noisy_camera_man.png', (255*img).astype(np.uint8))


# Add noise
X = img + np.random.normal(0, sigma, size = img.shape)
X = np.clip(X, 0, 1)


# Get patches
patches = get_patches(X, patch_size)


# Run K-SVD Denoising algorithm
# D = np.random.randn(patch_size**2, k)
D = get_overcomplete_dictionary(patch_size, int(np.sqrt(k)))
visualize_dictionary(D)


ksvd = KSVD()
# print(D.reshape((36,-1)).shape)
# input()

for i in range(num_iters):
    # reg = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs).fit(X=D.reshape((36,-1)), y=patches.T)

    # reg = OrthogonalMatchingPursuit().fit(X=D, y=patches)
    # A = (reg.coef_).T
    A = ksvd.omp(D, patches, 3)

    D, A = ksvd.update_dictionary(D, patches, A)
    visualize_dictionary(D, 'D1.png')
    # X_hat = ksvd.denoise(R, D, A, X.reshape(-1,1), lam)
    # R, patches = get_patches(X_hat.reshape(X.shape), patch_size) 

    # if i%5==0:
    #     img_hat = 255*X_hat.reshape(X.shape)
    #     cv2.imwrite('X_hat_'+str(i)+'.png', (255*X_hat.reshape(X.shape)).astype(np.uint8))


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