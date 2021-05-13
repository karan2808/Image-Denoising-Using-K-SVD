from ksvd import *
from process_images import *
from metrics import *
from sklearn.linear_model import orthogonal_mp
import matplotlib.pyplot as plt 

# Hyperparameters
sigma = 5            # noise variance
img_size = 64       # size image to resized to 
patch_size = 8       # size of patches extracted from image
k = 256              # dictionary size
lam = 100000000      # noisy image weightage
n_nonzero_coefs = 10  # number of nonzero entries in alphas
num_iters = 10       # iterations

# Load the image
img = cv2.imread('./examples/original_images/camera_man.png', 0)
img = cv2.resize(img, (img_size, img_size))
# img = img / 255
# cv2.imwrite('./new_test/noisy_camera_man.png', img)


# Add noise
X = img + np.random.normal(0, sigma, size = img.shape).astype(img.dtype)
print(img.dtype)
print(X.dtype)
cv2.imwrite('./new_test/noisy_camera_man2.png', X)
print('printed image')
# X = np.clip(X, 0, 1)
# print(np.amax(X))
# input()


# Get patches
patches = get_patches(X, patch_size)
patches_img = get_patches(img, patch_size)


# Run K-SVD Denoising algorithm
# D = np.random.randn(patch_size**2, k)
D = get_overcomplete_dictionary(patch_size, int(np.sqrt(k)))
visualize_dictionary(D)
print(D.dtype)
print('printed dictionary')

for i in range(num_iters):
    print(i)
    # reg = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs).fit(X=D.reshape((36,-1)), y=patches.T)
    # reg = OrthogonalMatchingPursuit().fit(X=D, y=patches) #tol=(1.075*sigma)**2
    # A =orthogonal_mp(D, patches, tol=(1.15*sigma)**2, precompute=False)
    # print(A)
    # input()

    # print('got reg')
    # A = (reg.coef_).T
    A = omp(D, patches, n_nonzero_coefs)
    D, A = update_dictionary(D, patches, A)


    print(np.linalg.norm(patches_img - D@A))


    # if i%5==0:
    visualize_dictionary(D, './new_test/D_'+str(i)+'.png')
    X_hat = denoise(img, patch_size, D, A, lam)
    # X_hat = (X_hat - np.amin(X_hat))/(np.amax(X_hat)-np.amin(X_hat))
    print(psnr((img*255).astype(np.uint8), (255*X_hat.reshape(X.shape)).astype(np.uint8)))
    plt.imsave('./new_test/X_'+str(i)+'.png', X_hat.reshape(X.shape), cmap='gray')



    
