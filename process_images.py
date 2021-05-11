from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image as image_transforms
import cv2
import numpy as np


def resize_image(image, shape_w, shape_h):
    resized_img = cv2.resize(image, (shape_w, shape_h))
    return resized_img

# def get_patches(image, patch_size):

#     '''get image patches of N, patch_size, patch_size'''
#     patches = image_transforms.extract_patches_2d(image, (patch_size, patch_size))
#     return patches


def get_patches(image, patch_size):
    
    '''get image patches of N, patch_size, patch_size'''
    


    # Compute the selection tensor R
    indices = np.arange(image.size).reshape(image.shape)                                 
    patch_idx = image_transforms.extract_patches_2d(indices, (patch_size, patch_size))  # shape (num patches, patch_size, patch_size)
    patch_idx = patch_idx.reshape(-1, patch_size**2, 1)                                 # shape (num patches, patch_size*patch_size)

    N_p = patch_idx.shape[0]   # number of patches
    N = image.size             # number of pixels in image

    R = np.zeros((N_p, patch_size**2, N))  
    np.put_along_axis(R, patch_idx, 1, axis=-1)

    # Compute actual image patches from the selection matrices
    y = image.reshape(1,-1,1).repeat(N_p,axis=0)
    patches = R @ y



    patches = patches.reshape(patch_size**2, -1)


    return R, patches

def get_overcomplete_dictionary(n, K, normalized=True, inverse=True):
    """
    Builds a Dictionary matrix matrix using the inverse discrete cosine transform of type II,
    cf. https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    Args:
        n: number of dictionary rows
        K: number of dictionary columns
        normalized: If True, the columns will be l2-normalized
        inverse: Uses the inverse transform (as usually needed in applications)
    Returns:
        Dictionary build from the Kronecker-Delta of the inverse discrete cosine transform of type II applied to the identity.
    """
    D = np.zeros((K, n))
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        y = np.array([sum(np.multiply(v, np.cos((0.5 + np.arange(n)) * k * np.pi / K))) for k in range(K)])
        if normalized:
            y[0] = 1 / np.sqrt(2) * y[0]
            y = np.sqrt(2 / n) * y
        D[:, i] = y

    if inverse:
        D = D.T
    return np.kron(D.T, D.T)

def visualize_dictionary(D):
    n, K = D.shape
    M = D
    # stretch atoms
    for k in range(K):
        M[:, k] = M[:, k] - (M[:, k].min())
        if M[:, k].max():
            M[:, k] = M[:, k] / D[:, k].max()

    # patch size
    n_r = int(np.sqrt(n))

    # patches per row / column
    K_r = int(np.sqrt(K))

    # we need n_r*K_r+K_r+1 pixels in each direction
    dim = n_r * K_r + K_r + 1
    V = np.ones((dim, dim)) * np.min(D)

    # compute the patches
    patches = [np.reshape(D[:, i], (n_r, n_r)) for i in range(K)]

    # place patches
    for i in range(K_r):
        for j in range(K_r):
            V[j * n_r + 1 + j:(j + 1) * n_r + 1 + j, i * n_r + 1 + i:(i + 1) * n_r + 1 + i] = patches[
                i * K_r + j]
    V *= 255
    cv2.imshow('V.png', V.astype(np.uint8))
    cv2.waitKey(0)