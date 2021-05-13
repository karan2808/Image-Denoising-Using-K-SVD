import numpy as np 
from sklearn.feature_extraction.image import extract_patches_2d
from process_images import get_Y
from sklearn.linear_model import orthogonal_mp
import matplotlib.pyplot as plt 

class KSVDDenoiser():
    def __init__(self, patch_size = 8, iterations = 10, lambd = 10, sigma = 3, sparsity = 4, noise_gain = 1.15, viz_dict = False):
        self.lambd          = lambd
        self.iterations     = iterations
        self.patch_size     = patch_size
        self.image_shape    = None
        self.sigma          = sigma
        self.noise_gain     = noise_gain
        self.sparsity       = sparsity
        self.A              = None
        self.D              = None
        self.viz_dict       = viz_dict


    def denoise(self, image, dictionary):

        self.D = dictionary
        
        if image.shape[0] != image.shape[1]:
            print("The image must be a square matrix")
            return

        self.image_shape = image.shape[0]
        
        # convert image to patches and vectorize 
        Y = get_Y(image, self.patch_size)

        for itr in range(self.iterations):
            print("Iteration Number " + str(itr))
            self.sparse_code(Y)
            self.dictionary_update(Y)
            if self.viz_dict:
                self.visualize_dictionary()

        recon_img   = np.zeros(image.shape)
        weight_img  = np.zeros(image.shape)

        # Source: Matlab code in Michael Elads book cf. Elad, M. (2010).
        i, j = 0, 0
        for k in range((self.image_shape - self.patch_size + 1)** 2):
            patch       = np.reshape(self.D @ self.A[:, k], (self.patch_size, self.patch_size))
            recon_img[j:j + self.patch_size, i:i + self.patch_size] += patch
            weight_img[j:j + self.patch_size, i:i + self.patch_size] += 1
            if i < self.image_shape - self.patch_size:
                i += 1
            else:
                i, j = 0, j + 1
        return np.divide(recon_img + self.lambd * image, weight_img + self.lambd)

    def sparse_code(self, Y):
        print("Running OMP..")
        # self.A = orthogonal_mp(self.D, Y, n_nonzero_coefs=self.sparsity,
        #                      tol=self.noise_gain * self.sigma, precompute=True)
        n, K     = self.D.shape
        alphas   = []
        for yy in Y.T:
            coeffs = np.zeros(K)
            res    = yy
            i      = 0
            while True:
                proj    = np.dot(self.D.T, res)
                max_i   = int(np.argmax(np.abs(proj)))
                alpha   = proj[max_i]
                res     = res - alpha * self.D[:, max_i]
                if np.isclose(alpha, 0):
                    break
                # update coefficients
                coeffs[max_i] += alpha
                if self.noise_gain is not None:
                    if np.linalg.norm(res) ** 2 < n * (self.noise_gain * self.sigma)**2 or i > n/2:
                        break
                else:
                    if np.count_nonzero(coeffs) >= self.sparsity:
                        break
                i += 1
            alphas.append(coeffs)
        self.A = np.array(alphas).T

    def dictionary_update(self, Y):
        print("Updating Dictionary..")
        n, K = self.D.shape 
        Res  = Y - np.dot(self.D, self.A)
        for k in range(K):
            # get non zero entries 
            nk = np.nonzero(self.A[k, :])[0]
            if len(nk):
                continue
            Ri             = np.dot(self.D[:, k, None], self.A[None, k, nk]) + Res[:, nk]
            U, Sg, V       = np.linalg.svd(Ri)
            self.D[:, k]   = U[:, 0]
            # 1 svd step
            self.A[k, nk]  = Sg[0] * V[0, :]
            Res[:, nk]     = Ri - np.dot(self.D[:, k, None], self.A[None, k, nk])

    def visualize_dictionary(self):
        n, K = self.D.shape
        # patch size
        n_r = int(np.sqrt(n))
        # patches per row / column
        K_r = int(np.sqrt(K))
        # we need n_r*K_r+K_r+1 pixels in each direction
        dim = n_r * K_r + K_r + 1
        V = np.ones((dim, dim)) * np.min(self.D)
        # compute the patches
        patches = [np.reshape(self.D[:, i], (n_r, n_r)) for i in range(K)]
        # place patches
        for i in range(K_r):
            for j in range(K_r):
                V[j * n_r + 1 + j:(j + 1) * n_r + 1 + j, i * n_r + 1 + i:(i + 1) * n_r + 1 + i] = patches[
                    i * K_r + j]
        V *= 255
        plt.imshow(V)
        plt.show()