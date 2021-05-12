import numpy as np 
import numpy.matlib
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.sparse.linalg import svds


# class KSVD:
#     def __init__(self):
#         return

    # def __init__(self, rank = 1, sparsity = 1, max_iterations = 1, max_tolerance = 1):
        # self.rank           = rank 
        # self.sparsity       = sparsity
        # self.max_iterations = max_iterations
        # self.max_tolerance  = max_tolerance

    # def find_distance_between_dictionaries(self, original, new):
    #     catch_counter = 0
    #     total_dist    = 0
    #     for i in range(new.shape[1]):
    #         new[:, i] = np.sign(new[0, i]) * new[:, i]
        
    #     for i in range(original.shape[1]):
    #         d = np.sign(original[0, i]) * original[:, i]
    #         distances = np.sum((new - np.matlib.repmat(d, 1, new.shape[1]))**2, axis = 0)
    #         minval    = np.amin(distances)
    #         idx       = np.argmin(distances)
    #         err       = 1 - np.absolute(new[:, idx].T @ d)
    #         total_dist    = total_dist + err
    #         catch_counter = catch_counter + np.sum(err < 0.01)
    #     ratio = 100 * catch_counter/original.shape[1]

    # def clear_dictionary(self, dictionary, coefficient_matrix, data):
    #     ''' remove nearly identical atoms from the dictionary '''
    #     t1      = 3
    #     t2      = 0.99
    #     K       = dictionary.shape[1]
    #     err     = np.sum((data - dictionary @ coefficient_matrix)**2, axis = 0)
    #     G       = dictionary.T @ dictionary
    #     G       = G - np.diag(np.diag(G))
    #     for j in range(K):
    #         if np.amax(G[j, :]) > t2 or len(coefficient_matrix[j, :] > 1e-7) <= t1:
    #             val              = np.amax(err)
    #             idx              = np.argmax(err)
    #             err[idx]         = 0
    #             dictionary[:, j] = data[:, idx] / np.linalg.norm(data[:, idx])
    #             G                = dictionary.T @ dictionary
    #             G                = G - np.diag(np.diag(G))

def omp(D, X, L):
    ''' Sparse coding based on given dictionary and number of columns to use
    D -> Dictionary
    X -> Image/Signals
    L -> Max number of columns/atoms for each signal

    output -> Sparse Coefficient Matrix
    '''
    n, P = X.shape
    n, K = D.shape
    A    = np.zeros((K, P))

    for k in range(P):
        a           = []
        x           = X[:, k]              # shape n, 1
        residual    = x
        idxs        = np.zeros((L)).astype(np.int)  

        for j in range(L):
            projection = D.T @ x            # shape K, 1
            max_idx    = np.argmax(projection)
            max_v      = np.amax(projection)
            idxs[j]    = max_idx
            cols       = D[:, idxs[:j+1]]   # shape n, j
            a          = np.linalg.pinv(cols) @ x # shape j, 1
            residual   = x - cols @ a
            if np.sum(residual) < 1e-6:
                break
        
        temp             = np.zeros((K))
        temp[idxs[:j+1]] = a
        A[:, k]          = temp
        
    return A


def update_dictionary(D, P, A):
    ''' 
    Updates columns of dictionary
    D -> Dictionary
    P -> Patch matrix, each column represents patch of the image
    A -> Sparse image/signal matrix, each column is sparse representation of the patch

    output -> Updated dictionary, Sparse signal representations


    Notes: 
        N_p = num patches
        n = num pixels in patch
        N = num pixels in image
        k = num atoms 

        D.shape = (n, K)
        P.shape = (n, N_p)
        A.shape = (K, N_p)
    '''

    n, K = D.shape
    n, N_p = P.shape

    # Update each column of D one at a time
    for k in range(K):
        nonzero_idxs = np.nonzero(A[k])[0]
        if len(nonzero_idxs) == 0:
            continue
        diff = P[:, nonzero_idxs] - (D @ A)[:, nonzero_idxs]
        res = diff + (D[:, k].reshape((-1, 1)) @ A[k, nonzero_idxs].reshape((1, -1)))
        
        # Rank 1 approximation
        u, s, v = svds(res, 1)
        D[:, k] = u.flatten()
        A[k, nonzero_idxs] = s * v.flatten()

    return D, A

    
def denoise(R, D, A, y, lam):
    ''' 
    Reconstructs image using dictionary and sparse representations
    R -> Array of matrices that select patches of image
    D -> Dictionary
    A -> Sparse image/signal matrix, each column is sparse representation of the patch
    y -> noisy, observed image
    lam -> regularization term that controls how well reconstruction should match observed image

    output -> reconstructed image


    Notes: 
        N_p = num patches
        n = num pixels in patch
        N = num pixels in image
        k = num atoms 

        R.shape = (N_p, n, N)
        D.shape = (n, k)
        A.shape = (k, N_p)
        y.shape = (N,1)
        lam.shape = (1,)
    '''

    N_p = R.shape[0]
    
    A = np.expand_dims(A.transpose(1,0), -1)        # shape (N_p, k, 1)
    D = np.expand_dims(D, 0) .repeat(N_p, axis=0)    # shape (N_p, n, k)
    Rt = R.transpose(0,2,1)                     # shape = (N_p, N, n)
    
    mat_inv = np.diag(np.reciprocal(np.diag(lam + np.sum(Rt @ R, axis=0))))
    x_hat = mat_inv  @ (lam + np.sum(Rt @ D @ A, axis=0))

    return x_hat







