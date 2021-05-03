import numpy as np 
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy.matlib

class KSVD:
    def __init__(self, rank, sparsity, max_iterations, max_tolerance):
        self.rank           = rank 
        self.sparsity       = sparsity
        self.max_iterations = max_iterations
        self.max_tolerance  = max_tolerance
    
    def find_distance_between_dictionaries(self, original, new):
        catch_counter = 0
        total_dist    = 0
        for i in range(new.shape[1]):
            new[:, i] = np.sign(new[0, i]) * new[:, i]
        
        for i in range(original.shape[1]):
            d = np.sign(original[0, i]) * original[:, i]
            distances = np.sum((new - np.matlib.repmat(d, 1, new.shape[1]))**2, axis = 0)
            minval    = np.amin(distances)
            idx       = np.argmin(distances)
            err       = 1 - np.absolute(new[:, idx].T @ d)
            total_dist    = total_dist + err
            catch_counter = catch_counter + np.sum(err < 0.01)
        ratio = 100 * catch_counter/original.shape[1]

    def clear_dictionary(self, dictionary, coefficient_matrix, data):
        ''' remove nearly identical atoms from the dictionary '''
        t1      = 3
        t2      = 0.99
        K       = dictionary.shape[1]
        err     = np.sum((data - dictionary @ coefficient_matrix)**2, axis = 0)
        G       = dictionary.T @ dictionary
        G       = G - np.diag(np.diag(G))
        for j in range(K):
            if np.amax(G[j, :]) > t2 or len(coefficient_matrix[j, :] > 1e-7) <= t1:
                val              = np.amax(err)
                idx              = np.argmax(err)
                err[idx]         = 0
                dictionary[:, j] = data[:, idx] / np.linalg.norm(data[:, idx])
                G                = dictionary.T @ dictionary
                G                = G - np.diag(np.diag(G))

    def omp(self, D, X, L):
        ''' Sparse coding based on given dictionary and number of columns to use
        D -> Dictionary
        X -> Image/Signals
        L -> Max number of columns/atoms for each signal

        output -> Sparse Coefficient Matrix
        '''

        n, P = X.shape
        n, K = D.shape
        A    = np.zeros((n, P))
        for k in range(P):
            a           = []
            x           = X[:, k]               # shape n, 1
            residual    = x
            idxs        = np.zeros((L, 1))  

            for j in range(L):
                projection = D.T @ x            # shape K, 1
                max_idx    = np.argmax(projection)
                max_v      = np.amax(projection)
                idxs[j]    = max_idx
                cols       = D[:, idxs[:j+1]]   # shape n, j
                a          = np.linalg.pinv(cols) @ x # shape j, 1
                residual   = x - cols @ a
                if np.sum(residual ** 2) < 1e-6:
                    break
            
            temp             = np.zeros((K, 1))
            temp(idxs[:j+1]) = a
            A[:, k]          = temp

        return A