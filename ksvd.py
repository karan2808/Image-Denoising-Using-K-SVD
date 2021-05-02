import numpy as np 
from sklearn.linear_model import OrthogonalMatchingPursuit

class KSVD:
    def __init__(self, rank, sparsity, max_iterations, max_tolerance):
        self.rank           = rank 
        self.sparsity       = sparsity
        self.max_iterations = max_iterations
        self.max_tolerance  = max_tolerance

    
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