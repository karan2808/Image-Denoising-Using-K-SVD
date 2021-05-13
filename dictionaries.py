import numpy as np 

def idctii_dict(n, K):
    '''
    Build an overcomplete dictionary
    '''
    H = np.zeros((K, n))
    for i in range(n):
        ip       = np.zeros(n)
        ip[i]    = 1.0
        y        = np.array([sum(np.multiply(ip, np.cos((0.5 + np.arange(n)) * k * np.pi / K))) for k in range(K)])
        y[0]     = (1 / (2**(1/2))) * y[0]
        y        = (2 / n)**(1/2) * y
        H[:, i]  = y
    # H     = H.T
    dict_ = np.kron(H.T, H.T)
    return dict_


def get_dictionary(n, K):
    if n > K:
        print("n has to be smaller than K")
        return
    # get unitary or overcomplete dictionary depending on values of n, K 
    dictionary = idctii_dict(n, K)
    return dictionary


