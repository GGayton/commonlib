import numpy as np

def corrMatrixFromCovMatrix(V):
    
    X = np.diag(V)**0.5
    X = X.reshape(-1,1)
    
    corr = V/(X@X.T)
    
    return corr
