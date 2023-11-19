import numpy as np
from numpy import linalg as LA
from ncon import ncon

#following tutorial --> https://www.tensors.net/mps

#initialize tensors
d = 2
chi = 10
A = np.random.rand(chi, d, chi)
B = np.random.rand(chi, d, chi)
sAB = np.ones(chi)/np.sqrt(chi) #setting trivial initial weights
sBA = np.ones(chi)/np.sqrt(chi)


#finding environment tensors
#contracting infinite MPS from the left for environment tensor sigBA
sigBA = np.random.rand(chi,chi) #init random starting point
tol = 1e-10

#define tensor network
tensors = [np.diag(sBA), np.diag(sBA), A, A.conj(), 
           np.diag(sAB), np.diag(sAB),B,B.conj()]
labels = [[1,2],[1,3],[2,4],[3,5,6],[4,5,7],[6,8],[7,9],[8,10,-1],[9,10,-2]]

for k in range(1000):
    sigBA_new = ncon([sigBA,*tensors],labels)#contract transfer operator
    sigBA_new = sigBA_new / np.trace(sigBA_new)
    

