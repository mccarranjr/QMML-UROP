import numpy as np
from numpy import linalg as LA
from ncon import ncon

#initialize tensors
d = 2
chi = 10
A = np.random.rand(chi, d, chi)
B = np.random.rand(chi, d, chi)
sAB = np.ones(chi)/np.sqrt(chi) #setting trivial initial weights
sBA = np.ones(chi)/np.sqrt(chi)


#finding environment tensors
