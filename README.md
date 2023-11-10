# QMML-UROP
UROP Project Code

Tasks: Run VQE to find the exact groundstate energy for the 2-D Transverse Field Ising Model assuming square lattice structure. Then incorporate classical shadow method and compare the expected energy with the exact energy


For the VQE running on a 3x3 lattice (with J=b=1 and no periodic boundary conditions) our hamiltonian is 

(-1.0) [X1]
+ (-1.0) [X7]
+ (-1.0) [X2]
+ (-1.0) [X5]
+ (-1.0) [X0]
+ (-1.0) [X6]
+ (-1.0) [X4]
+ (-1.0) [X3]
+ (-1.0) [X8]
+ (-1.0) [Z1 Z2]
+ (-1.0) [Z4 Z5]
+ (-1.0) [Z3 Z4]
+ (-1.0) [Z6 Z7]
+ (-1.0) [Z4 Z7]
+ (-1.0) [Z1 Z4]
+ (-1.0) [Z0 Z3]
+ (-1.0) [Z3 Z6]
+ (-1.0) [Z0 Z1]
+ (-1.0) [Z7 Z8]
+ (-1.0) [Z2 Z5]
+ (-1.0) [Z5 Z8]

and the groundstate energy is -13.78135127969275

For the VQE running on a 3x3 lattice (with J=b=1 and periodic boundary conditions) our hamiltonian is

(-1.0) [X5]
+ (-1.0) [X3]
+ (-1.0) [X4]
+ (-1.0) [X1]
+ (-1.0) [X2]
+ (-1.0) [X7]
+ (-1.0) [X0]
+ (-1.0) [X6]
+ (-1.0) [X8]
+ (-1.0) [Z6 Z8]
+ (-1.0) [Z4 Z7]
+ (-1.0) [Z1 Z4]
+ (-1.0) [Z2 Z5]
+ (-1.0) [Z0 Z6]
+ (-1.0) [Z3 Z6]
+ (-1.0) [Z1 Z7]
+ (-1.0) [Z6 Z7]
+ (-1.0) [Z5 Z8]
+ (-1.0) [Z0 Z2]
+ (-1.0) [Z3 Z4]
+ (-1.0) [Z1 Z2]
+ (-1.0) [Z0 Z1]
+ (-1.0) [Z4 Z5]
+ (-1.0) [Z2 Z8]
+ (-1.0) [Z7 Z8]
+ (-1.0) [Z3 Z5]
+ (-1.0) [Z0 Z3]

and the groundstate energy is -19.12628664321259