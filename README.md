# QMML-UROP
Quantum Machine Learning UROP Project Code


Task 1: Run VQE to find the exact groundstate energy for the 2-D Transverse Field Ising Model with J=b=1 assuming square lattice structure and periodic boundary conditions. For all of the code I'm using a 3x3 square lattice so a 9 qubit system. 

Relevant file: VQE_practice.py <br>
Exact energy -> -19.131369<br>
Energy my VQE finds -> -19.126286643211333<br>

Task 2: VQE allows us to prepare the groundstate on a quantum circuit, now use the classical shadow method to reconstruct the groundstate density matrix and compare the expected energy with the exact energy.

Relevant file: classical_shadow_practice.py <br>
Exact energy -> -19.131369 <br>
Classical shadow energy -19.034999999999975-5.516420653606247e-16j<br>

Task 3: Use imaginary time evolution to improve our classical shadow groundstate energy.

Relevant file: ITE_Ising_Shadow<br>
Exact energy -> -19.131369 <br>
ITE energy -> -19.131360622675505<br>


