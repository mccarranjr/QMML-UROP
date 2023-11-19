import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
import argparse

from functions import (
    find_neighbors, construct_hamiltonian,
    ising_variational_circuit
)


parser = argparse.ArgumentParser(description = 'Classical Shadow: Expectation of the 2-D Transverse Field Ising Model Hamiltonian')
parser.add_argument('-J', type = float, help = 'Coupling strength')
parser.add_argument('-b', type=float, help= 'Transverse field strength')
parser.add_argument('-lattice_size', type = int, help = 'Size of our latice: lattice_size = n gives an (n x n) lattice')
parser.add_argument('-periodic', type = int, help = 'Are we implementing periodic interaction: 1 if yes, 0 if no')

args = parser.parse_args()

J = args.J
b = args.b
lattice_size = args.lattice_size
is_periodic = lambda periodic: periodic == 1
periodic = is_periodic(args.periodic)

num_qubits = lattice_size**2
neighbors = find_neighbors(lattice_size,periodic)
ising_hamiltonian = construct_hamiltonian(J,b,lattice_size,periodic,neighbors)
#print(f'Hamiltonian:\n {ising_hamiltonian}')


def VQE(num_qubits, hamiltonian, neighbors):

    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def vqe(params, num_qubits):
        ising_variational_circuit(params, num_qubits, neighbors)
        return qml.expval(hamiltonian)

    # Initialize random parameters
    params = np.random.rand(3*num_qubits)

    # Run the VQE 
    result = minimize(lambda params: vqe(params, num_qubits), params, method="CG")


    optimized_params = result.x
    ground_state_energy = result.fun

    #print("Optimized Parameters:", optimized_params)
    #print("Minimum Energy:", ground_state_energy)
    return optimized_params, ground_state_energy
#print(VQE(num_qubits,ising_hamiltonian,neighbors))
