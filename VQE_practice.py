import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
import argparse


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

def find_neighbors(lattice_size,periodic=False):
    """
    takes in a lattice size and returns a dictionary whose keys are sites
    and whose values are all adjacent neighboring sites
    """
    all_neighbors = {i:set() for i in range(lattice_size**2)}
    is_a_valid_site = lambda x,y: x>=0 and x<=lattice_size-1 and y>=0 and y<=lattice_size-1 #makes sure a site is in the square lattice


    for i in range(lattice_size):
        for j in range(lattice_size):

            site = i*lattice_size + j
            neighbors = [[i+1,j],[i-1,j],[i,j+1],[i,j-1]]

            for x,y in neighbors: 
                if is_a_valid_site(x,y): #make sure neighbors are in the lattice
                    neighbor_site = x*lattice_size + y
                    if neighbor_site > site:
                        all_neighbors[site].add(neighbor_site)
            print(periodic,'periodic val')
            if periodic: #for periodic boundary conditions
                if i == 0:
                    neighbor_site = lattice_size**2-lattice_size+site
                    all_neighbors[site].add(neighbor_site)

                if j == 0:
                    neighbor_site = site+lattice_size-1
                    all_neighbors[site].add(neighbor_site)

        # currently the hamiltonian has no repeats ex: if we have Z0Z1 we don't also include Z1Z0
        # if we want repeats we for periodic boundaries we can uncomment the code below 
                        
               #if i == lattice_size-1:
                    #neighbor_site = j
                    #all_neighbors[site].append(neighbor_site)

                #if j == lattice_size-1:
                    #neighbor_site = site-lattice_size+1
                    #all_neighbors[site].append(neighbor_site)
    return all_neighbors

def construct_hamiltonian(J,b,lattice_size,periodic):
    """
    Given the coupling strength term and the magnetic field strength
    returns the 2-D transverse field ising model hamiltonian for a 
    square lattice
    """
    neighbors = find_neighbors(lattice_size,periodic)
    hamiltonian_terms = []
    coeffs = []

    for i in range(lattice_size**2):
        coeffs.append(-b) 
        hamiltonian_terms.append(qml.PauliX(i))
        for j in neighbors[i]:

            coeffs.append(-J) 
            hamiltonian_terms.append(qml.PauliZ(i)@qml.PauliZ(j))
            hamiltonian_terms = list(set(hamiltonian_terms))

    hamiltonian = qml.Hamiltonian(coeffs,hamiltonian_terms)
    return hamiltonian


ising_hamiltonian = construct_hamiltonian(J,b,lattice_size,periodic)
print(f'Hamiltonian:\n {ising_hamiltonian}')


def variational_rot_circuit(params, wires=num_qubits-1):
    """
    Variational circuit taken from Pennylane tutorial
    """
    for i in range(num_qubits):
        qml.RX(params[i], wires=i)
        qml.RY(params[i+1], wires=i)

def variational_circuit(params, num_qubits):
    """
    Variational circuit taken from Pennylane tutorial
    """
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)
    for i in range(num_qubits):
        qml.RX(params[i],wires=i)
    for i in range(num_qubits-1):
        qml.CNOT(wires=[i, i + 1])



dev = qml.device("default.qubit", wires=num_qubits)
@qml.qnode(dev)
def vqe(params, h, J, num_qubits):
    variational_circuit(params, num_qubits)
    return qml.expval(ising_hamiltonian)

# Initialize random parameters
params = np.random.rand(2*num_qubits)


# Run the VQE 
result = minimize(lambda params: vqe(params, b, J, num_qubits), params, method="CG")


optimized_params = result.x
ground_state_energy = result.fun

print("Optimized Parameters:", optimized_params)
print("Minimum Energy:", ground_state_energy)
