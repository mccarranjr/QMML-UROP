import argparse
import pennylane as qml
import pennylane.numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools as it
import networkx as nx
from pennylane import classical_shadow, shadow_expval, ClassicalShadow


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

neighbors = find_neighbors(lattice_size,periodic)
hamiltonian = construct_hamiltonian(J,b,lattice_size,periodic)
obs = hamiltonian.ops
def rmsd(x,y):
    return np.sqrt(np.mean((x-y)**2))

groups = qml.pauli.group_observables(obs.ops)
n_wires = lattice_size**2
n_groups = len(groups)
def circuit(params, num_qubits, neighbors):
    """
    Variational circuit taken from Pennylane tutorial
    """
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)
    for i in range(num_qubits):
        qml.RX(params[i],wires=i)
    for i in range(num_qubits-1):
        for j in neighbors[i]:
            qml.CNOT(wires=[i, j])
#def circuit(params, num_qubits):
 #   for i in range(num_qubits):
  #      qml.RY(params[i], wires=i)
   # for i in range(num_qubits - 1):
    #    qml.CNOT(wires=[i, i + 1])



res_exact = -18.123105625608858
params = [-2.24740009e-07,  4.56700313e-07, -5.65206808e-07, -2.97388235e-07,
 -1.18113064e-07, -2.45850078e-07, -1.29999176e-07,  3.47617272e-08,
 -3.26060921e-07,  2.44977754e-01,  2.47688093e-01,  8.45451923e-01,
  7.28891589e-01,  7.25971502e-01,  2.35832127e-01,  5.74805121e-01,
  4.47693615e-01,  2.72235609e-01,]
#res_exact = -4.192258972413278
#res_exact = -2.74
#res_exact = -7.177314337238802
#params = [ 1.57078873e+00,  3.50510566e-01,  9.51288946e-02,  3.24008308e-02, -1.25566502e-07, -6.27872040e-06,  5.59337222e-02,  8.35758168e-01, 9.02273407e-02]
#params = [1.57079841e+00, 3.79251325e-01, 1.89625300e-01, 4.14761445e-06]
d_qwc = []
d_sha = []

shotss = np.arange(20,220,10)

for shots in shotss:
    for _ in range(10):
        dev_finite = qml.device("default.qubit",wires=range(n_wires),shots=int(shots))
        @qml.qnode(dev_finite,interface="autograd")
        def qnode_finite(H):
            circuit(params,n_wires)
            return qml.expval(H)

        with qml.Tracker(dev_finite) as tracker_finite:
            res_finite = qnode_finite(hamiltonian)

        dev_shadow = qml.device("default.qubit",wires=range(n_wires),shots=int(shots)*n_groups)
        @qml.qnode(dev_shadow,interface="autograd")
        def qnode():
            circuit(params,n_wires)
            return classical_shadow(wires=range(n_wires))

        with qml.Tracker(dev_shadow) as tracker_shadows:
            bits, recipes = qnode()

            shadow = ClassicalShadow(bits,recipes)
            res_shadow = shadow.expval(hamiltonian,k=1)

            #assert tracker_finite.totals["shots"] >= tracker_shadows.totals["shots"]

            d_qwc.append(rmsd(res_finite,res_exact))
            d_sha.append(rmsd(res_shadow,res_exact))


dq = np.array(d_qwc).reshape(len(shotss), 10)
dq, ddq = np.mean(dq, axis=1), np.var(dq, axis=1)
ds = np.array(d_sha).reshape(len(shotss), 10)
ds, dds = np.mean(ds, axis=1), np.var(ds, axis=1)
plt.errorbar(shotss*n_groups, ds, yerr=dds, fmt="x-", label="shadow")
plt.errorbar(shotss*n_groups, dq, yerr=ddq, fmt="x-", label="qwc")
plt.title("Figure 3")
plt.xlabel("total number of shots T", fontsize=20)
plt.ylabel("Error (RMSD)", fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

