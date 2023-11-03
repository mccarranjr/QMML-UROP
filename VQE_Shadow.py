import argparse
import pennylane as qml
import pennylane.numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools as it
import networkx as nx
from pennylane import classical_shadow, shadow_expval, ClassicalShadow


#Hamiltonian is H = J*sum(Zi*Zj) - h*sum(Xi)
parser = argparse.ArgumentParser(description = 'Classical Shadow: Expectation of the 2-D Transverse Field Ising Model Hamiltonian')
parser.add_argument('-J', type = float, help = 'Coupling strength')
parser.add_argument('-b', type=float, help= 'Transverse field strength')
parser.add_argument('-lattice_size', type = int, help = 'Size of our latice: lattice_size = n gives an (n x n) lattice')
args = parser.parse_args()

X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
J, b, lattice_size = args.J, args.b, args.lattice_size

#spins = 2*np.random.randint(2, size=(lattice_size,lattice_size))-1
#spin_up_indices = np.where(spins.flatten() > 0)


def find_neighbors(lattice_size):
    """
    takes in a lattice size and returns a dictionary whose keys are sites
    and whose values are all adjacent neighboring sites
    """
    all_neighbors = {i:[] for i in range(lattice_size**2)}
    is_a_valid_site = lambda x,y: x>=0 and x<=lattice_size-1 and y>=0 and y<=lattice_size-1 #makes sure a site is in the square lattice


    for i in range(lattice_size):
        for j in range(lattice_size):

            site = i*lattice_size + j
            neighbors = [[i+1,j],[i-1,j],[i,j+1],[i,j-1]]

            for x,y in neighbors: 
                if is_a_valid_site(x,y): #make sure neighbors are in the lattice
                    all_neighbors[site].append(x*lattice_size + y)
    return all_neighbors

def construct_hamiltonian(J,b,lattice_size):
    """
    Given the coupling strength term and the magnetic field strength
    returns the 2-D transverse field ising model hamiltonian for a 
    square lattice
    """
    neighbors = find_neighbors(lattice_size)
    hamiltonian_terms = []
    coeffs = []

    for i in range(lattice_size):
        coeffs.append(-b) 
        hamiltonian_terms.append(qml.PauliX(i))
        for j in neighbors[i]:
            coeffs.append(-J) 
            hamiltonian_terms.append(qml.PauliZ(i)@qml.PauliZ(j))
        
    hamiltonian = qml.Hamiltonian(coeffs,hamiltonian_terms)
    return hamiltonian

hamiltonian = construct_hamiltonian(J,b,lattice_size)
all_neighbors = find_neighbors(lattice_size)
obs = hamiltonian
all_neighbors = find_neighbors(lattice_size)
print(hamiltonian)
def rmsd(x,y):
    return np.sqrt(np.mean((x-y)**2))

groups = qml.pauli.group_observables(obs.ops)
n_wires = lattice_size**2
n_groups = len(groups)
def circuit(params, num_qubits):
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])




#res_exact = -4.192258972413278
#res_exact = -2.74
res_exact = -7.177314337238802
params = [ 1.57078873e+00,  3.50510566e-01,  9.51288946e-02,  3.24008308e-02, -1.25566502e-07, -6.27872040e-06,  5.59337222e-02,  8.35758168e-01, 9.02273407e-02]
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

