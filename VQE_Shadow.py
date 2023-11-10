import argparse
import pennylane as qml
import pennylane.numpy as np
from matplotlib import pyplot as plt
from VQE_practice import VQE, find_neighbors, construct_hamiltonian
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

#taken from VQE_practice code
neighbors = find_neighbors(lattice_size,periodic)
hamiltonian = construct_hamiltonian(J,b,lattice_size,periodic,neighbors)

obs = hamiltonian.ops #Return the operators defining the Hamiltonian.
groups = qml.pauli.group_observables(obs) #partitions observables into qubit-wise commuting, 
n_groups = len(groups)                    #fully commuting, or anti-commuting

print(f"our hamiltonian is \n {hamiltonian}\n")
print(f"number of ops in H: {len(obs)}, number of qwc groups: {n_groups}\n")
print(f"Each group has sizes {[len(_) for _ in groups]}\n")

def rmsd(x,y):
    return np.sqrt(np.mean((x-y)**2))

def circuit(params, num_qubits, neighbors):
    """
    Variational circuit taken from IBM's Efficient SU(2) documentation
    """
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)
        qml.RX(params[3*i+1],wires=i)

        for j in neighbors[i]:
            qml.CNOT(wires=[i, j])

        qml.RY(params[3*i+2], wires=i)

optimal_params, res_exact = VQE(num_qubits, hamiltonian, neighbors) #VQE taken from VQE_practice code
print(f"The groundstate energy for this hamiltonian is {res_exact}\n")

d_qwc = []
d_sha = []

shotss = np.arange(20,2000,100)

for shots in shotss:
    for _ in range(10):
        
        dev_finite = qml.device("default.qubit",wires=range(num_qubits),shots=int(shots))
        @qml.qnode(dev_finite,interface="autograd")
        def qnode_finite(hamiltonian):
            circuit(optimal_params,num_qubits,neighbors)
            return qml.expval(hamiltonian)

        with qml.Tracker(dev_finite) as tracker_finite: #lets us store number of device executions 
            res_finite = qnode_finite(hamiltonian)

        dev_shadow = qml.device("default.qubit",wires=range(num_qubits),shots=int(shots)*n_groups)
        @qml.qnode(dev_shadow,interface="autograd")
        def qnode(hamiltonian):
            circuit(optimal_params,num_qubits,neighbors)
            return qml.shadow_expval(hamiltonian)#classical_shadow(wires=range(num_qubits))

        with qml.Tracker(dev_shadow) as tracker_shadows:
            res_shadow = qnode(hamiltonian) 

            #could also have done classical shadow with the code below but this is not differentiable:
            #bits, recipes = qnode()
            #shadow = ClassicalShadow(bits,recipes)
            #res_shadow = shadow.expval(hamiltonian,k=1)

            assert tracker_finite.totals["shots"] >= tracker_shadows.totals["shots"]

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
plt.ylim(0,3)
plt.show()

