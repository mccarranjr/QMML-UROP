import pennylane as qml
from pennylane import numpy as np
from functions import (
    find_neighbors, construct_hamiltonian 
    )
from VQE_practice import VQE
from scipy.linalg import expm
import argparse
from matplotlib import pyplot as plt

np.random.seed(10)

parser = argparse.ArgumentParser(description = 'Classical Shadow: Expectation of the 2-D Transverse Field Ising Model Hamiltonian')
parser.add_argument('-J', type = float, help = 'Coupling strength')
parser.add_argument('-b', type=float, help= 'Transverse field strength')
parser.add_argument('-lattice_size', type = int, help = 'Size of our latice: lattice_size = n gives an (n x n) lattice')
parser.add_argument('-periodic', type = int, help = 'Are we implementing periodic interaction: 1 if yes, 0 if no')
parser.add_argument('-num_snapshots', type = int, help = 'Number of snapshots used to create our classical shadow representation')
parser.add_argument('-tau', type = float, help = 'Imaginary time step for imaginary time evolution')
parser.add_argument('-time_steps', type = int, help = 'Number of time steps we apply the imaginary time evolution operator')
args = parser.parse_args()

J = args.J
b = args.b
lattice_size = args.lattice_size
is_periodic = lambda periodic: periodic == 1
periodic = is_periodic(args.periodic)
num_qubits = lattice_size**2
num_snapshots = args.num_snapshots
tau = args.tau
time_steps = args.time_steps

im_time_evo_op = lambda step: expm(-step*tau*qml.matrix(hamiltonian))

neighbors = find_neighbors(lattice_size,periodic)
hamiltonian = construct_hamiltonian(J,b,lattice_size,periodic,neighbors)
#opt_params, curr_energy = VQE(num_qubits, hamiltonian, neighbors)
#print(curr_energy,opt_params)


# for 3x3 lattice with periodice boundary conditions
# the optimal parameters and VQE_energy were found using VQE_practice.py
opt_params = [-4.18276290e-07, -6.13030748e-08,  2.52708489e-01, -7.77878068e-08,
       -2.99356307e-09,  2.52721373e-01,  1.02800675e-02, -3.27243045e-08,
        2.53353876e-01, -2.92291622e-08, -9.68577237e-07,  2.52721884e-01,
        5.51256128e-08, -7.58435451e-08,  2.52734054e-01,  1.03054950e-02,
       -4.20498697e-07,  2.53958793e-01,  1.02802252e-02, -2.44150501e-07,
        2.53353775e-01,  1.03059920e-02, -2.80372170e-07,  2.53960352e-01,
        1.05437284e-01, -1.44041963e-06,  1.48563628e-01] 
# VQE_energy = -19.126286643211333 #<-- energy found using VQE

exact_energy = -19.131369 # energy solved through exact diagonalization
                          # we are trying to get as close as possible to this energy

dev = qml.device("default.qubit", wires=num_qubits, shots=1)
@qml.qnode(dev)
def circuit_template_1_level(params = None, neighbors = None, obs=None): 
    """
    Single layer circuit taken from IBM efficient SU(2) documentation
    Returns list of expectation values for each observable passed in
    """
    for i in range(num_qubits):
        qml.RY(params[3*i], wires=i)
        qml.RX(params[3*i+1],wires=i)
        for j in neighbors[i]:
            qml.CNOT(wires=[i, j])
        qml.RY(params[3*i+2],wires=i)
    out = [qml.expval(o) for o in obs]
    return out



def calculate_classical_shadow(circuit_template = None, params = None,
                                shadow_size = None, num_qubits = None):
    """
    Given a circuit, creates a collection of snapshots consisting of a bit string
    and the index of a unitary operation.

    Args:
        circuit_template (function): A Pennylane QNode.
        params (array): Circuit parameters.
        shadow_size (int): The number of snapshots in the shadow.
        num_qubits (int): The number of qubits in the circuit.

    Returns:
        Tuple of two numpy arrays. The first array contains measurement outcomes (-1, 1)
        while the second array contains the index for the sampled Pauli's (0,1,2=X,Y,Z).
        Each row of the arrays corresponds to a distinct snapshot or sample while each
        column corresponds to a different qubit.
    """
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits)) #generate random measurement bases

    outcomes = np.zeros((shadow_size, num_qubits))

    for num_shadows in range(shadow_size): #pick random observalbes and calculate expectation values
        obs = [unitary_ensemble[int(unitary_ids[num_shadows,i])](i) for i in range(num_qubits)]
        outcomes[num_shadows, :] = circuit_template(params, neighbors, obs=obs)
    
    return (outcomes, unitary_ids)


def snapshot_state(b_list = None, obs_list = None, im_time_evo_op = None):
    """
    Helper function for `shadow_state_reconstruction` that reconstructs
     a state from a single snapshot in a shadow.

    Implements Eq. (S44) from https://arxiv.org/pdf/2002.08953.pdf

    Args:
        b_list (array): The list of classical outcomes for the snapshot.
        obs_list (array): Indices for the applied Pauli measurement.

    Returns:
        Numpy array with the reconstructed snapshot.
    """
    num_qubits = len(b_list)

    zero = np.array([[1,0]])
    one = np.array([[0,1]])
    zero_state = np.array([[1,0],[0,0]])
    one_state = np.array([[0,0],[0,1]])
    phase_z = np.array([[1,0],[0,-1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))

    unitaries = [hadamard, hadamard@phase_z, identity]
    #rhos = []
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]

        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return im_time_evo_op@rho_snapshot@im_time_evo_op.T

def shadow_state_reconstruction(shadow = None ,ITE_op = None):
    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.

    Args:
        shadow (tuple): A shadow tuple obtained from `calculate_classical_shadow`.

    Returns:
        Numpy array with the reconstructed quantum state.
    """
    num_snapshots, num_qubits = shadow[0].shape
    b_lists, obs_lists = shadow

    shadow_rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i], ITE_op)#each snapshot state has dim (512,512)
    return shadow_rho / np.trace(shadow_rho) #num_snapshots

def plot_energy_error(time_steps = None, exact_energy = None, hamiltonian = None):
    '''
    Iteratively applies the imaginary time evolution operator to see how 
    qiuckly the our classical shodow state converges to the exact groundstate
    wavefunvtion.
    
    Args:
        time_steps (int): How many times we are going to apply the time evolution operator
        exact_energy (float): The exact groundstate energy found through exact diagonalization
        hamiltonian (Pennylane Hamiltonian): Our systems hamiltonian
        
    Returns:
        Graph with energy on the y axis and number of steps on the x asis showing
        how quickly the energy converges to the exact energy
    '''
    out = []
    ham = qml.matrix(hamiltonian)

    for step in range(time_steps):
        shadow = calculate_classical_shadow(
                circuit_template_1_level, opt_params, num_snapshots, num_qubits
                )
        ITE_op = im_time_evo_op(step)
        density_matrix = shadow_state_reconstruction(shadow, ITE_op)
        energy = np.trace(density_matrix@ham)
        error = np.sqrt((energy-exact_energy)**2)
        out.append(error)

    plt.plot(range(time_steps), out)
    plt.xlabel('Time Steps')
    plt.ylabel('Energy Error')
    plt.title('Energy Error over Time')
    plt.show()

def single_application(circuit_template, params, shadow_size, num_qubits):

    shadow = calculate_classical_shadow( 
        circuit_template = circuit_template_1_level, params = opt_params, 
        shadow_size = num_snapshots, num_qubits = num_qubits
        )

    ITE_op = im_time_evo_op(1)
    density_matrix = shadow_state_reconstruction(shadow = shadow, ITE_op = ITE_op)
    ham = qml.matrix(hamiltonian)
    energy = np.trace(density_matrix@ham)
    print(f'\nThe reconstructed density matrix has dimensions {density_matrix.shape}\n')
    print(f'Our classical shadow representation of the groundstate gives groundstate energy E = {energy}\n')
    print(f'The error is {np.sqrt((energy-exact_energy)**2)}')
    return

print(single_application(
    circuit_template_1_level, opt_params, num_snapshots,num_qubits )
    )

#The energy I get is -19.13123051907408
#The exact energy is -19.131369

#to see how the error changes as the number of time steps increases uncomment the line below
#plot_energy_error(time_steps, exact_energy, hamiltonian)


    






