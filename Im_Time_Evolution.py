import pennylane as qml
from pennylane import numpy as np
from pennylane import ClassicalShadow, classical_shadow
from functions import (
    find_neighbors, construct_hamiltonian, ising_variational_circuit,
    ising_2_level_variational_circuit, construct_full_hamiltonian 
    )
from VQE_practice import VQE
from scipy.linalg import expm
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

neighbors = find_neighbors(lattice_size,periodic)
hamiltonian = construct_hamiltonian(J,b,lattice_size,periodic,neighbors)
#opt_params, curr_energy = VQE(num_qubits, hamiltonian, neighbors)
#print(curr_energy,opt_params)
opt_params = [-4.18276290e-07, -6.13030748e-08,  2.52708489e-01, -7.77878068e-08,
       -2.99356307e-09,  2.52721373e-01,  1.02800675e-02, -3.27243045e-08,
        2.53353876e-01, -2.92291622e-08, -9.68577237e-07,  2.52721884e-01,
        5.51256128e-08, -7.58435451e-08,  2.52734054e-01,  1.03054950e-02,
       -4.20498697e-07,  2.53958793e-01,  1.02802252e-02, -2.44150501e-07,
        2.53353775e-01,  1.03059920e-02, -2.80372170e-07,  2.53960352e-01,
        1.05437284e-01, -1.44041963e-06,  1.48563628e-01] 

curr_energy = -19.126286643211333

dev = qml.device("lightning.qubit", wires=num_qubits, shots=1)
@qml.qnode(dev)
def circuit_template_1_level(params, neighbors, obs=None):
    for i in range(num_qubits):
        qml.RY(params[3*i], wires=i)
        qml.RX(params[3*i+1],wires=i)
        for j in neighbors[i]:
            qml.CNOT(wires=[i, j])
        qml.RY(params[3*i+2],wires=i)
    return [qml.expval(o) for o in obs]


num_snapshots = 1000
def calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits):
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))

    outcomes = np.zeros((shadow_size, num_qubits))

    for ns in range(shadow_size):
        obs = [unitary_ensemble[int(unitary_ids[ns,i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = circuit_template(params, neighbors, obs=obs)
    
    return (outcomes, unitary_ids)

def collect_shadow(num_snapshots):
    params = opt_params

    shadow = calculate_classical_shadow(
        circuit_template_1_level, params, num_snapshots, num_qubits
        )
    return shadow
#print(collect_shadow(num_snapshots)[0])

tau = 0
im_time_evo_op = expm(-tau*qml.matrix(hamiltonian))

def snapshot_state(b_list, obs_list):
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
    return rho_snapshot

def shadow_state_reconstruction(shadow):
    num_snapshots, num_qubits = shadow[0].shape
    b_lists, obs_lists = shadow

    shadow_rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])
        #shadow_rho = snapshot_state(b_lists[i], obs_lists[i])

    return shadow_rho / num_snapshots

out = []

def average():
    for i in range(10):
        shadow_state = shadow_state_reconstruction(collect_shadow(num_snapshots))
        ham = qml.matrix(hamiltonian)
        im_time_evolved_state = im_time_evo_op@shadow_state#@im_time_evo_op.T
        im_time_evolved_state = im_time_evolved_state/np.sqrt(im_time_evolved_state.T@im_time_evolved_state)
        time_ev_energy = np.trace(im_time_evolved_state@ham)
        out.append(time_ev_energy)
    avg = sum(out)/10
    print(f'avg = {avg}')
    np_out = np.array([out])
    print(f'mean = {np.mean(np_out)} /n std = {np.std(np_out)}')




shadow_state = shadow_state_reconstruction(collect_shadow(num_snapshots))
#print(shadow_state)
print(f'The reconstructed density matrix has dimensions {shadow_state.shape}')
ham = qml.matrix(hamiltonian)
im_time_evolved_state = im_time_evo_op@shadow_state@im_time_evo_op.T
im_time_evolved_state /= np.sqrt(im_time_evolved_state.T@im_time_evolved_state)
shadow_energy = np.trace(shadow_state@ham)
time_ev_energy = np.trace(im_time_evolved_state@ham)
print(f'Our classical shadow representation of the groundstate gives E = {time_ev_energy}\n\n')
print(f'The error is {time_ev_energy-curr_energy}')




    






