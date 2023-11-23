import pennylane as qml
from pennylane import numpy as np
from pennylane import ClassicalShadow, classical_shadow, expval
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
num_snapshots = 1000

neighbors = find_neighbors(lattice_size,periodic)
hamiltonian = construct_hamiltonian(J,b,lattice_size,periodic,neighbors)
#opt_params, curr_energy = VQE(num_qubits, hamiltonian, neighbors)
#print(curr_energy,opt_params)


#optimal parameters and current energy were found using VQE
opt_params = [-4.18276290e-07, -6.13030748e-08,  2.52708489e-01, -7.77878068e-08,
       -2.99356307e-09,  2.52721373e-01,  1.02800675e-02, -3.27243045e-08,
        2.53353876e-01, -2.92291622e-08, -9.68577237e-07,  2.52721884e-01,
        5.51256128e-08, -7.58435451e-08,  2.52734054e-01,  1.03054950e-02,
       -4.20498697e-07,  2.53958793e-01,  1.02802252e-02, -2.44150501e-07,
        2.53353775e-01,  1.03059920e-02, -2.80372170e-07,  2.53960352e-01,
        1.05437284e-01, -1.44041963e-06,  1.48563628e-01] 

VQE_energy = -19.126286643211333 #<-- energy found using VQE
exact_energy = -19.131369


dev = qml.device("default.qubit", wires=num_qubits, shots=1)
@qml.qnode(dev)
def circuit_template_1_level(params, neighbors, obs=None): 
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



def calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits):
    """
    Using a randomly generated measurement basis measures the groundstate
    The outcomes matrix describes outcome of the measurement
    The unitary_ids matrix indexes the measurement applied to each qubit
    """
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits)) #generate random measurement bases

    outcomes = np.zeros((shadow_size, num_qubits))

    for num_shadows in range(shadow_size):
        obs = [unitary_ensemble[int(unitary_ids[num_shadows,i])](i) for i in range(num_qubits)]
        outcomes[num_shadows, :] = circuit_template(params, neighbors, obs=obs)
    
    return (outcomes, unitary_ids)

def run_calculate_classical_shadow(num_snapshots):
    params = opt_params

    shadow = calculate_classical_shadow(
        circuit_template_1_level, params, num_snapshots, num_qubits
        )
    return shadow

tau = 18.5
im_time_evo_op = expm(-tau*qml.matrix(hamiltonian))

def snapshot_state(b_list, obs_list):
    num_qubits = len(b_list)

    zero = np.array([[1,0]]).T
    one = np.array([[0,1]]).T
    phase_z = np.array([[1,0],[0,-1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))

    unitaries = [hadamard, hadamard@phase_z, identity] #inverses of the measurement basis each qubit is measured in
    state_vec = np.array([[1]])
    U = np.array([[1]])

    for i in range(num_qubits):
        state = zero if b_list[i] == 1 else one
        unitary = unitaries[obs_list[i]]
        state_vec = np.kron(state_vec,state)
        U = np.kron(U,unitary)

    #state_vec has shape (512,1) nd U has shape (512,512)
    #First invert the measurements to get classical snapshot state U^dagger@|b>
    #Then apply imaginary time evolution
    
    shadow_state_vec = U.conj().T@state_vec 
    time_ev_state = im_time_evo_op.T@shadow_state_vec 
    time_ev_state = time_ev_state / np.sqrt(time_ev_state.conj().T@time_ev_state)
    
    # I'm not sure if this is correct but instead of tensoring 3*U^dagger|b_i><b_i|U - I
    # together for i in range [1,9] to get the density matrix I first get the 
    # (512,1) state vector then apply time evolution because e^(-tau*H) is (512,512)
    # and after doing time evolution I then try to reconstruct the density matrix

    rho_snapshot = 9**num_qubits * (time_ev_state@time_ev_state.conj().T) - np.eye(512) 
    
    return rho_snapshot


def shadow_state_reconstruction(shadow):
    num_snapshots, num_qubits = shadow[0].shape
    b_lists, obs_lists = shadow

    shadow_rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])#each snapshot state has dim (512,512)
    return shadow_rho / num_snapshots



snapshots = calculate_classical_shadow(
        circuit_template_1_level, opt_params, num_snapshots, num_qubits
        )
density_matrix = shadow_state_reconstruction(snapshots)
print(f'The reconstructed density matrix has dimensions {density_matrix.shape}')
ham = qml.matrix(hamiltonian)
density_matrix = density_matrix / np.trace(density_matrix)
energy = np.trace(density_matrix@ham)
#time_ev_energy = np.trace(im_time_evolved_state@ham)
print(f'Our classical shadow representation of the groundstate gives E = {energy}\n\n')
print(f'The error is {np.sqrt((energy-exact_energy)**2)}')




    






