import numpy as np
from qutip import Qobj
from scipy import sparse


def general_H0_creator(n_qubit):

    sx = np.array([[0, 1], [1, 0]])
    si = np.array([[1, 0], [0, 1]])

    HB = np.kron(si, si)
    for i in range(n_qubit - 2):
        HB = np.kron(si, HB)
    HB = n_qubit * HB / 2

    for j in range(n_qubit):
        if j == 0:
            B = sx
            for i in range(n_qubit - 1 - j):
                B = np.kron(si, B)
        else:
            for i in range(j):
                if i == 0:
                    B = si
                else:
                    B = np.kron(si, B)
            B = np.kron(sx, B)
            for i in range(n_qubit - 1 - j):
                B = np.kron(si, B)

        HB = HB - B / 2
    return Qobj(HB)


def create_sat_hamiltonians(n_qubit, data, problem_row):

    resultc = data.tolist()

    result = [int(x) for x in resultc[problem_row]]
    N = 2 ** n_qubit

    H0 = general_H0_creator(n_qubit)

    HC = np.zeros((N, N))
    for i in result:
        HC[i, i] = HC[i, i] + 1
    HP = HC

    return H0, Qobj(HP)


def create_grover_hamiltonians(num_of_qubits):
    N = 2**num_of_qubits
    ind2find = np.random.randint(N)
    H_f = np.diag(np.ones(N))
    H_f[ind2find, ind2find] = 0
    return general_H0_creator(num_of_qubits), Qobj(H_f)

# %% Globals
s_z = np.matrix([[1, 0], [0, -1]])
ss_z = sparse.csr_matrix(s_z)


# %% QUBO Functions
def operate_only_on_qubit_j(operation, num_of_qubits, j):
    if num_of_qubits == 1:
        return operation
    if num_of_qubits - 1 == j:
        return sparse.kron(sparse.eye(2), operate_only_on_qubit_j(operation, num_of_qubits - 1, j - 1))
    else:
        return sparse.kron(operate_only_on_qubit_j(operation, num_of_qubits - 1, j), sparse.eye(2))


def operate_only_on_qubit_j_diagonal(operation, num_of_qubits, j):
    operation_diag = operation.diagonal()
    N = 2 ** num_of_qubits
    K = 2 ** (j + 1)
    change_fac = N // K
    full_oper_diag = np.zeros(N)
    for i in range(N):
        index = i // change_fac % 2
        full_oper_diag[i] = operation_diag[index]
    full_oper = sparse.diags(full_oper_diag, format='csr')
    return full_oper_diag, full_oper


def QUBO_to_Hamiltonian(QUBO, sparse_type='csr'):
    num_of_qubits = np.size(QUBO, axis=0)
    N = 2 ** num_of_qubits
    qubo2IsingOper = (sparse.eye(2) + ss_z) / 2
    if sparse_type == 'csc':
        H_p = sparse.csc_matrix((N, N))
    else:
        H_p = sparse.csr_matrix((N, N))

    # sig_all

    for i in range(num_of_qubits):
        sig_i_diag, sig_i = operate_only_on_qubit_j_diagonal(qubo2IsingOper, num_of_qubits, i)
        for j in range(num_of_qubits):
            if i == j:
                H_p += QUBO[i, j] * sig_i
            else:
                sig_j_diag, _ = operate_only_on_qubit_j_diagonal(qubo2IsingOper, num_of_qubits, j)
                product_i_j_vec = sig_i_diag * sig_j_diag
                product_i_j = sparse.diags(product_i_j_vec, format='csr')

                H_p += QUBO[i, j] * product_i_j
    return H_p


def randomize_QUBO_small_gap(num_of_qubits):
    gap = 0.1
    norm_f = gap * 2
    gap_is_good = False
    while not gap_is_good:
        # QUBO_mat = norm_f*(np.random.randn(1)+1)*(np.random.randn(num_of_qubits, num_of_qubits))
        QUBO_mat = norm_f * (np.random.randn(1)) * (np.random.randn(num_of_qubits, num_of_qubits))
        QUBO_mat = np.tril(QUBO_mat) + np.tril(QUBO_mat, -1).T
        H_f = QUBO_to_Hamiltonian(QUBO_mat)
        wf = np.sort(H_f.diagonal())
        cur_gap = wf[1] - wf[0]
        gap_is_good = np.abs(cur_gap - gap) < 0.05

    return QUBO_mat


def get_random_QUBO_Hf(num_of_qubits):
    QUBO_mat = randomize_QUBO_small_gap(num_of_qubits)
    H_f_sparse = QUBO_to_Hamiltonian(QUBO_mat)
    H_final = Qobj(H_f_sparse.toarray())
    return general_H0_creator(num_of_qubits), H_final
