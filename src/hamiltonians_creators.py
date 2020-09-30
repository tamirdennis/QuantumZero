import numpy as np
from qutip import Qobj


def create_sat_hamiltonians(n_qubit, data, problem_row):

    resultc = data.tolist()

    result = [int(x) for x in resultc[problem_row]]
    N = 2 ** n_qubit

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

    HC = np.zeros((N, N))
    for i in result:
        HC[i, i] = HC[i, i] + 1
    HP = HC

    return Qobj(HB), Qobj(HP)


def get_grover_Hf(num_of_qubits):
    N = 2**num_of_qubits
    ind2find = np.random.randint(N)
    H_f = np.diag(np.ones(N))
    H_f[ind2find, ind2find] = 0
    return H_f

