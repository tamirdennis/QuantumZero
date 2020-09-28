import numpy as np
import gym

from src.static_env import StaticEnv
from qutip import *
from src.config import DATA_PATH, N_QUBITS, T

data = np.loadtxt(DATA_PATH)
resultc = data.tolist()

result = [int(x) for x in resultc[0]]


def satSystem(n_qubit, result):
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

    return HB, HP


H0, H_final = satSystem(N_QUBITS, result)
H0 = Qobj(H0)
H_final = Qobj(H_final)


def H_final_coeff(t, args):
    sum_over_sin_xi = 0
    for i in range(1, args['M'] + 1):
        sum_over_sin_xi += args[str(i)] * np.sin(i * np.pi * t / T)
    return t / T + sum_over_sin_xi


def H0_coeff(t, args):
    return 1 - H_final_coeff(t, args)


qutip.Qobj()
H = [[H0, H0_coeff], [H_final, H_final_coeff]]
t = np.array(list(range(T + 1)))
psi0 = basis(2 ** N_QUBITS, 0)


class SAT3Env(gym.Env, StaticEnv):
    """
    Simple gym environment with the goal to navigate the player from its
    starting position to the highest point on a two-dimensional map within
    a limited number of steps. Rewards are defined as the difference in
    altitude between states minus a penalty for each step. The player starts
    in the lower left corner of the map and the highest point is in the upper
    right corner. Map layout mirrors CliffWalking environment:
    top left = (0, 0), top right = (0, m-1), bottom left = (n-1, 0),
    bottom right = (n-1, m-1).
    The setup of this environment was inspired by the energy landscape in
    protein folding.
    """
    M = 5
    l = 0.2
    delta = 0.01
    n_actions = int((2 * l / delta) + 1)

    def __init__(self):
        pass

    def reset(self):
        return 0, 0, False, None

    def step(self, action):
        return 0, 0, False, None

    def render(self, mode='human'):
        pass

    @staticmethod
    def next_state(state, action):
        return state + [-SAT3Env.l + action * SAT3Env.delta]

    @staticmethod
    def is_done_state(state, depth=None):
        return len(state) == SAT3Env.M

    @staticmethod
    def initial_state():
        return []

    @staticmethod
    def get_obs_for_states(some_states):
        return None

    @staticmethod
    def get_rho_final_of_done_state(state):
        assert len(state) == SAT3Env.M
        args = {'M': SAT3Env.M}
        for i, x_i_value in enumerate(state):
            args[str(i + 1)] = x_i_value
        rho = qutip.mesolve(H, psi0, t, args=args)
        rho_final = rho.states[-1]
        return rho_final

    @staticmethod
    def get_rho_lowest_energy_eigenstate_of_H_final():
        _, evecs = H_final.eigenstates()
        return evecs[0]

    @staticmethod
    def get_return(state, step_idx):
        if step_idx != SAT3Env.M:
            return 0
        rho_final = SAT3Env.get_rho_final_of_done_state(state)
        return - abs((rho_final.trans() * H_final * rho_final)[0][0][0])


if __name__ == '__main__':
    pass
