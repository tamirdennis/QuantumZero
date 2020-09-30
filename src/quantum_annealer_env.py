import numpy as np
import gym

from src.static_env import StaticEnv
from qutip import *


class QuantumAnnealerEnv(gym.Env, StaticEnv):
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
    H_final = None
    H = None
    psi0 = None
    M = None
    l = None
    delta = None
    n_actions = None
    t = None
    final_t = None

    def __init__(self):
        pass

    def reset(self):
        return 0, 0, False, None

    def step(self, action):
        return 0, 0, False, None

    def render(self, mode='human'):
        pass

    @staticmethod
    def H_final_coeff(t, args):
        sum_over_sin_xi = 0
        for i in range(1, args['M'] + 1):
            sum_over_sin_xi += args[str(i)] * np.sin(i * np.pi * t / QuantumAnnealerEnv.final_t)
        return t / QuantumAnnealerEnv.final_t + sum_over_sin_xi

    @staticmethod
    def H0_coeff(t, args):
        return 1 - QuantumAnnealerEnv.H_final_coeff(t, args)

    @staticmethod
    def init_env_from_params(H0, H_final, final_t, num_x_components, l, delta):

        QuantumAnnealerEnv.final_t = final_t
        QuantumAnnealerEnv.H_final = H_final
        QuantumAnnealerEnv.H = [[H0, QuantumAnnealerEnv.H0_coeff], [QuantumAnnealerEnv.H_final, QuantumAnnealerEnv.H_final_coeff]]
        QuantumAnnealerEnv.t = np.array(list(range(1, final_t + 1)))
        QuantumAnnealerEnv.psi0 = H0.eigenstates()[1][0]
        QuantumAnnealerEnv.M = num_x_components
        QuantumAnnealerEnv.l = l
        QuantumAnnealerEnv.delta = delta
        QuantumAnnealerEnv.n_actions = int((2 * l / delta) + 1)

    @staticmethod
    def next_state(state, action):
        return state + [-QuantumAnnealerEnv.l + action * QuantumAnnealerEnv.delta]

    @staticmethod
    def is_done_state(state, depth=None):
        return len(state) == QuantumAnnealerEnv.M

    @staticmethod
    def initial_state():
        return []

    @staticmethod
    def get_obs_for_states(some_states):
        return None

    @staticmethod
    def get_rho_final_of_done_state(state):
        assert len(state) == QuantumAnnealerEnv.M
        args = {'M': QuantumAnnealerEnv.M}
        for i, x_i_value in enumerate(state):
            args[str(i + 1)] = x_i_value
        rho = qutip.mesolve(QuantumAnnealerEnv.H, QuantumAnnealerEnv.psi0, QuantumAnnealerEnv.t, args=args)
        rho_final = rho.states[-1]
        return rho_final

    @staticmethod
    def get_rho_lowest_energy_eigenstate_of_H_final():
        _, evecs = QuantumAnnealerEnv.H_final.eigenstates()
        return evecs[0]

    @staticmethod
    def get_return(state, step_idx):
        if step_idx != QuantumAnnealerEnv.M:
            return 0
        rho_final = QuantumAnnealerEnv.get_rho_final_of_done_state(state)
        return -abs((rho_final.trans() * QuantumAnnealerEnv.H_final * rho_final)[0][0][0])


if __name__ == '__main__':
    pass
