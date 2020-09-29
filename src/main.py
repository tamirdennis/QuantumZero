"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
import time

# from src.trainer import Trainer
# from src.policy import HillClimbingPolicy
# from src.replay_memory import ReplayMemory
# from src.hill_climbing_env import HillClimbingEnv
from src.mcts import MCTS, execute_regular_mcts_episode, MCTSNode
from src.quantum_annealer_env import QuantumAnnealerEnv
from src.config import DATA_PATH, N_QUBITS, T_LIST, NUM_PROBLEMS_PER_T, M, l, DELTA, N_EXP, N_SIM,\
    NUM_EPISODES_PER_PROBLEM
import numpy as np
from src.sat_hamiltonyans_creator import create_sat_hamiltonians
import csv
import os


def log(test_env, iteration, step_idx, total_rew):
    """
    Logs one step in a testing episode.
    :param test_env: Test environment that should be rendered.
    :param iteration: Number of training iterations so far.
    :param step_idx: Index of the step in the episode.
    :param total_rew: Total reward collected so far.
    """
    time.sleep(0.3)
    print()
    print(f"Training Episodes: {iteration}")
    test_env.render()
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")


if __name__ == '__main__':

    data = np.loadtxt(DATA_PATH)
    run_statistics_cols = ['T', 'Problem Index', 'Fidelity', 'Time To Solve', 'Num Episodes Per Problem']
    os.makedirs('output', exist_ok=True)
    output_csv_path = os.path.join('output', 'run_statistics_200-80.csv')
    with open(output_csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(run_statistics_cols)
    run_statistics_rows = []
    for final_t in T_LIST:
        for i in range(NUM_PROBLEMS_PER_T):
            start_time = time.time()
            H0, H_final = create_sat_hamiltonians(N_QUBITS, data, i)
            QuantumAnnealerEnv.init_env_from_params(H0, H_final, final_t, num_x_components=M, l=l, delta=DELTA)
            n_actions = QuantumAnnealerEnv.n_actions
            mcts = MCTS(QuantumAnnealerEnv)
            mcts.initialize_search()
            best_merit = float('inf')
            best_node_path: MCTSNode = None
            for j in range(NUM_EPISODES_PER_PROBLEM):

                curr_best_node, best_merit = execute_regular_mcts_episode(mcts, num_expansion=N_EXP,
                                                                          num_simulations=N_SIM,
                                                                          best_merit=best_merit)
                best_node_path = best_node_path if curr_best_node is None else curr_best_node

                if j % 10 == 0 and j != 0 and j != NUM_EPISODES_PER_PROBLEM - 1 and best_node_path is not None:
                    print(f'after {j} episodes of mcts the best fidelity is:')
                    print(best_node_path.get_fidelity_of_node())
            new_row = [final_t, i, best_node_path.get_fidelity_of_node(), time.time() - start_time,
                       NUM_EPISODES_PER_PROBLEM]
            run_statistics_rows.append(new_row)
            print(f'Adding following row to csv: \n {run_statistics_cols}')
            print(run_statistics_rows[-1])
            with open(output_csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(new_row)


