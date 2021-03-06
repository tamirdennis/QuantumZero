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
from src.config import DATA_PATH, N_QUBITS, T_LIST, PROBLEMS_INDEXES_LIST, M, l, DELTA, N_EXP, N_SIM,\
    NUM_EPISODES_PER_PROBLEM, MAX_t_POINTS
import numpy as np
from src.hamiltonians_creators import create_sat_hamiltonians, create_grover_hamiltonians, get_random_QUBO_Hf
import csv
import os


if __name__ == '__main__':

    data = np.loadtxt(DATA_PATH)
    run_statistics_cols = ['T', 'Problem Index', 'Fidelity', 'Time To Solve', 'Num Episodes Per Problem', 'best_X']
    os.makedirs('output', exist_ok=True)
    output_csv_path = os.path.join('output', 'run_statistics_grover.csv')
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(run_statistics_cols)
    run_statistics_rows = []
    for i in PROBLEMS_INDEXES_LIST:
        H0, H_final = create_grover_hamiltonians(N_QUBITS)
        for final_t in T_LIST:
            start_time = time.time()
            QuantumAnnealerEnv.init_env_from_params(H0, H_final, final_t, num_x_components=M, l=l, delta=DELTA,
                                                    max_t_points=MAX_t_POINTS)
            n_actions = QuantumAnnealerEnv.n_actions
            mcts = MCTS(QuantumAnnealerEnv)
            mcts.initialize_search()
            best_merit = -float('inf')
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
                       NUM_EPISODES_PER_PROBLEM, str(best_node_path.state)]
            run_statistics_rows.append(new_row)
            print(f'Adding following row to csv: \n {run_statistics_cols}')
            print(run_statistics_rows[-1])
            with open(output_csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(new_row)


