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
from src.mcts import MCTS, execute_regular_mcts_episode
from src.sat3_env import SAT3Env
from qutip import fidelity


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
    n_actions = SAT3Env.n_actions
    n_sim = 5
    n_exp = 10

    # trainer = Trainer(lambda: HillClimbingPolicy(n_obs, 20, n_actions))
    # network = trainer.step_model
    #
    # mem = ReplayMemory(200,
    #                    { "ob": np.long,
    #                      "pi": np.float32,
    #                      "return": np.float32},
    #                    { "ob": [],
    #                      "pi": [n_actions],
    #                      "return": []})

    # def test_agent(iteration):
    #     test_env = HillClimbingEnv()
    #     total_rew = 0
    #     state, reward, done, _ = test_env.reset()
    #     step_idx = 0
    #     while not done:
    #         log(test_env, iteration, step_idx, total_rew)
    #         p, _ = network.step(np.array([state]))
    #         # print(p)
    #         action = np.argmax(p)
    #         state, reward, done, _ = test_env.step(action)
    #         step_idx+=1
    #         total_rew += reward
    #     log(test_env, iteration, step_idx, total_rew)

    value_losses = []
    policy_losses = []
    mcts = MCTS(SAT3Env)
    mcts.initialize_search()
    best_merit = -float('inf')
    best_node_path = None
    for i in range(150):
        # if i % 50 == 0 and i != 0:
        #     test_agent(i)
        #     plt.plot(value_losses, label="value loss")
        #     plt.plot(policy_losses, label="policy loss")
        #     plt.legend()
        #     plt.show()

        curr_best_node, best_merit = execute_regular_mcts_episode(mcts, num_expansion=n_exp,
                                                                  num_simulations=n_sim,
                                                                  best_merit=best_merit)
        best_node_path = best_node_path if curr_best_node is None else curr_best_node
        if i % 25 == 0 and i != 0:
            print(f'after {i} episodes of mcts the best fidelity is:')
            best_node_path.TreeEnv.get_return(best_node_path.state, best_node_path.depth)
            best_rho_final = best_node_path.TreeEnv.get_rho_final_of_done_state(best_node_path.state)
            true_rho_final = best_node_path.TreeEnv.get_rho_lowest_energy_eigenstate_of_H_final()
            print(fidelity(best_rho_final, true_rho_final))
        # mem.add_all({"ob": obs, "pi": pis, "return": returns})
        #
        # batch = mem.get_minibatch()
        #
        # vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        # value_losses.append(vl)
        # policy_losses.append(pl)
    print()
