"""
Adapted from https://github.com/tensorflow/minigo/blob/master/mcts.py

Implementation of the Monte-Carlo tree search algorithm as detailed in the
AlphaGo Zero paper (https://www.nature.com/articles/nature24270).
"""
import math
import random as rd
import collections
from typing import List

import numpy as np
import random

np.seterr(all='ignore')

# Exploration constant
c_PUCT = 2
# Dirichlet noise alpha parameter.
D_NOISE_ALPHA = 0.03
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly
TEMP_THRESHOLD = 5


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

    def revert_virtual_loss(self, up_to=None): pass

    def add_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass

    def update_visits(self, up_to): pass


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    def __init__(self, state, n_actions, TreeEnv, action=None, parent=None,
                 using_NN_estimations=False):
        """
        :param state: State that the node should hold.
        :param n_actions: Number of actions that can be performed in each
        state. Equal to the number of outgoing edges of the node.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param action: Index of the action that led from the parent node to
        this node.
        :param parent: Parent node.
        """
        self.TreeEnv = TreeEnv
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth+1
        self.parent = parent
        self.action = action
        self.state = state
        self.n_actions = n_actions
        self.is_expanded = False
        self.n_vlosses = 0  # Number of virtual losses on this node
        self.child_N = np.zeros([n_actions], dtype=np.float32)
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self.original_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_prior = np.zeros([n_actions], dtype=np.float32)
        self.children = {}
        self.using_NN_estimations = using_NN_estimations

    @property
    def N(self):
        """
        Returns the current visit count of the node.
        """
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return (self.W + 0.000000000001) / self.N

    @property
    def child_Q(self):
        return np.divide((self.child_W + 0.000000000001), self.child_N)

    @property
    def child_U(self):
        if self.using_NN_estimations:
            return (c_PUCT * math.sqrt(1 + self.N) *
                    self.child_prior / (1 + self.child_N))
        else:
            return c_PUCT * np.sqrt(2 * np.divide(math.log(self.N + 0.000000000001), self.child_N))

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        return self.child_Q + self.child_U

    def select_leaf(self):
        """
        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        child_N and child_W values but not as a dedicated node in the parent's
        children-mapping). Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.
        :return: Expanded leaf MCTSNode.
        """
        current = self
        while True:
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            # Choose action with highest score.
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action):
        """
        Adds a child node for the given action if it does not yet exists, and
        returns it.
        :param action: Action to take in current state which leads to desired
        child node.
        :return: Child MCTSNode.
        """
        if action not in self.children:
            # Obtain state following given action.
            new_state = self.TreeEnv.next_state(self.state, action)
            self.children[action] = MCTSNode(new_state, self.n_actions,
                                             self.TreeEnv,
                                             action=action, parent=self)
        return self.children[action]

    def add_virtual_loss(self, up_to):
        """
        Propagate a virtual loss up to a given node.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses += 1
        self.W -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """
        Undo adding virtual loss.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses -= 1
        self.W += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        """
        Revert visit increments.
        Sometimes, repeated calls to select_leaf return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But select_leaf has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.
        :param up_to: The node to propagate until.
        """
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def incorporate_estimates(self, action_probs, value, up_to):
        """
        Call if the node has just been expanded via `select_leaf` to
        incorporate the prior action probabilities and state value estimated
        by the neural network.
        :param action_probs: Action probabilities for the current node's state
        predicted by the neural network.
        :param value: Value of the current node's state predicted by the neural
        network.
        :param up_to: The node to propagate until.
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # TODO: Add assert here
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def update_visits(self, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.N += 1
        if self.parent is None or self is up_to:
            return
        self.parent.update_visits(up_to)

    def is_done(self):
        return self.TreeEnv.is_done_state(self.state, self.depth)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly large than 1 to encourage diversity in early steps.
        :return: Numpy array of shape (n_actions).
        """
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)

    def print_tree(self, level=0):
        node_string = "\033[94m|" + "----"*level
        node_string += "Node: action={}\033[0m".format(self.action)
        node_string += "\n• state:\n{}".format(self.state)
        node_string += "\n• N={}".format(self.N)
        node_string += "\n• score:\n{}".format(self.child_action_score)
        node_string += "\n• Q:\n{}".format(self.child_Q)
        node_string += "\n• P:\n{}".format(self.child_prior)
        print(node_string)
        for _, child in sorted(self.children.items()):
            child.print_tree(level+1)

    def perform_random_playouts(self, n_simulations):
        done_nodes = []
        matching_done_nodes_merits = []
        for i in range(n_simulations):
            current_leaf = self
            while not current_leaf.is_done():
                random_action = np.random.choice(current_leaf.n_actions)
                current_leaf = current_leaf.maybe_add_child(random_action)
            immediate_merit = current_leaf.TreeEnv.get_return(current_leaf.state, current_leaf.depth)
            done_nodes.append(current_leaf)
            matching_done_nodes_merits.append(immediate_merit)

        return done_nodes, matching_done_nodes_merits

class MCTS:
    """
    Represents a Monte-Carlo search tree and provides methods for performing
    the tree search.
    """

    def __init__(self, TreeEnv, seconds_per_move=None, agent_netw=None,
                 simulations_per_move=800, n_simulations=8):
        """
        :param agent_netw: Network for predicting action probabilities and
        state value estimate.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param seconds_per_move: Currently unused.
        :param simulations_per_move: Number of traversals through the tree
        before performing a step.
        :param n_simulations: Number of leaf nodes to collect before evaluating
        them in conjunction.
        
        """
        self.agent_netw = agent_netw
        self.TreeEnv = TreeEnv
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.n_simulations = n_simulations
        self.temp_threshold = None        # Overwritten in initialize_search
        self.using_NN_estimations = self.agent_netw is not None

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

        self.root = None

    def initialize_search(self, state=None):
        init_state = self.TreeEnv.initial_state()
        n_actions = self.TreeEnv.n_actions
        self.root = MCTSNode(init_state, n_actions, self.TreeEnv)
        # Number of steps into the episode after which we always select the
        # action with highest action probability rather than selecting randomly
        self.temp_threshold = TEMP_THRESHOLD

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

    def tree_search(self, n_simulations=None):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evaluates these leaf nodes.
        :param n_simulations: Number of leaf states which the agent network can
        evaluate at once. Limits the number of simulations.
        :return: The leaf nodes which were expanded.
        """
        if n_simulations is None:
            n_simulations = self.n_simulations
        leaves = []
        # Failsafe for when we encounter almost only done-states which would
        # prevent the loop from ever ending.
        failsafe = 0
        while len(leaves) < n_simulations and failsafe < n_simulations * 2:
            failsafe += 1
            # self.root.print_tree()
            # print("_"*50)
            leaf = self.root.select_leaf()
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.
            if leaf.is_done():
                value = self.TreeEnv.get_return(leaf.state, leaf.depth)
                leaf.backup_value(value, up_to=self.root)
                continue
            # Otherwise, discourage other threads to take the same trajectory
            # via virtual loss and enqueue the leaf for evaluation by agent
            # network.
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        # Evaluate the leaf-states all at once and backup the value estimates.
        if leaves:
            action_probs, values = self.agent_netw.step(
                self.TreeEnv.get_obs_for_states([leaf.state for leaf in leaves]))
            for leaf, action_prob, value in zip(leaves, action_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_estimates(action_prob, value, up_to=self.root)
        return leaves

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        """
        if self.root.depth > self.temp_threshold:
            action = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = rd.random()
            action = cdf.searchsorted(selection)
            assert self.root.child_N[action] != 0
        return action

    def take_action(self, action):
        """
        Takes the specified action for the root state. The subsequent child
        state becomes the new root state of the tree.
        :param action: Action to take for the root state.
        """
        # Store data to be used as experience tuples.
        ob = self.TreeEnv.get_obs_for_states([self.root.state])
        self.obs.append(ob)
        self.searches_pi.append(
            self.root.visits_as_probs()) # TODO: Use self.root.position.n < self.temp_threshold as argument
        self.qs.append(self.root.Q)
        reward = (self.TreeEnv.get_return(self.root.children[action].state,
                                          self.root.children[action].depth)
                  - sum(self.rewards))
        self.rewards.append(reward)

        # Resulting state becomes new root of the tree.
        self.root = self.root.maybe_add_child(action)
        del self.root.parent.children


def execute_regular_mcts_episode(mcts, num_simulations, num_expansion, best_merit=-float('inf')):
    """
        Executes a single episode of the task using Monte-Carlo tree search.
        An episode consists of 4 stages: selection, expansion, simulation and back propagation.
        It returns the experience tuples collected during the search.
        :param mcts: mcts object to run the episode on.
        :param num_simulations: Number of simulations (traverses from root to leaf)
        per action.
        :param num_expansion: Number of nodes to expand the selected node with.
        :return: The observations for each step of the episode, the policy outputs
        as output by the MCTS (not the pure neural network outputs), the individual
        rewards in each step, total return for this episode and the final state of
        this episode.
        """
    # the first action of the episode.
    k_node: MCTSNode = mcts.root.select_leaf()
    k_node.is_expanded = True
    if k_node.is_done():
        return
    curr_best_path_node = None
    expanded_random_actions = random.sample(range(k_node.n_actions), min(num_expansion, k_node.n_actions))
    for action in expanded_random_actions:
        new_child: MCTSNode = k_node.maybe_add_child(action)
        done_nodes, immediate_merits = new_child.perform_random_playouts(num_simulations)
        # backpropagation:
        for done_node, im_merit in zip(done_nodes, immediate_merits):
            done_node.backup_value(im_merit, up_to=mcts.root)
            done_node.update_visits(up_to=mcts.root)
            if im_merit > best_merit:
                curr_best_path_node = done_node
                best_merit = im_merit
    # TODO return the current best path. when upgrading using NN add memory and return it for training.
    return curr_best_path_node, best_merit


def execute_episode_using_NN(agent_netw, num_simulations, TreeEnv):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(agent_netw, TreeEnv)
    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_leaf()
    probs, vals = agent_netw.step(
        TreeEnv.get_obs_for_states([first_node.state]))
    first_node.incorporate_estimates(probs[0], vals[0], first_node)

    while True:
        # mcts.root.inject_noise()
        current_simulations = mcts.root.N

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search()

        # mcts.root.print_tree()
        # print("_"*100)

        # expansion:
        action = mcts.pick_action()
        mcts.take_action(action)

        if mcts.root.is_done():
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    ret = [TreeEnv.get_return(mcts.root.state, mcts.root.depth) for _
           in range(len(mcts.rewards))]

    total_rew = np.sum(mcts.rewards)

    obs = np.concatenate(mcts.obs)
    return (obs, mcts.searches_pi, ret, total_rew, mcts.root.state)


