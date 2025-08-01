import numpy as np

from toylib_projects.dotsAndBoxes import dots_and_boxes
from toylib_projects.dotsAndBoxes import mcts
from toylib_projects.dotsAndBoxes.interfaces import visualize


def human_player(game: dots_and_boxes.DotsAndBoxesGame, last_action: int) -> int:
    """A human player based on terminal input."""
    del game, last_action # Unused in this function
    try:
        move = int(input("Enter your move (-1 to exit): "))
    except ValueError as e:
        print("Invalid input. Please enter a number.")
        raise e
    return move


def get_mcts_player(
    policy_fn: callable,
    max_simulattions: int = 100,
) -> callable:
    def inner(game: dots_and_boxes.DotsAndBoxesGame, last_action: int) -> int:
        """Returns a player function that uses MCTS to select the next move."""
        del last_action
        root = mcts.TwoPlayerGameTreeNode(game=game)
        action = mcts.mcts(
            root=root,
            max_simulattions=max_simulattions,
            policy_fn=policy_fn
        )
        visualize.plot_tree(root)
        return action

    return inner

class MCTSPlayer:
    def __init__(self, policy_fn: callable, max_simulations: int = 100, debug: bool = False, wait_for_input: bool = False):
        self.policy_fn = policy_fn
        self.max_simulations = max_simulations

        # Debug and visualization options
        self.root = None
        self.debug = debug
        self.wait_for_input = wait_for_input

    def __call__(self, state: dots_and_boxes.DotsAndBoxesGame, last_action: int) -> int:
        # If the root node is None, create a new root node
        self.root = mcts.TwoPlayerGameTreeNode(game=state)

        # Find the best action for this player using MCTS
        action = mcts.mcts(
            root=self.root,
            max_simulations=self.max_simulations,
            policy_fn=self.policy_fn
        )

        # Visualize the tree if debugging is enabled
        if self.debug:
            visualize.plot_tree(self.root, max_children_per_node=8, max_depth=4)
        if self.wait_for_input:
            key = input("Press Enter to continue ('q' or '-1' to exit)...")
            if key == 'q' or key == '-1':
                print("Exiting...")
                return -1
        return action

def random_policy(state: dots_and_boxes.DotsAndBoxesGame) -> int:
    """A random policy that selects a valid action at random."""
    return np.random.choice(state.valid_actions)