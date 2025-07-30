"""A basic implementation of Monte Carlo Tree Search (MCTS) for a game.

The overall idea is as follows:
1. We construct a Tree of States, with each node representing a game state and
    edges representing possible actions from the given state.
2. Each node keeps track of:
- The number of visits (how many times this node has been explored)
- The number of wins (how many times this node has led to a win)
3. The MCTS algorithm consists of four main steps:
- Selection: Traverse the tree to find a leaf node using a selection policy (e.g., UCT).
- Expansion: If the leaf node is not terminal, expand it by adding a child node for one of the possible actions.
- Simulation: Simulate a random game from the new node to a terminal state.
- Backpropagation: Update the node's visit and win counts based on the result of the simulation.

"""
import abc
import dataclasses
import numpy as np

from . import dots_and_boxes


@dataclasses.dataclass
class GameTreeNode(abc.ABC):
    # Game specific: State and auxiliary information
    state: np.ndarray
    auxiliary: dict = dataclasses.field(default_factory=dict)

    # parent and children nodes
    children: list['GameTreeNode'] = dataclasses.field(default_factory=list)
    parent: 'GameTreeNode' = None
    action: int = -1  # Action taken to reach this node from the parent

    # Statistics for MCTS
    num_visits: int = 0
    # Wins are from the perspective of player who is to move at this node.
    wins: list[int] = dataclasses.field(default_factory=lambda: [0, 0])  # [player_0_wins, player_1_wins]

    def best_child(self) -> 'GameTreeNode':
        """Return the child node with the highest number of visits."""
        raise NotImplementedError
    
    def best_action(self) -> int:
        """Return the best action from this node based on the win statistics."""
        raise NotImplementedError
    
    def expand(self) -> 'GameTreeNode':
        """Expand the tree by adding a new leaf node.
        Returns the new node added."""
        raise NotImplementedError
    
    def simulate(self, policy_fn: callable) -> int:
        """Simulate a game from the starting node using the tree's rollout function.
        
        Returns:
            True, if the player who is to move at the root node wins, False otherwise.
        """
        raise NotImplementedError
    
    def backpropagate(self, result: int) -> None:
        """Accumulate the result of a simulation in the tree.
        
        The result is from the perspective of the player who is to move at this node.
        """
        raise NotImplementedError


class TwoPlayerGameTreeNode(GameTreeNode):
    def __init__(self, game: dots_and_boxes.DotsAndBoxesGame) -> None:
        state, aux = game.to_vector()
        super().__init__(state=state, auxiliary=aux)
        self._game_state = game

    @property
    def valid_actions(self) -> list[int]:
        return self._game_state.valid_actions

    @property
    def q_value(self) -> float:
        """Average win ratio of the root node.
        """
        if self.num_visits == 0:
            return float('-inf')
        player = self._game_state.next_player
        # (wins - losses) / total_sims
        return (self.wins[player] - self.wins[1 - player])/ self.num_visits

    def best_child_uct(self, c: float = np.sqrt(2)) -> 'TwoPlayerGameTreeNode':
        """Get the best child node based on Upper Confidence Bound for Trees (UCT).
        
        If the next player is different from the player who is to move at this node,
        we reverse the perspective of the Q-value.
        """
        # https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
        
        # Compute the Q-values for each child node
        child_q_values = []
        for node in self.children:
            if node._game_state.next_player == self._game_state.next_player:  # same player
                child_q_values.append(node.q_value)
            else:  # other player
                child_q_values.append(1 - node.q_value)  # reverse perspective

        uct_values = [
            (child_q_value / child.num_visits) +
            c * np.sqrt(np.log(self.num_visits) / child.num_visits)
            for child_q_value, child in zip(child_q_values, self.children)
        ]
        return self.children[np.argmax(uct_values)]

    def best_action(self) -> int:
        """Return the best move from the root node based on the win statistics."""
        if not self.children:
            return -1  # No moves available

        return self.best_child_uct(c=0).action

    def _move_and_add_child(self, action: int) -> 'TwoPlayerGameTreeNode':
        """Perfrom one action from the current node and add a new child node."""
        new_state = self._game_state.move(action)
        new_node = TwoPlayerGameTreeNode(game=new_state)

        self.children.append(new_node)
        new_node.parent = self
        new_node.action = action

        return new_node

    def expand(self) -> 'TwoPlayerGameTreeNode':
        """Expand the tree by adding a new leaf node.

        If the root is fully expanded, selects the best child node and expands that.

        Returns the new node added."""
        # If the game is over, we can not expand further: return self
        if self._game_state.game_over():
            return self

        # A node corresponding to each valid action has been created
        if len(self.valid_actions) == len(self.children):
            # Select the best child based on UCT and then expand that
            return self.best_child_uct().expand()

        next_action = self.valid_actions[len(self.children)]
        return self._move_and_add_child(next_action)

    def simulate(self, policy_fn: callable) -> int:
        """Simulate a game from the starting node using the tree's rollout function.

        Returns:
            Index of the winning player (0 or 1), -1 if the game is a draw.
        """
        game = self._game_state
        while not game.game_over():
            action = policy_fn(game)
            game = game.move(action)

        # Find the result and return reward
        winner = game.get_winner()
        if winner is None:
            return -1
        return winner

    def backpropagate(self, winner: int) -> None:
        """Accumulate the result of a simulation in the tree.
        
        The result is from the perspective of the player who is to move at this node.
        """
        self.num_visits += 1
        if winner >= 0:
            self.wins[winner] += 1

        if self.parent:
            self.parent.backpropagate(winner)


def mcts(
    root: GameTreeNode,
    max_simulations: int,
    policy_fn: callable
) -> int:
    """Runs Monte Carlo Tree Search (MCTS) from the root node for a given number of simulations.
    
    Expands the tree and returns the best action from the root node.
    """
    for _ in range(max_simulations):
        # Add a new leaf node to the tree
        leaf_node = root.expand()

        # Run a simulation from the leaf node
        winner = leaf_node.simulate(policy_fn=policy_fn)

        # Backpropagate the result of the simulation
        leaf_node.backpropagate(winner)
    
    return root.best_action()
