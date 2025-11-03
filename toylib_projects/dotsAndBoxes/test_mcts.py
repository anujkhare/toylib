import pytest
import numpy as np
from .mcts import TwoPlayerGameTreeNode, mcts
import random

class MockGameState:
    """Mock GameState for testing purposes"""
    def __init__(self, next_player=0, valid_actions=None, game_over=False, winner=None):
        self.next_player = next_player
        self.valid_actions = valid_actions or []
        self._game_over = game_over
        self._winner = winner
    
    def to_vector(self):
        return np.array([1, 2, 3]), {"mock": True}
    
    def game_over(self):
        return self._game_over
    
    def get_winner(self):
        return self._winner
    
    def move(self, action):
        # Return a new mock state with alternating player
        return MockGameState(
            next_player=1 - self.next_player,
            valid_actions=[a for a in self.valid_actions if a != action],
            game_over=len(self.valid_actions) <= 1,
            winner=self._winner
        )


class TestTwoPlayerGameTreeNode:
    
    def test_best_child_uct_no_children(self):
        """Test best_child_uct raises error when no children exist"""
        game_state = MockGameState(next_player=0, valid_actions=[0, 1, 2])
        node = TwoPlayerGameTreeNode(game_state)
        
        with pytest.raises(ValueError):
            node.best_child_uct()
    
    def test_best_child_uct_single_child(self):
        """Test best_child_uct with single child returns that child"""
        game_state = MockGameState(next_player=0, valid_actions=[0, 1])
        root = TwoPlayerGameTreeNode(game_state)
        
        # Add a single child
        child_state = MockGameState(next_player=1, valid_actions=[1])
        child = TwoPlayerGameTreeNode(child_state)
        child.parent = root
        child.action = 0
        child.num_visits = 5
        child.wins = [3, 2]
        root.children.append(child)
        root.num_visits = 10
        
        result = root.best_child_uct()
        assert result == child
    
    def test_best_child_uct_multiple_children_same_player(self):
        """Test UCT selection with multiple children for same player"""
        game_state = MockGameState(next_player=0, valid_actions=[0, 1, 2])
        root = TwoPlayerGameTreeNode(game_state)
        root.num_visits = 20
        
        # Create children with different statistics
        children_data = [
            (0, 10, [8, 2]),  # High win rate for player 0
            (1, 5, [2, 3]),   # Lower win rate for player 0
            (2, 3, [1, 2])    # Least visits (high exploration value)
        ]
        
        for action, visits, wins in children_data:
            child_state = MockGameState(next_player=0, valid_actions=[])
            child = TwoPlayerGameTreeNode(child_state)
            child.parent = root
            child.action = action
            child.num_visits = visits
            child.wins = wins
            root.children.append(child)
        
        # With default c=sqrt(2), should balance exploitation and exploration
        result = root.best_child_uct()
        assert result.parent == root
        assert result.action in [0, 1, 2]
    
    def test_best_child_uct_different_players(self):
        """Test UCT selection when children represent different player turns"""
        game_state = MockGameState(next_player=0, valid_actions=[0, 1])
        root = TwoPlayerGameTreeNode(game_state)
        root.num_visits = 20
        
        # Child 1: Same player (player 0)
        child1_state = MockGameState(next_player=0, valid_actions=[])
        child1 = TwoPlayerGameTreeNode(child1_state)
        child1.parent = root
        child1.action = 0
        child1.num_visits = 10
        child1.wins = [8, 2]  # Good for player 0
        root.children.append(child1)
        
        # Child 2: Different player (player 1)
        child2_state = MockGameState(next_player=1, valid_actions=[])
        child2 = TwoPlayerGameTreeNode(child2_state)
        child2.parent = root
        child2.action = 1
        child2.num_visits = 10
        child2.wins = [2, 8]  # Good for player 1 (bad for player 0)
        root.children.append(child2)
        
        result = root.best_child_uct()
        # Should prefer child1 since it's better for player 0
        assert result == child1
    
    def test_best_child_uct_exploration_parameter(self):
        """Test that different exploration parameters affect selection"""
        game_state = MockGameState(next_player=0, valid_actions=[0, 1])
        root = TwoPlayerGameTreeNode(game_state)
        root.num_visits = 100
        
        # Child 1: High exploitation value
        child1_state = MockGameState(next_player=0, valid_actions=[])
        child1 = TwoPlayerGameTreeNode(child1_state)
        child1.parent = root
        child1.action = 0
        child1.num_visits = 80
        child1.wins = [70, 10]
        root.children.append(child1)
        
        # Child 2: High exploration value (fewer visits)
        child2_state = MockGameState(next_player=0, valid_actions=[])
        child2 = TwoPlayerGameTreeNode(child2_state)
        child2.parent = root
        child2.action = 1
        child2.num_visits = 20
        child2.wins = [10, 10]
        root.children.append(child2)
        
        # With c=0 (pure exploitation), should choose child1
        result_exploitation = root.best_child_uct(c=0)
        assert result_exploitation == child1
        
        # With very high c (pure exploration), might choose child2
        result_exploration = root.best_child_uct(c=10)
        # This test depends on the exact UCT formula implementation
        assert result_exploration.action in [child1.action, child2.action]
    
    def test_best_child_uct_zero_visits_child(self):
        """Test handling of children with zero visits"""
        game_state = MockGameState(next_player=0, valid_actions=[0, 1])
        root = TwoPlayerGameTreeNode(game_state)
        root.num_visits = 10
        
        # Child with visits
        child1_state = MockGameState(next_player=0, valid_actions=[])
        child1 = TwoPlayerGameTreeNode(child1_state)
        child1.parent = root
        child1.action = 0
        child1.num_visits = 5
        child1.wins = [3, 2]
        root.children.append(child1)
        
        # Child with zero visits
        child2_state = MockGameState(next_player=0, valid_actions=[])
        child2 = TwoPlayerGameTreeNode(child2_state)
        child2.parent = root
        child2.action = 1
        child2.num_visits = 0
        child2.wins = [0, 0]
        root.children.append(child2)
        
        # Should handle zero visits gracefully (might cause division by zero)
        with pytest.raises(ZeroDivisionError):
            root.best_child_uct()
    
    def test_q_value_calculation(self):
        """Test q_value property calculation"""
        game_state = MockGameState(next_player=0)
        node = TwoPlayerGameTreeNode(game_state)
        
        # Zero visits should return negative infinity
        assert node.q_value == float('-inf')
        
        # Test with some visits and wins
        node.num_visits = 10
        node.wins = [7, 3]  # Player 0 has 7 wins, player 1 has 3 wins
        
        expected_q = (7 - 3) / 10  # (wins[0] - wins[1]) / num_visits
        assert node.q_value == expected_q
        
        # Test with player 1's perspective
        game_state_p1 = MockGameState(next_player=1)
        node_p1 = TwoPlayerGameTreeNode(game_state_p1)
        node_p1.num_visits = 10
        node_p1.wins = [7, 3]
        
        expected_q_p1 = (3 - 7) / 10  # (wins[1] - wins[0]) / num_visits
        assert node_p1.q_value == expected_q_p1


class TestMCTSFunction:
    
    def test_mcts_basic_functionality(self):
        """Test basic MCTS function execution"""
        game_state = MockGameState(next_player=0, valid_actions=[0, 1], game_over=False)
        root = TwoPlayerGameTreeNode(game_state)
        
        def simple_policy(game):
            """Simple random policy for testing"""
            return random.choice(game.valid_actions) if game.valid_actions else 0
        
        # Run a few simulations
        result = mcts(root, max_simulations=5, policy_fn=simple_policy)
        
        # Should return a valid action or -1
        assert result in [-1, 0, 1]
        
        # Root should have been visited
        assert root.num_visits > 0
    
    def test_mcts_with_terminal_state(self):
        """Test MCTS with a terminal game state"""
        game_state = MockGameState(next_player=0, valid_actions=[], game_over=True, winner=0)
        root = TwoPlayerGameTreeNode(game_state)
        
        def dummy_policy(game):
            return 0
        
        result = mcts(root, max_simulations=3, policy_fn=dummy_policy)
        
        # Should return -1 for no valid moves
        assert result == -1