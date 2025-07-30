import pytest
import numpy as np
from .dots_and_boxes import DotsAndBoxesGame, Direction


class TestDotsAndBoxesGame:
    def test_gamestate_initialization(self):
        """Test DotsAndBoxesGame initialization with basic parameters"""
        game = DotsAndBoxesGame(rows=4, cols=3)

        assert game.rows == 4
        assert game.cols == 3
        assert game.next_player == 0

        # Check array dimensions
        assert game.filled_vertical.shape == (3, 3)  # (rows-1, cols)
        assert game.filled_horizontal.shape == (4, 2)  # (rows, cols-1)
        assert game.boxes_by_player.shape == (2, 3, 2)  # (2, rows-1, cols-1)

        # Check arrays are initialized with zeros/False
        assert not game.filled_vertical.any()
        assert not game.filled_horizontal.any()
        assert not game.boxes_by_player.any()
        assert np.array_equal(game.scores, [0, 0])

    def test_gamestate_arrays_are_boolean(self):
        """Test that filled arrays are boolean type"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        assert game.filled_vertical.dtype == np.bool_
        assert game.filled_horizontal.dtype == np.bool_
        assert game.boxes_by_player.dtype == np.bool_

    def test_to_dict_conversion(self):
        """Test DotsAndBoxesGame to dictionary conversion"""
        game = DotsAndBoxesGame(rows=3, cols=3, next_player=1)

        # Modify some values to test conversion
        game.filled_vertical[0, 0] = True
        game.filled_horizontal[1, 1] = True
        game.boxes_by_player[0, 0, 1] = True
        game.scores[1] = 2

        result = game.to_dict()

        assert result["rows"] == 3
        assert result["cols"] == 3
        assert result["next_player"] == 1
        assert result["filled_vertical"][0][0]
        assert result["filled_horizontal"][1][1]
        assert result["boxes_by_player"][0][0][1]

    def test_load_from_dict(self):
        """Test loading DotsAndBoxesGame from dictionary"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        data = {
            "rows": 3,
            "cols": 3,
            "next_player": 1,
            "filled_vertical": [[True, False, False], [False, True, False]],
            "filled_horizontal": [[False, True], [True, False], [False, False]],
            "boxes_by_player": [
                [[True, False], [False, True]],
                [[False, True], [True, False]],
            ],
            "line_owners": {
                "vertical": [[0, -1, -1], [-1, 1, -1]],
                "horizontal": [[-1, 0], [1, -1], [-1, -1]],
            },
            "scores": [0, 0],
        }

        game.load_from_dict(data)

        assert game.next_player == 1
        assert game.filled_vertical[0, 0]
        assert game.filled_vertical[1, 1]
        assert game.filled_horizontal[0, 1]
        assert game.filled_horizontal[1, 0]
        assert game.boxes_by_player[0, 0, 0]
        assert game.boxes_by_player[1, 0, 1]

    def test_load_from_dict_dimension_mismatch(self):
        """Test that loading fails with mismatched dimensions"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        data = {
            "rows": 4,  # Different dimension
            "cols": 3,
            "next_player": 0,
            "filled_vertical": [],
            "filled_horizontal": [],
            "boxes_by_player": [],
            "line_owners": {"vertical": [], "horizontal": []},
            "scores": [],
        }

        with pytest.raises(AssertionError, match="Loaded dimensions do not match"):
            game.load_from_dict(data)

    def test_roundtrip_dict_conversion(self):
        """Test that to_dict and load_from_dict are inverse operations"""
        game1 = DotsAndBoxesGame(rows=4, cols=3, next_player=1)

        # Set some values
        game1.filled_vertical[1, 0] = True
        game1.filled_horizontal[2, 1] = True
        game1.boxes_by_player[0, 1, 1] = True
        game1.boxes_by_player[1, 0, 0] = True
        game1.scores[0] = 1
        game1.scores[1] = 2

        # Convert to dict and back
        data = game1.to_dict()
        game2 = DotsAndBoxesGame(rows=4, cols=3)
        game2.load_from_dict(data)

        # Verify all fields match
        assert game2.rows == game1.rows
        assert game2.cols == game1.cols
        assert game2.next_player == game1.next_player
        np.testing.assert_array_equal(game2.filled_vertical, game1.filled_vertical)
        np.testing.assert_array_equal(game2.filled_horizontal, game1.filled_horizontal)
        np.testing.assert_array_equal(game2.boxes_by_player, game1.boxes_by_player)
        np.testing.assert_array_equal(game2.scores, game1.scores)

    def test_action_space(self):
        """Test action space calculation"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        # Should have (3-1)*3 + 3*(3-1) = 6 + 6 = 12 total actions
        assert len(game.action_space) == 12
        assert game.n_vertical_moves == 6
        assert game.n_horizontal_moves == 6

    def test_action_to_coordinates(self):
        """Test action to coordinates conversion"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        # Test vertical actions (0-5)
        r, c, direction = game._action_to_coordinates(0)
        assert (r, c, direction) == (0, 0, Direction.VERTICAL)

        r, c, direction = game._action_to_coordinates(5)
        assert (r, c, direction) == (1, 2, Direction.VERTICAL)

        # Test horizontal actions (6-11)
        r, c, direction = game._action_to_coordinates(6)
        assert (r, c, direction) == (0, 0, Direction.HORIZONTAL)

        r, c, direction = game._action_to_coordinates(11)
        assert (r, c, direction) == (2, 1, Direction.HORIZONTAL)

    def test_valid_actions(self):
        """Test valid actions calculation"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        # Initially all actions should be valid
        valid = game.valid_actions
        assert len(valid) == 12

        # Fill some lines and check valid actions
        game.filled_vertical[0, 0] = True
        game.filled_horizontal[1, 1] = True

        valid = game.valid_actions
        assert 0 not in valid  # First vertical action should be invalid
        assert 9 not in valid  # Horizontal action (1,1) should be invalid
        assert len(valid) == 10

    def test_move_basic(self):
        """Test basic move functionality"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        # Make a move
        new_game = game.move(0)  # Draw vertical line at (0,0)

        # Original game should be unchanged
        assert not game.filled_vertical[0, 0]

        # New game should have the line filled
        assert new_game.filled_vertical[0, 0]
        assert new_game.next_player == 1  # Should switch players

    def test_move_completes_box(self):
        """Test move that completes a box"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        # Set up a box that's almost complete (missing one line)
        game.filled_vertical[0, 0] = True
        game.filled_vertical[0, 1] = True
        game.filled_horizontal[0, 0] = True
        # Missing: game.filled_horizontal[1, 0] = True

        # Complete the box
        new_game = game.move(8)  # Horizontal line at (1,0)

        # Check that box is completed and assigned to current player
        assert new_game.boxes_by_player[0, 0, 0] == True
        assert new_game.scores[0] == 1
        assert new_game.next_player == 0  # Same player continues

    def test_game_over(self):
        """Test game over detection"""
        game = DotsAndBoxesGame(rows=2, cols=2)  # Only one box

        assert not game.game_over()

        # Fill all lines to complete the box
        game.filled_vertical[0, 0] = True
        game.filled_vertical[0, 1] = True
        game.filled_horizontal[0, 0] = True
        game.filled_horizontal[1, 0] = True
        game.boxes_by_player[0, 0, 0] = True

        assert game.game_over()

    def test_get_winner(self):
        """Test winner determination"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        # Game not over
        assert game.get_winner() is None

        # Set up completed game with player 0 winning
        game.boxes_by_player[0, :, :] = True  # Player 0 gets all boxes
        game.scores[0] = 4
        game.scores[1] = 0

        assert game.get_winner() == 0

        # Tie game
        game.scores[1] = 4
        assert game.get_winner() is None

    def test_with_different_dimensions(self):
        """Test DotsAndBoxesGame with various dimensions"""
        test_cases = [(2, 2), (3, 4), (5, 3), (10, 2)]

        for rows, cols in test_cases:
            game = DotsAndBoxesGame(rows=rows, cols=cols)
            assert game.rows == rows
            assert game.cols == cols
            assert game.filled_vertical.shape == (rows - 1, cols)
            assert game.filled_horizontal.shape == (rows, cols - 1)
            assert game.boxes_by_player.shape == (2, rows - 1, cols - 1)
            assert game.n_vertical_moves == (rows - 1) * cols
            assert game.n_horizontal_moves == rows * (cols - 1)

    def test_invalid_move(self):
        """Test that invalid moves raise ValueError"""
        game = DotsAndBoxesGame(rows=3, cols=3)

        # Fill a line
        game.filled_vertical[0, 0] = True

        # Try to make the same move again
        with pytest.raises(ValueError, match="Invalid action"):
            game.move(0)
