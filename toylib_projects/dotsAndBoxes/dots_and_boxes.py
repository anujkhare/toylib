import copy
import dataclasses
import enum
import numpy as np
from typing import Any, Optional, Tuple



class Direction(enum.Enum):
    """Enum for the direction of the line in the game"""

    VERTICAL = 0  # Vertical line
    HORIZONTAL = 1  # Horizontal line


@dataclasses.dataclass
class DotsAndBoxesGame:
    rows: int  # number of rows of lines
    cols: int  # number of columns of lines
    next_player: int = 0  # Player to move next (0 or 1)

    @property
    def valid_actions(self) -> np.ndarray:
        """Get the valid actions for the current game state"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __post_init__(self):
        # There are (rows - 1) vertical slots and (cols - 1) horizontal slots
        # row 0 of filled_vertical corresponds to the first row of vertical lines,
        # with filled_vertical[r, c] indicating the vertical line going down from (r, c)
        self.filled_vertical = np.zeros((self.rows - 1, self.cols), dtype=np.bool_)
        self.filled_horizontal = np.zeros((self.rows, self.cols - 1), dtype=np.bool_)
        self.scores = np.zeros(2, dtype=int)  # scores for player 1 and player 2

        ################################################################################
        # Auxiliary information that is not strictly necessary for the game logic,
        # but useful for visualization and UI.
        # These are NOT guaranteed to be up-to-date with the game state - especially during
        # simulation if the game state is serialized and deserialized.

        # (rows-1, cols-1) bool array for boxes won by P1 and P2 respectively.
        self.boxes_by_player = np.zeros(
            (2, self.rows - 1, self.cols - 1), dtype=np.bool_
        )

        self.line_owners = {
            "vertical": np.full((self.rows - 1, self.cols), -1, dtype=int),
            "horizontal": np.full((self.rows, self.cols - 1), -1, dtype=int),
        }
        ################################################################################

    def to_dict(self):
        """Convert GameState to a dictionary. Includes auxiliary information."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "next_player": self.next_player,
            "filled_vertical": self.filled_vertical.tolist(),
            "filled_horizontal": self.filled_horizontal.tolist(),
            "boxes_by_player": self.boxes_by_player.tolist(),
            "line_owners": {
                "vertical": self.line_owners["vertical"].tolist(),
                "horizontal": self.line_owners["horizontal"].tolist(),
            },
            "scores": self.scores.tolist(),
        }

    def load_from_dict(self, data):
        """Create GameState from a dictionary, including auxiliary information."""
        assert self.rows == data["rows"] and self.cols == data["cols"], (
            "Loaded dimensions do not match the initialized dimensions."
        )
        self.next_player = data["next_player"]
        self.filled_vertical = np.array(data["filled_vertical"], dtype=np.bool_)
        self.filled_horizontal = np.array(data["filled_horizontal"], dtype=np.bool_)
        self.boxes_by_player = np.array(data["boxes_by_player"], dtype=np.bool_)
        self.line_owners["vertical"] = np.array(
            data["line_owners"]["vertical"], dtype=int
        )
        self.line_owners["horizontal"] = np.array(
            data["line_owners"]["horizontal"], dtype=int
        )
        self.scores = np.array(data["scores"], dtype=int)

    def to_vector(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Convert the game state to a vector representation and an auxiliary information map."""
        # Flatten the filled vertical and horizontal lines
        filled_vertical_flat = self.filled_vertical.flatten()
        filled_horizontal_flat = self.filled_horizontal.flatten()

        # Combine all parts into a single vector
        return np.concatenate(
            (
                filled_vertical_flat,
                filled_horizontal_flat,
                self.scores.ravel(),
                np.array([self.next_player], dtype=int),
            )
        ), {
            "rows": self.rows,
            "cols": self.cols,
        }

    @classmethod
    def load_from_vector(cls, vector: np.ndarray, aux: dict[str, Any]) -> None:
        """Load the game state from a vector representation and an auxiliary information map."""
        assert "rows" in aux and "cols" in aux, (
            "Auxiliary information must contain 'rows' and 'cols'."
        )
        rows, cols = aux["rows"], aux["cols"]
        obj = cls(rows=rows, cols=cols)

        n_vertical = cols * (rows - 1)
        n_horizontal = rows * (cols - 1)
        obj.filled_vertical = vector[:n_vertical].reshape((rows - 1, cols))
        obj.filled_horizontal = vector[n_vertical : n_vertical + n_horizontal].reshape(
            (rows, cols - 1)
        )
        obj.scores = vector[-3:-1]
        obj.next_player = vector[-1]

        # Load auxiliary information
        if "boxes_by_player" in aux:
            obj.boxes_by_player = np.array(aux["boxes_by_player"], dtype=np.bool_)
        if "line_owners" in aux:
            obj.line_owners["vertical"] = np.array(
                aux["line_owners"]["vertical"], dtype=int
            )
            obj.line_owners["horizontal"] = np.array(
                aux["line_owners"]["horizontal"], dtype=int
            )

        return obj

    def move(self, action: int) -> "GameState":
        """Make a move in the game and return a new game state.

        A copy of the game state is returned with the move applied.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def n_vertical_moves(self) -> int:
        """Get the number of vertical moves available in the game"""
        return (self.rows - 1) * self.cols

    @property
    def n_horizontal_moves(self) -> int:
        """Get the number of horizontal moves available in the game"""
        return self.rows * (self.cols - 1)

    @property
    def action_space(self) -> np.ndarray:
        """Get the action space for the current game"""
        return np.arange(self.n_vertical_moves + self.n_horizontal_moves)

    def _action_to_coordinates(self, action: int) -> tuple[int, int, Direction]:
        """Convert an action index to coordinates in the game grid"""
        # Row major: vertical (1, 1) denotes a vertical edge starting at (1, 1) and going downwards,
        # horizontal (1, 1) denotes a horizontal edge starting at (1, 1) and going rightwards.
        if action < self.n_vertical_moves:
            # Vertical move
            return action // self.cols, action % self.cols, Direction.VERTICAL
        else:
            # Horizontal move
            action -= self.n_vertical_moves
            return (
                action // (self.cols - 1),
                action % (self.cols - 1),
                Direction.HORIZONTAL,
            )

    @property
    def valid_actions(self) -> np.ndarray:
        """Get the valid actions for the current game state"""
        valid_vertical = ~self.filled_vertical.flatten()
        valid_horizontal = ~self.filled_horizontal.flatten()
        return np.where(np.concatenate((valid_vertical, valid_horizontal)))[0]

    def _is_box_completed(self, r: int, c: int) -> bool:
        """Check if a box is completed at the given coordinates.

        The (r, c) coordinates refer to the top-left corner of the box.

        If the coordinates are out of bounds, return False.
        """
        if r < 0 or r >= self.rows - 1 or c < 0 or c >= self.cols - 1:
            return False
        return (
            self.filled_vertical[r, c]
            and self.filled_horizontal[r, c]
            and self.filled_vertical[r, c + 1]
            and self.filled_horizontal[r + 1, c]
        )

    def move(self, action: int) -> "DotsAndBoxesGame":
        """Make a move in the game and return a new game state.

        A copy of the game state is returned with the move applied.
        """
        # Check if the action is valid
        if action not in self.valid_actions:
            raise ValueError("Invalid action")
        new_game = copy.deepcopy(self)

        r, c, direction = self._action_to_coordinates(action)
        current_player = self.next_player

        # Mark the edge as filled in the game state
        if direction == Direction.VERTICAL:
            new_game.filled_vertical[r, c] = True
            new_game.line_owners["vertical"][r, c] = current_player
        else:
            new_game.filled_horizontal[r, c] = True
            new_game.line_owners["horizontal"][r, c] = current_player

        # Check if this move completes any boxes
        boxes_completed = []

        # For a vertical line at (r, c), check boxes to the left and right
        if direction == Direction.VERTICAL:
            # Check box to the left (r, c-1)
            if c > 0 and new_game._is_box_completed(r, c - 1):
                new_game.boxes_by_player[current_player, r, c - 1] = True
                boxes_completed.append((r, c - 1))
            # Check box to the right (r, c)
            if c < new_game.cols - 1 and new_game._is_box_completed(r, c):
                new_game.boxes_by_player[current_player, r, c] = True
                boxes_completed.append((r, c))
        else:  # Horizontal line
            # Check box above (r-1, c)
            if r > 0 and new_game._is_box_completed(r - 1, c):
                new_game.boxes_by_player[current_player, r - 1, c] = True
                boxes_completed.append((r - 1, c))
            # Check box below (r, c)
            if r < new_game.rows - 1 and new_game._is_box_completed(r, c):
                new_game.boxes_by_player[current_player, r, c] = True
                boxes_completed.append((r, c))

        # Update the score for the current player
        if boxes_completed:
            new_game.scores[current_player] += len(boxes_completed)

        # Switch to the next player only if no boxes were completed
        if not boxes_completed:
            new_game.next_player = 1 - new_game.next_player

        # Return the new game state
        return new_game

    def get_scores(self) -> Tuple[int, int]:
        """Get the scores for both players"""
        return self.scores[0], self.scores[1]

    def game_over(self) -> bool:
        """Check if the game is over (all boxes are filled)"""
        total_boxes = (self.rows - 1) * (self.cols - 1)
        return np.sum(self.boxes_by_player) == total_boxes

    def get_winner(self) -> Optional[int]:
        """Get the winner of the game. Returns None if game is not over or tied"""
        score1, score2 = self.get_scores()
        if not self.game_over() or score1 == score2:
            return None
        if score1 > score2:
            return 0  # Player 1 wins
        elif score2 > score1:
            return 1  # Player 2 wins
