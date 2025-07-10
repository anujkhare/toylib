import abc
import dataclasses
import numpy as np
from typing import Type
import enum


class Direction(enum.Enum):
    """Enum for the direction of the line in the game"""

    VERTICAL = 0  # Vertical line
    HORIZONTAL = 1  # Horizontal line


@dataclasses.dataclass
class GameState:
    rows: int  # number of rows of lines
    cols: int  # number of columns of lines
    next_player: int = 0

    def __post_init__(self):
        # There are (rows - 1) vertical slots and (cols - 1) horizontal slots
        # row 0 of filled_vertical corresponds to the first row of vertical lines,
        # with filled_vertical[r, c] indicating the vertical line going down from (r, c)
        self.filled_vertical = np.zeros((self.rows - 1, self.cols), dtype=np.bool_)
        self.filled_horizontal = np.zeros((self.rows, self.cols - 1), dtype=np.bool_)

        # (rows-1, cols-1) bool array for boxes won by P1 and P2 respectively.
        self.boxes_by_player = np.zeros(
            (2, self.rows - 1, self.cols - 1), dtype=np.bool_
        )
        # self.boxes_player_1 = np.zeros((self.rows-1, self.cols-1), dtype=np.bool_)
        # # (m-1, n-1) bool array for boxes won by P2
        # self.boxes_player_2 = np.zeros((self.rows-1, self.cols-1), dtype=np.bool_)

    def to_dict(self):
        """Convert GameState to a dictionary"""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "next_player": self.next_player,
            "filled_vertical": self.filled_vertical.tolist(),
            "filled_horizontal": self.filled_horizontal.tolist(),
            "boxes_by_player": self.boxes_by_player.tolist(),
        }

    def load_from_dict(self, data):
        """Create GameState from a dictionary"""
        assert self.rows == data["rows"] and self.cols == data["cols"], (
            "Loaded dimensions do not match the initialized dimensions."
        )
        self.next_player = data["next_player"]
        self.filled_vertical = np.array(data["filled_vertical"], dtype=np.bool_)
        self.filled_horizontal = np.array(data["filled_horizontal"], dtype=np.bool_)
        self.boxes_by_player = np.array(data["boxes_by_player"], dtype=np.bool_)


@dataclasses.dataclass
class Game:
    rows: int
    cols: int

    def __post_init__(self):
        self.state = GameState(self.rows, self.cols)

    def print_state(self):
        """Print the current game state"""
        print(f"Current Player: {self.game_state.next_player + 1}")
        print("\nVertical edges filled:")
        print(self.game_state.filled_vertical.astype(int))
        print("\nHorizontal edges filled:")
        print(self.game_state.filled_horizontal.astype(int))
        print("\nPlayer 1 boxes:")
        print(self.game_state.boxes_by_player[0].astype(int))
        print("\nPlayer 2 boxes:")
        print(self.game_state.boxes_by_player[1].astype(int))

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
        valid_vertical = ~self.state.filled_vertical.flatten()
        valid_horizontal = ~self.state.filled_horizontal.flatten()
        return np.where(np.concatenate((valid_vertical, valid_horizontal)))[0]

    def _is_box_completed(self, r: int, c: int) -> bool:
        """Check if a box is completed at the given coordinates.

        The (r, c) coordinates refer to the top-left corner of the box.

        If the coordinates are out of bounds, return False.
        """
        if r < 0 or r >= self.rows - 1 or c < 0 or c >= self.cols - 1:
            return False
        return (
            self.state.filled_vertical[r, c]
            and self.state.filled_horizontal[r, c]
            and self.state.filled_vertical[r, c + 1]
            and self.state.filled_horizontal[r + 1, c]
        )

    def move(self, action: int) -> None:
        """Make a move in the game"""
        # Check if the action is valid
        if action not in self.valid_actions:
            raise ValueError("Invalid action")
        r, c, direction = self._action_to_coordinates(action)

        # Mark the edge as filled in the game state
        if direction == Direction.VERTICAL:
            self.state.filled_vertical[r, c] = True
        else:
            self.state.filled_horizontal[r, c] = True

        # Check if this move completes any boxes
        boxes_completed = False
        
        # For a vertical line at (r, c), check boxes to the left and right
        if direction == Direction.VERTICAL:
            # Check box to the left (r, c-1)
            if c > 0 and self._is_box_completed(r, c - 1):
                self.state.boxes_by_player[self.state.next_player, r, c - 1] = True
                boxes_completed = True
            # Check box to the right (r, c)
            if c < self.cols - 1 and self._is_box_completed(r, c):
                self.state.boxes_by_player[self.state.next_player, r, c] = True
                boxes_completed = True
        else:  # Horizontal line
            # Check box above (r-1, c)
            if r > 0 and self._is_box_completed(r - 1, c):
                self.state.boxes_by_player[self.state.next_player, r - 1, c] = True
                boxes_completed = True
            # Check box below (r, c)
            if r < self.rows - 1 and self._is_box_completed(r, c):
                self.state.boxes_by_player[self.state.next_player, r, c] = True
                boxes_completed = True

        # Switch to the next player only if no boxes were completed
        if not boxes_completed:
            self.state.next_player = 1 - self.state.next_player