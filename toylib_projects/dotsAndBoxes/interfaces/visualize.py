"""Utilities to visuzalize the game state of Dots and Boxes in colab/jupyter."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .. import dots_and_boxes


def plot_dots_and_boxes(game_state: dots_and_boxes.DotsAndBoxesGame):
    """
    Plot a Dots and Boxes game state using matplotlib.

    Parameters:
    game_state (dict): Dictionary containing the game state with keys:
        - state: dict with rows, cols, next_player, filled_vertical, filled_horizontal,
                 boxes_by_player, and line_owners
    """
    state = game_state.to_dict()
    rows = state["rows"]
    cols = state["cols"]

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(cols * 1.5, rows * 1.5))

    # Set axis limits and properties
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Invert y-axis to match typical game representation
    ax.axis("off")

    # Define colors for players
    player_colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#95E1D3",
        "#F6D186",
    ]  # Red, Teal, Mint, Yellow
    line_width = 3
    dot_size = 200

    # Draw dots
    for i in range(rows):
        for j in range(cols):
            ax.scatter(j, i, s=dot_size, c="black", zorder=3)

    # Draw horizontal lines
    for i in range(rows + 1):
        for j in range(cols):
            if i < len(state["filled_horizontal"]) and j < len(
                state["filled_horizontal"][i]
            ):
                if state["filled_horizontal"][i][j]:
                    # Get line owner
                    owner = state["line_owners"]["horizontal"][i][j]
                    color = player_colors[owner] if owner >= 0 else "black"
                    ax.plot(
                        [j, j + 1], [i, i], color=color, linewidth=line_width, zorder=2
                    )
                else:
                    # Draw unfilled line as dashed gray
                    ax.plot(
                        [j, j + 1],
                        [i, i],
                        color="lightgray",
                        linewidth=line_width / 2,
                        linestyle="--",
                        zorder=1,
                    )

    # Draw vertical lines
    for i in range(rows):
        for j in range(cols + 1):
            if i < len(state["filled_vertical"]) and j < len(
                state["filled_vertical"][i]
            ):
                if state["filled_vertical"][i][j]:
                    # Get line owner
                    owner = state["line_owners"]["vertical"][i][j]
                    color = player_colors[owner] if owner >= 0 else "black"
                    ax.plot(
                        [j, j], [i, i + 1], color=color, linewidth=line_width, zorder=2
                    )
                else:
                    # Draw unfilled line as dashed gray
                    ax.plot(
                        [j, j],
                        [i, i + 1],
                        color="lightgray",
                        linewidth=line_width / 2,
                        linestyle="--",
                        zorder=1,
                    )

    # Fill completed boxes
    for i in range(rows):
        for j in range(cols):
            # Check which player owns the box
            for player_idx, player_boxes in enumerate(state["boxes_by_player"]):
                if i < len(player_boxes) and j < len(player_boxes[i]):
                    if player_boxes[i][j]:
                        # Draw filled box
                        rect = patches.Rectangle(
                            (j + 0.1, i + 0.1),
                            0.8,
                            0.8,
                            facecolor=player_colors[player_idx],
                            alpha=0.3,
                            zorder=0,
                        )
                        ax.add_patch(rect)
                        # Add player number in center of box
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            str(player_idx + 1),
                            ha="center",
                            va="center",
                            fontsize=16,
                            fontweight="bold",
                        )

    # Add title with game info
    title = f"Dots and Boxes ({rows}x{cols}) - Next Player: {state['next_player']}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add legend
    legend_elements = []
    for i in range(min(len(player_colors), 2)):  # Show legend for up to 2 players
        legend_elements.append(
            patches.Patch(color=player_colors[i], label=f"Player {i}")
        )
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2
    )

    plt.tight_layout()
    plt.show()


def plot_tree(
    root,
    max_depth=3,
    max_children_per_node=5,
    max_state_length=10,
    figsize=(15, 10),
    node_size=0.8,
    font_size=8,
    show_state=False,  # Changed default to False since we're showing wins/visits
):
    """
    Plot a tree visualization with configurable parameters.

    Parameters:
    - root: Tree object with root TreeNode
    - max_depth: Maximum depth to visualize (default: 3)
    - max_children_per_node: Maximum children to show per node (default: 5)
    - max_state_length: Maximum elements of state array to display (default: 10)
    - figsize: Figure size tuple (default: (15, 10))
    - node_size: Size of nodes (default: 0.8)
    - font_size: Font size for text (default: 8)
    - show_state: Whether to show state values in nodes (default: False)
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate positions for nodes
    positions = {}

    def calculate_positions(node, depth=0, x=0, parent_x=0, width=10):
        """Calculate x,y positions for each node"""
        if depth > max_depth:
            return

        positions[id(node)] = (x, -depth)

        # Limit children to display
        children_to_show = node.children[:max_children_per_node]
        num_children = len(children_to_show)

        if num_children > 0:
            # Calculate spacing for children
            child_width = width / max(num_children, 1)
            start_x = x - (width / 2) + (child_width / 2)

            for i, child in enumerate(children_to_show):
                child_x = start_x + i * child_width
                calculate_positions(child, depth + 1, child_x, x, child_width * 0.8)

    # Start position calculation from root
    calculate_positions(root)

    # Draw edges first (so they appear behind nodes) with action labels
    def draw_edges(node, depth=0):
        if depth >= max_depth:
            return

        if id(node) in positions:
            node_pos = positions[id(node)]
            children_to_show = node.children[:max_children_per_node]

            for child in children_to_show:
                if id(child) in positions:
                    child_pos = positions[id(child)]
                    # Draw edge
                    ax.plot(
                        [node_pos[0], child_pos[0]],
                        [node_pos[1], child_pos[1]],
                        "k-",
                        alpha=0.6,
                        linewidth=1,
                    )
                    
                    # Add action label on edge
                    if hasattr(child, 'action') and child.action >= 0:
                        mid_x = (node_pos[0] + child_pos[0]) / 2
                        mid_y = (node_pos[1] + child_pos[1]) / 2
                        ax.text(
                            mid_x,
                            mid_y,
                            str(child.action),
                            ha="center",
                            va="center",
                            fontsize=font_size - 2,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                            zorder=5
                        )
                    
                    draw_edges(child, depth + 1)

    draw_edges(root)

    # Draw nodes
    def draw_nodes(node, depth=0):
        if depth > max_depth:
            return

        if id(node) in positions:
            pos = positions[id(node)]

            # Create node circle
            circle = patches.Circle(
                pos,
                node_size / 2,
                facecolor="lightblue",
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(circle)

            # Add node text - show wins/visits or state
            if show_state and node.state is not None:
                # Truncate state array if too long
                state_display = node.state[:max_state_length]
                if len(node.state) > max_state_length:
                    state_text = f"[{', '.join(map(str, state_display))}...]"
                else:
                    state_text = f"[{', '.join(map(str, state_display))}]"

                # Split long text into multiple lines
                if len(state_text) > 20:
                    # Find a good break point
                    mid = len(state_text) // 2
                    comma_pos = state_text.find(",", mid)
                    if comma_pos != -1:
                        line1 = state_text[: comma_pos + 1]
                        line2 = state_text[comma_pos + 1 :]
                        ax.text(
                            pos[0],
                            pos[1],
                            f"{line1}\n{line2}",
                            ha="center",
                            va="center",
                            fontsize=font_size - 1,
                            wrap=True,
                        )
                    else:
                        ax.text(
                            pos[0],
                            pos[1],
                            state_text,
                            ha="center",
                            va="center",
                            fontsize=font_size - 1,
                        )
                else:
                    ax.text(
                        pos[0],
                        pos[1],
                        state_text,
                        ha="center",
                        va="center",
                        fontsize=font_size,
                    )
            else:
                # Show wins and visits
                if hasattr(node, 'wins') and hasattr(node, 'num_visits'):
                    wins_text = f"W: {node.wins}\nV: {node.num_visits}\nP: {node._game_state.next_player}"
                    ax.text(
                        pos[0],
                        pos[1],
                        wins_text,
                        ha="center",
                        va="center",
                        fontsize=font_size,
                    )
                else:
                    # Fallback to depth
                    ax.text(
                        pos[0],
                        pos[1],
                        f"D{depth}",
                        ha="center",
                        va="center",
                        fontsize=font_size,
                    )

            # Recursively draw children
            children_to_show = node.children[:max_children_per_node]
            for child in children_to_show:
                draw_nodes(child, depth + 1)

    draw_nodes(root)

    # Add legend/info
    info_text = f"Max Depth: {max_depth} | Max Children: {max_children_per_node} | W: wins, V: visits"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    # Set axis properties
    ax.set_xlim(-8, 8)
    ax.set_ylim(-max_depth - 1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"MCTS Tree Visualization (Showing up to depth {max_depth})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig, ax


# Alternative compact version that shows more nodes
def plot_tree_compact(
    root,
    max_depth=4,
    max_children_per_node=3,
    max_state_length=5,
    figsize=(12, 8),
    show_indices=False,
):
    """
    A more compact tree visualization showing state indices or values.
    """
    fig, ax = plt.subplots(figsize=figsize)

    positions = {}

    def calculate_positions(node, depth=0, x=0, width=8):
        if depth > max_depth:
            return

        positions[id(node)] = (x, -depth * 1.5)

        children_to_show = node.children[:max_children_per_node]
        num_children = len(children_to_show)

        if num_children > 0:
            child_width = width / max(num_children, 1)
            start_x = x - (width / 2) + (child_width / 2)

            for i, child in enumerate(children_to_show):
                child_x = start_x + i * child_width
                calculate_positions(child, depth + 1, child_x, child_width * 0.7)

    calculate_positions(root)

    # Draw edges
    def draw_edges(node, depth=0):
        if depth >= max_depth:
            return

        if id(node) in positions:
            node_pos = positions[id(node)]
            children_to_show = node.children[:max_children_per_node]

            for child in children_to_show:
                if id(child) in positions:
                    child_pos = positions[id(child)]
                    ax.plot(
                        [node_pos[0], child_pos[0]],
                        [node_pos[1], child_pos[1]],
                        "k-",
                        alpha=0.5,
                        linewidth=0.8,
                    )
                    draw_edges(child, depth + 1)

    draw_edges(tree.root)

    # Draw nodes
    def draw_nodes(node, depth=0):
        if depth > max_depth:
            return

        if id(node) in positions:
            pos = positions[id(node)]

            # Smaller circles for compact view
            circle = patches.Circle(
                pos,
                0.3,
                facecolor="lightcoral" if depth == 0 else "lightblue",
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(circle)

            # Show abbreviated state
            if node.state is not None:
                if show_indices:
                    # Show non-zero indices
                    nonzero_idx = np.nonzero(node.state)[0][:max_state_length]
                    text = f"{list(nonzero_idx)}" if len(nonzero_idx) > 0 else "[]"
                else:
                    # Show first few values
                    values = node.state[:max_state_length]
                    text = f"{[f'{v:.1f}' if isinstance(v, float) else str(v) for v in values]}"

                ax.text(pos[0], pos[1], text, ha="center", va="center", fontsize=6)

            children_to_show = node.children[:max_children_per_node]
            for child in children_to_show:
                draw_nodes(child, depth + 1)

    draw_nodes(root)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-max_depth * 1.5 - 1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Compact Tree View (Depth {max_depth})", fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    return fig, ax
