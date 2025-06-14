# This is the Player 1 AI controller file.
# Contestants should implement their snake AI logic within this file.
# Your code will determine Player 1's next move based on the provided game state.
#
# You will need to implement two functions:
# 1. `get_next_move(board_state_data, player_state_data, opponent_state_data)`:
#    This function should return the next direction for the snake ("up", "down", "left", or "right").
# 2. `set_player_name()`:
#    This function should return a string representing the name of your AI player.
#
# For more detailed information on the expected input and output formats,
# please refer to the documentation at `docs/controller_api.md`.
# Happy Coding!

from typing import Literal
import heapq

def set_player_name() -> str:
    """Returns the AI player's display name."""
    return "SnekySnek"

def get_next_move(board_state: dict, player_state: dict, opponent_state: dict) -> Literal["left", "right", "up", "down"] | None:
    """
    Determines the next move for the snake using A* pathfinding.

    Parameters:
    - board_state (dict): Information about the game world.
    - player_state (dict): The player's snake data.
    - opponent_state (dict): The opponent's snake data.

    Returns:
    - One of: "left", "right", "up", or "down".
    """
    head = (player_state["head_position"]["row"], player_state["head_position"]["col"])
    food_locations = board_state["food_locations"]
    obstacles = set(board_state["obstacle_locations"])
    body_positions = [(segment["row"], segment["col"]) for segment in player_state["body"]]
    opponent_positions = [(segment["row"], segment["col"]) for segment in opponent_state["body"]]

    # Combine obstacles and other blocked positions
    blocked_positions = obstacles.union(body_positions).union(opponent_positions)

    if not food_locations:
        return player_state["direction"]  # No food available, keep moving

    # Find shortest path using A* search
    target = min(food_locations, key=lambda f: abs(head[0] - f[0]) + abs(head[1] - f[1]))
    path = a_star_search(head, target, blocked_positions, board_state["rows"], board_state["cols"])

    if not path:
        return player_state["direction"]  # No valid path, continue current direction
    
    # Determine next move based on path
    next_position = path[1]  # First step after the current position
    move_direction = get_direction(head, next_position)
    
    return move_direction

def a_star_search(start, goal, blocked_positions, rows, cols):
    """Performs A* pathfinding from start to goal."""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, blocked_positions, rows, cols):
            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def reconstruct_path(came_from, current):
    """Reconstructs path from goal to start."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def get_neighbors(position, blocked_positions, rows, cols):
    """Returns valid neighboring positions."""
    row, col = position
    neighbors = [
        (row, col - 1),  # Left
        (row, col + 1),  # Right
        (row - 1, col),  # Up
        (row + 1, col),  # Down
    ]
    return [n for n in neighbors if 0 <= n[0] < rows and 0 <= n[1] < cols and n not in blocked_positions]

def get_direction(start, end) -> Literal["left", "right", "up", "down"] | None:
    """Determines movement direction based on position change."""
    row_start, col_start = start
    row_end, col_end = end

    if col_end < col_start:
        return "left"
    elif col_end > col_start:
        return "right"
    elif row_end < row_start:
        return "up"
    elif row_end > row_start:
        return "down"

    return None  # Default fallback