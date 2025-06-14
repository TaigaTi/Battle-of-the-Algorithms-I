# controllers/joshua_algorithm.py
"""
Joshua Layne's Algorithm
This module implements Joshua's algorithm for determining the next move in a snake game.

"""

from collections import deque
from config import config

GRID_SIZE = config.GRID_SIZE

def set_player_name():
    """
    Set the player's name for the game. 
    This function is called once at the start of the game and returns a string.
    
    Returns:
        str: The name of the player.
    """
    return "Joshua"

def get_next_move(board_state, player_state, opponent_state):
    """get_next_move 
    
    This is the function that determines the next move for the snake.
    It uses a heuristic approach to evaluate potential moves based on food proximity, the opponent's position, and available space.
    It tries to avoid moving into obstacles, the opponent's body, or its own body.

    Args:
        board_state (dict): Information about the game board.
        player_state (dict): Information about the current player's snake.
        opponent_state (dict): Information about the opponent's snake.

    Returns:
        str: The next direction ("left", "right", "up", "down").
    """

    # Extract the player's head position, direction, and body segments
    head = (player_state["head_position"]["row"], player_state["head_position"]["col"])
    direction = player_state["direction"]
    body = player_state["body"]

    # Extract board dimensions and food locations
    rows, cols = board_state["rows"], board_state["cols"]
    food_list = board_state.get("food_locations", [])

    # Convert obstacle locations to a set for quick lookup
    obstacles = set(tuple(pos) for pos in board_state["obstacle_locations"])
    
    # Extract opponent's head position and body segments
    opponent_head = (opponent_state["head_position"]["row"], opponent_state["head_position"]["col"])
    opponent_body = set((seg["row"], seg["col"]) for seg in opponent_state["body"])
    
    # Stores the player's body segments excluding the head and converts it to a set for quick lookup
    self_body = set((seg["row"], seg["col"]) for seg in body[1:])
    
    # Combine obstacles, opponent's body, and player's body into a single set of blocked positions
    blocked = obstacles | opponent_body | self_body
    
    # Extract recent positions from player state, defaulting to an empty list if not present
    recent_positions = player_state.get("recent_positions", [])
    # Detect looping: check if the head has visited the same spots recently
    looping = recent_positions.count(head) > 1  # been here before recently


    # Sets the opposite directions
    opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
    
    # Define possible move directions
    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    # Function to check if a position is within bounds and not blocked
    def is_safe(pos):
        """
        Determines whether a given grid position is a safe and valid move for the snake.

        A position is considered safe if:
        - It lies within the bounds of the game board.
        - It is not occupied by the snake's own body, the opponent's body, or any static obstacles.

        
        Args:
            pos (tuple): The position to check (row, col).
            
        Returns:
            bool: True if the position is safe to move into, False otherwise.
        """	
        # row and column extracted from pos
        r, c = pos
        return 0 <= r < rows and 0 <= c < cols and pos not in blocked

    # Flood-fill algorithm to count available space around a position
    def flood_fill(start, limit=100):
        """
        Estimates the number of safe tiles accessible from a starting position using a flood fill algorithm.

        This function performs a breadth-first search (BFS) from the given starting position,
        counting how many safe grid cells are reachable without crossing into blocked or out-of-bounds areas.
        The search is capped by an optional limit to ensure performance.

        Args:
            start (tuple): A (row, col) tuple indicating the starting position for the flood fill.
            limit (int, optional): Maximum number of tiles to explore before stopping. Defaults to 100.

        Returns:
            int: The number of reachable safe cells from the starting position, up to the given limit.
        """
        
        # Stores visited positions in a set
        visited = set()
        
        # Queue for BFS
        q = deque([start])
        count = 0
        
        # Directions for moving in the grid
        while q and count < limit:
            # Pop the next position from the queue as row and column
            r, c = q.popleft()
            
            # If the position has been visited or is not safe, skip it
            if (r, c) in visited or not is_safe((r, c)):
                continue
            
            # Mark the position as visited
            visited.add((r, c))
            
            # Increment the count of reachable safe cells
            count += 1
            
            # Add all four possible moves (up, down, left, right) to the queue	
            for dr, dc in directions.values():
                q.append((r + dr, c + dc))
        return count

    def manhattan(a, b):
        """
        Calculates the Manhattan distance between two positions on a grid.

        The Manhattan distance is the sum of the absolute differences of their row and column indices.

        Args:
            a (tuple): The first position as a (row, col) tuple.
            b (tuple): The second position as a (row, col) tuple.

        Returns:
            int: The Manhattan distance between the two positions.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_closest_food_to(pos):
        """
        Finds the food item closest to the given position based on Manhattan distance.

        If there are no food items available, the function returns None.

        Args:
            pos (tuple): The (row, col) position from which to measure distance.

        Returns:
            tuple or None: The (row, col) position of the closest food item, or None if no food exists.
        """
        
        # Checks if food_list is empty
        if not food_list:
            return None
        
        # Else, returns the food item with the minimum Manhattan distance to the given position
        return min(food_list, key=lambda f: manhattan(pos, f))

    # Targets
    my_target = get_closest_food_to(head)
    opponent_target = get_closest_food_to(opponent_head)
    
    # Check if both snakes are targeting the same food
    going_for_same_food = my_target == opponent_target and my_target is not None

    # Sets move options
    move_options = []

    # Iterate through possible moves
    for move, (dr, dc) in directions.items():
        # Skip if the move is the opposite of the current direction
        if move == opposites.get(direction):
            continue

        # Calculate the new position after the move
        new_pos = (head[0] + dr, head[1] + dc)
        
        # Momentum bonus: prefer to continue in the same direction
        momentum_bonus = 1 if move == direction else 0
        
        # Skip if the new position is not safe
        if not is_safe(new_pos):
            continue

        # Dead-end check: if the new position is surrounded by obstacles or the opponent's body
        min_required_space = len(body)
        if flood_fill(new_pos) < min_required_space:
            continue

        # Heuristic: move toward food
        food_score = manhattan(new_pos, my_target) if my_target else 0

        # Heuristic: try to block opponent if targeting same food
        block_score = 0
        # If both snakes are going for the same food, prefer to block the opponent
        if going_for_same_food:
            block_score = manhattan(new_pos, opponent_target)  # prefer lower

        # Heuristic: chase opponent head if we're longer
        chase_score = 0
        if (len(body) > 15 and len(body) >= len(opponent_state["body"])):
            chase_score = manhattan(new_pos, opponent_head)

        # Flood-fill for safety
        space = flood_fill(new_pos)
        
        # To give a boost to moves that create more space
        escape_bonus = space if looping else 0
        
        # Loop penalty to avoid circling
        loop_penalty = recent_positions.count(new_pos)

        # Append move with compound score
        move_options.append((
            move,
            food_score,
            block_score,
            chase_score,
            -space,  # prefer more space
            -2 * loop_penalty,  # discourage repeating recent positions
            -momentum_bonus,  # momentum bonus for continuing in the same direction
            -escape_bonus  # escape bonus encourages breaking loop
        ))

    # Sort by: (food_score, block_score, chase_score, -space, -loop_penalty, -momentum_bonus, -escape_bonus)
    move_options.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[5], x[6], x[7]))

    # If there are valid move options, choose the best one
    if move_options:
        # Track recent positions to prevent circling
        recent_positions.append(head)
        if len(recent_positions) > 10:
            recent_positions.pop(0)
        player_state["recent_positions"] = recent_positions
        
        return move_options[0][0]

    # fallback: any safe move
    for move, (dr, dc) in directions.items():
        new_pos = (head[0] + dr, head[1] + dc)
        if is_safe(new_pos):
            return move

    return direction
