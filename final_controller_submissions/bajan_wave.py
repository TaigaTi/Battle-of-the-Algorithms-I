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

#things i should do:
#add flood fill to find safe paths to food
#check whos closer and only go if ur closer
#add consideration for enemy head when bigger than me for head in flood fill
#add future sight where it takes into consideration the next moves of the enemy potentially

from config import config
from collections import deque

GRID_SIZE = config.GRID_SIZE

def get_next_move(board_state, player_state, opponent_state):
    
    """
    A very basic snake controller. It always tries to move right, unless
    that's not a safe move, in which case it tries down, then up, then left.
    If none of those are safe, it returns the current direction.

    This example uses the game's GRID_SIZE for all moves and checks, ensuring
    compatibility regardless of the configured grid.

    Args:
        board_state (dict): Information about the game board.
        player_state (dict): Information about the current player's snake.
        opponent_state (dict): Information about the opponent's snake.

    Returns:
        str: The next direction ("left", "right", "up", "down").
    """
    head_x = player_state["head_position"]["row"]
    head_y = player_state["head_position"]["col"]
    enemy_head_x = opponent_state["head_position"]["row"]
    enemy_head_y = opponent_state["head_position"]["col"]
    direction = player_state["direction"]
    body = player_state["body"] # Create a set of obstacles from the snake's body for quick lookup  
    enemy_body = opponent_state["body"]  # Create a set of obstacles from the opponent's snake body for quick lookup
    max_rows = board_state["rows"]
    max_cols = board_state["cols"]
    food = set(board_state["food_locations"])
    larger = len(body) > len(enemy_body)

    directions = [
            (0, 1, "right"),
            (0, -1, "left"),
            (-1, 0, "up"),
            (1, 0, "down")
    ]
  
    obstacles = set()
    for segment in body:
        obstacles.add((segment["row"], segment["col"])) 

    for segment in enemy_body:
        obstacles.add((segment["row"], segment["col"]))

    

    
    obstacles.add((head_x, head_y))  # Add the head position to obstacles to prevent self-collision
    obstacles.add((enemy_head_x, enemy_head_y))  # Add the enemy head position to obstacles to prevent collision with opponent
    for obstacle in board_state["obstacle_locations"]:
        obstacles.add(obstacle)  # Add board obstacles
    

    
  
    
    def is_safe_move(x, y):
        """Checks if a move to (x, y) is safe (not out of bounds or self-collision)."""
        if x < 0 or x >= board_state["rows"] or y < 0 or y >= board_state["cols"]:
            return False
        if(x,y) in obstacles:
            return False
        return True
    
    
    def flood_fill(start, obstacles):
        """Flood fill algorithm to find all reachable cells from the start position."""
        queue = deque([start])
        visited = set()
        visited.add(start)
        
        while queue:
            x, y = queue.popleft()
            for dx, dy, dir_name in directions:
                new_x, new_y = x + dx, y + dy
                if is_safe_move(new_x, new_y) and (new_x, new_y) not in visited:
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y))
        return visited
    
    def annotated_distance_map(start, obstacles):
        """
        Modified BFS that records both:
        1. Distance to each reachable tile
        2. The first move direction needed to get there from the start

        Returns:
            dict: {(x, y): (distance, first_move_direction)}
        """
        queue = deque()
        distances = {}
        
        for dx, dy, dir_name in directions:
            new_x, new_y = start[0] + dx, start[1] + dy
            if is_safe_move(new_x, new_y):
                queue.append(((new_x, new_y), 1, dir_name))
                distances[(new_x, new_y)] = (1, dir_name)
        
        while queue:
            (x, y), dist, first_move = queue.popleft()

            for dx, dy, _ in directions:
                new_x = x + dx
                new_y = y + dy
                if is_safe_move(new_x, new_y) and (new_x, new_y) not in distances:
                    distances[(new_x, new_y)] = (dist + 1, first_move)
                    queue.append(((new_x, new_y), dist + 1, first_move))

        return distances

    
    
    fastest_path_map = annotated_distance_map((head_x, head_y), obstacles)
    enemy_fastest_path_map = annotated_distance_map((enemy_head_x, enemy_head_y), obstacles)
    #cache your flood fill results by looping through directions and checking if its a safe move if it is then flood fill in that direction and save it
    flood_fill_cache = {}
    for dx, dy, dir_name in directions:
        new_x = head_x + dx
        new_y = head_y + dy
        
        if is_safe_move(new_x, new_y):
            flood_fill_cache[dir_name] = len(flood_fill((new_x, new_y), obstacles))
        else:
            flood_fill_cache[dir_name] = -1 #this ensures that once there is a flood fill value that isn't -1 the move is safe

    kill_tiles = set()
    enemy_pos = (enemy_head_x, enemy_head_y)
    enemy_dist = fastest_path_map.get(enemy_pos, (float('inf'),))[0]

    is_much_bigger = len(body) > len(enemy_body) + 5
    is_bigger_and_closer = (len(body) + 3 > len(enemy_body)) and enemy_dist < 10
    is_slightly_bigger_and_very_close = (len(body) > len(enemy_body) + 1) and enemy_dist < 3

    should_attack = is_much_bigger or is_bigger_and_closer or is_slightly_bigger_and_very_close
    if should_attack:
        for dx, dy, dir_name in directions:
            target_x = enemy_head_x + dx
            target_y = enemy_head_y + dy
            if 0 <= target_x < max_rows and 0 <= target_y < max_cols:
                if flood_fill_cache[dir_name] > len(body) + 3:
                    kill_tiles.add((target_x, target_y))

        # Try to find a path to any kill tile
        kill_path = []
        if kill_tiles:
            closest_kill_tile = min(kill_tiles, key=lambda tile: fastest_path_map[tile][0] if tile in fastest_path_map else float('inf'))
            if closest_kill_tile in fastest_path_map:
                kill_path = [fastest_path_map[closest_kill_tile][1]]  # Get the first move direction to the closest kill tile

        if kill_path:
            # Check that this move doesn't trap us
            first_dir = kill_path[0]
            for dx, dy, dir_name in directions:
                if dir_name == first_dir:
                    new_head = (head_x + dx, head_y + dy)
                    if flood_fill_cache[first_dir] > len(body) + 3:
                        return first_dir
    safefood = set()
    #before i make my food object, i should check if the enemy is closer to the food than me
    
    #writing code that copies the above logic for food using the distance map
    for food_item in food:
        if food_item in fastest_path_map:
            my_dist, my_first_move = fastest_path_map[food_item]
            enemy_dist, enemy_first_move = enemy_fastest_path_map[food_item] if food_item in enemy_fastest_path_map else (float('inf'), None)

            # Prioritize if you're closer, or tied but bigger
            if my_dist < enemy_dist or (my_dist == enemy_dist and len(body) >= len(enemy_body)):
                safefood.add(food_item)

    ideal_path = []
    ideal_distance = -1
    if safefood:
        # Use the distance map to find the closest food item
        closest_food = min(safefood, key=lambda food_item: fastest_path_map[food_item][0])
        ideal_path = [fastest_path_map[closest_food][1]]  # Get the first move direction to the closest food
        ideal_distance = fastest_path_map[closest_food][0]  # Capture the distance to the closest food
    # If you haven't found the ideal path, use the distance map to find the closest food item
    if not ideal_path and food: 
    
        closest_food = min(food, key=lambda food_item: fastest_path_map[food_item][0] if food_item in fastest_path_map else float('inf'))
        if closest_food in fastest_path_map:
            ideal_path = [fastest_path_map[closest_food][1]]
            ideal_distance = fastest_path_map[closest_food][0]  # Capture the distance to the closest food
    
    # If there's a path to food, follow it
    
    if ideal_distance > 0:
        # If there's a path to food, follow it
        next_move = ideal_path[0]
        for dx, dy, dir_name in directions:
            if next_move == dir_name:
                #gonna add some functionality that uses my bfs function to see if I can still get to my tale
                #the trick is to start from the next move and see if i can the last item in the body object
                tail_position = (body[-1]["row"], body[-1]["col"])
                if (tail_position in fastest_path_map or len(body)==0) and flood_fill_cache[next_move] > len(body) + 3:
                    return next_move
        return next_move


    #gonna add a worse case scenario where it loops through the direction and flood fills the board in that direction and picks the  first move that is bigger than body plus 3 and bigger than 10 if it exists if not it picks the move with the most space
    max_reachable = 0
    best_direction = None
    for dx, dy, dir_name in directions:
        new_x = head_x + dx
        new_y = head_y + dy
        
        if flood_fill_cache[dir_name] > 0:
            # Flood fill to see how much space is available in that direction
            reachable_count = flood_fill_cache[dir_name]+1  # +1 to include the head position
            if reachable_count >= len(body) + 4 and reachable_count >= 11:
                return dir_name
            if reachable_count > max_reachable:
                max_reachable = reachable_count
                best_direction = dir_name
    
    return best_direction
    

    

def set_player_name():
    """
    Sets the player's name.

    Returns:
        str: The name of the player as a string.
    """
    return "Bajan Wave"