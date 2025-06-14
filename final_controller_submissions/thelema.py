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


from config import config
from collections import defaultdict, deque



def set_player_name():
    return "solid_snake"


def reconstruct_path(cameFrom, current):
    total_path = [current]

    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.insert(0, current)

    return total_path

def get_neighbours(current, grid):
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    neighbours = []
    rows = len(grid)
    cols = len(grid[0])
    for d in directions:
        r, c = current[0] + d[0], current[1] + d[1]
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 1:
            neighbours.append((r, c))
    
    return neighbours

def flood_fill(start, grid, limit):

    queue = deque([tuple(start)])
    visited = set([tuple(start)])


    while queue:
        cur = queue.popleft()

        for neighbour in get_neighbours(cur, grid):
            if neighbour not in visited:
                queue.append(neighbour)
                visited.add(neighbour)

                if len(visited) >= limit:
                    return len(visited)

    
    return len(visited)


def heuristic(node, goal, grid):
    # Manhattan distance
    manhattan = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    open_neighbors = len(get_neighbours(node, grid))  # grid must be accessible here
    return manhattan - 1.2 * open_neighbors  # Tune 0.2 as needed

def d(a, b, grid):
    return 1

def a_star(start, goal, grid, longest=True):
    
    openSet = {start}

    cameFrom = {}

    gScore = defaultdict(lambda: float("inf"))
    gScore[start] = 0

    fScore = defaultdict(lambda: float("inf"))
    fScore[start] = heuristic(start, goal, grid)


    while openSet:
        current = min(openSet, key=lambda x: fScore[x])
        
        if current == goal:
            return reconstruct_path(cameFrom, current)

        openSet.remove(current)

       
        for neighbour in get_neighbours(current, grid):

            tentative_gScore = gScore[current] + d(current, neighbour, grid)

            if tentative_gScore < gScore[neighbour]:
                cameFrom[neighbour] = current
                gScore[neighbour] = tentative_gScore
                fScore[neighbour] = tentative_gScore + heuristic(neighbour, goal, grid)
                if neighbour not in openSet:
                    openSet.add(neighbour)

    return None  # No path found

def generate_grid(player_state, opponent_state, board_state, set_tail_walkable=False):

    head = player_state["head_position"]

    #if board_state["obstacle_locations"]:
    direction = player_state["direction"]

    obstacles = board_state["obstacle_locations"]
    fruits = board_state["food_locations"]
    body = player_state["body"]
    score = player_state["score"]


    opponent_body = opponent_state["body"]
    opponent_head = opponent_state["head_position"]
    opponent_score = opponent_state["score"]


    rows = board_state["rows"]
    cols = board_state["cols"]

    grid = [[1 for c in range(cols)] for r in range(rows)]

    for ob in obstacles:
        grid[ob[0]][ob[1]] = 0

    my_body = body[1:]

    if set_tail_walkable:
        my_body = body[:-1]

    for b in my_body:
        grid[b["row"]][b["col"]] = 0

    for o in opponent_body:
        grid[o["row"]][o["col"]] = 0  

    return grid


def print_grid(grid, start=None, goal=None):
    for r, row in enumerate(grid):
        line = ""
        for c, val in enumerate(row):
            if start and (r, c) == start:
                line += "S "
            elif goal and (r, c) == goal:
                line += "G "
            elif val == 1:
                line += ". "
            else:
                line += "# "
        print(line)
    print()

def get_next_move(board_state, player_state, opponent_state):
    head = player_state["head_position"]

    #if board_state["obstacle_locations"]:
    direction = player_state["direction"]

    obstacles = board_state["obstacle_locations"]
    fruits = board_state["food_locations"]
    body = player_state["body"]
    score = player_state["score"]


    opponent_body = opponent_state["body"]
    opponent_head = opponent_state["head_position"]
    opponent_score = opponent_state["score"]

    rows = board_state["rows"]
    cols = board_state["cols"]
      


    normal_grid = generate_grid(player_state, opponent_state, board_state)
    tail_grid = generate_grid(player_state, opponent_state, board_state, True)

    
    
    start = (head["row"], head["col"])

    
    def is_safe_move(r, c):
        """Checks if a move to (r, c) is safe (not out of bounds or self-collision)."""
        opponent_head = opponent_state["head_position"]

        if r < 0 or r >= board_state["rows"] or c < 0 or c >= board_state["cols"]:
            return False
        for segment in body[:-1]:
            if segment["row"] == r and segment["col"] == c:
                return False

        for segment in opponent_body:
            if segment["row"] == r and segment["col"] == c:
                return False

        if score <= opponent_score:
            for neighbour in get_neighbours((opponent_head["row"], opponent_head["col"]), normal_grid):
                if neighbour == (r, c):
                    return False

        neighbours = get_neighbours((r, c), normal_grid)
        if not neighbours:
            return False

        for o in obstacles:
            if o[0] == r and o[1] == c:
                return False

        
        return True

    
    def fallback(direction, start, grid):
        head_row, head_col = start

        best_move = direction
        most_space = -1

        tail = (body[-1]["row"], body[-1]["col"])
        
        for move, dr, dc, forbidden in [("up",    -1,  0, "down"), ("down",   1,  0, "up"),("left",   0, -1, "right"),("right",  0,  1, "left")]:
            nr, nc = head_row + dr, head_col + dc
            if direction != forbidden and is_safe_move(nr, nc):
                new_grid = grid
                new_grid[head_row][head_col] = 0

                space = flood_fill((nr, nc), new_grid, len(body) * 3)  * 100 - (abs(nr - tail[0]) + abs(nc - tail[1]))
                if space > most_space:
                    best_move = move
                    most_space = space
        
        
        return best_move


    criteria = {"row": head["row"], "col": head["col"]}
    best_fruit_distance = 10000

    path = None
    ans = None
    goal = (fruits[-1][0], fruits[-1][0])

    
    penalty_body = -0.5
    penalty_head = -1.4
    edge_penalty = -1.0

    count = 0
    for x in fruits:
        edge_dist = min(x[0], rows - 1- x[0], x[1], cols - 1 - x[1], x[1])
        d = (
            2.0* (abs(x[0] - criteria["row"]) + abs(x[1] - criteria["col"]))
            + penalty_head * (abs(x[0] - opponent_head["row"]) + abs(x[1] - opponent_head["col"]))
            + penalty_body * min(
                abs(seg["row"] - x[0]) + abs(seg["col"] - x[1])
                for seg in opponent_body
            )

            + edge_penalty * (0 if edge_dist > 3 else 1)
        )
        
        
        if d < best_fruit_distance and len(get_neighbours((x[0], x[1]), normal_grid)) >= 3:
            best_fruit_distance = d
            goal = (x[0], x[1])
            

    

    if score - opponent_score > 20:
        path = a_star(start, (body[-1]["row"], body[-1]["col"]), tail_grid)
        
        if path and len(path) > 1:
            ans = path[1]
        else:
            path = a_star(start, goal, normal_grid)
        
            if path and len(path) > 1:
                ans = path[1]

                possible_path = a_star(ans, (body[-1]["row"], body[-1]["col"]), tail_grid)
                
                if possible_path:
                    ans = path[1]
                else:
                    return fallback(direction, start, normal_grid)

    elif score > 10:
        if best_fruit_distance < 0:
            path = a_star(start, goal, normal_grid)
        
        if path and len(path) > 1 and len(get_neighbours(path[1], normal_grid)) >= 3:
            ans = path[1]

            possible_path = a_star(ans, (body[-1]["row"], body[-1]["col"]), tail_grid)
            
            if possible_path:
                ans = path[1]
            else:
                return fallback(direction, start, normal_grid)

        else:
            path = a_star(start, (body[-1]["row"], body[-1]["col"]), tail_grid)
            if path and len(path) > 1:
                ans = path[1]
            else:
                return fallback(direction, start, normal_grid)
    else:
        if best_fruit_distance < 100:
            path = a_star(start, goal, normal_grid)
        
        if path and len(path) > 1:
            ans = path[1]
        else:
            return fallback(direction, start, normal_grid)

    if not ans:
        return fallback(direction, start, normal_grid)

    if (start[0] + -1, start[1] + 0) == ans and is_safe_move(start[0] - 1, start[1]):
        if direction != "down": # accounting for snake bumping into itself
            direction = "up"
    elif (start[0] + 1, start[1] + 0) == ans and is_safe_move(start[0] + 1, start[1]):
        if direction != "up": # accounting for snake bumping into itself
            direction = "down"
    elif (start[0] + 0, start[1] + -1) == ans and is_safe_move(start[0], start[1] - 1):
        if direction != "right": # accounting for snake bumping into itself
            direction = "left"
    elif (start[0] + 0, start[1] + 1) == ans and is_safe_move(start[0], start[1] + 1):
        if direction != "left": # accounting for snake bumping into itself
            direction = "right"
    
    

    

    return direction