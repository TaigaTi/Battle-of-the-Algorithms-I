from config import config
from collections import deque
import heapq
import random

GRID_SIZE = config.GRID_SIZE

#hold history so snake doesn't death loop
recent_positions = deque(maxlen=30)

def get_next_move(board_state, player_state, opponent_state):
    global recent_positions

    #board info
    width = board_state["cols"]
    height = board_state["rows"]
    food = board_state["food_locations"]
    obstacles = board_state["obstacle_locations"]
    
    #snake info
    body = {(bod["row"], bod["col"]) for bod in player_state["body"]}
    head = (player_state["head_position"]["row"],player_state["head_position"]["col"])
    tail = (player_state["body"][-1]["row"], player_state["body"][-1]["col"])
    snake_length = len(player_state["body"])
    opp = {(bod["row"], bod["col"]) for bod in opponent_state["body"]}
    opp_head = (opponent_state["head_position"]["row"],opponent_state["head_position"]["col"])
    opp_tail = (opponent_state["body"][-1]["row"], opponent_state["body"][-1]["col"])
    direction = player_state["direction"]
    avoid = body | opp | set(obstacles) 

    #track recent head positions to see if the snake is constantly looping the same area
    recent_positions.append(head)
    looped = list(recent_positions).count(head) >= 3

    #possible moves
    directions = {
        "up":    (-1,  0),
        "down":  ( 1,  0),
        "left":  ( 0, -1),
        "right": ( 0,  1)
    }
    
    # Avoid any moves that are non valid (opposite or out of map moves)
    def opposite(move,last_move):
        return (move,last_move) in [("left", "right"), ("right", "left"), ("up","down"), ("down","up")]
    def in_map(direction):
        row,col = direction
        return 0 <= row < height and 0 <= col < width
    def is_safe(direction):
        return in_map(direction) and direction not in avoid
    
    #Using a breadth first search to check if the tail is reachable
    def bfs(init, tail, danger):
        queue = deque([init])
        visited = set()
        while queue:
            cur = queue.popleft()
            if cur == tail:
                return True
            for d in directions.values():
                nr, nc = cur[0] + d[0], cur[1] + d[1]
                neighbor = (nr, nc)
                if (
                    in_map(neighbor) and
                    neighbor not in visited and
                    (neighbor not in avoid or neighbor == tail)
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False
    
    # the A* Heuristic 
    def heuristic(a,b):
        return abs(a[0] - b[0] + abs(a[1] - b[1]))

    #Find full path from start position to food 
    def path_finder(start, goal, avoid, tail):
        open_set = []
        #Heap would just be a tuple of cost to current position + estimated cost to goal, cost to current position, current position and path
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, []))
        visited = set()

        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]
            if current == goal:
                return path[1:]  # skip current head

            for d in directions.values():
                nr, nc = current[0] + d[0], current[1] + d[1]
                neighbor = (nr, nc)
                if (
                    in_map(neighbor) and
                    (neighbor not in avoid or neighbor == tail) and
                    neighbor not in visited
                ):
                    heapq.heappush(open_set, (
                        cost + 1 + heuristic(neighbor, goal),
                        cost + 1,
                        neighbor,
                        path
                    ))
        return []

    #Avoid areas bigger than the snake 
    def flood_fill(init, avoid, max_limit):
        queue = deque([init])
        visited = set()
        while queue and len(visited) < max_limit:
            cur = queue.popleft()
            if cur in visited or cur in avoid:
                continue
            if not (in_map(cur)):
                continue
            # if not bfs(cur,tail,avoid):
            #     continue
            visited.add(cur)

            for d in directions.values():
                queue.append((cur[0] + d[0], cur[1] + d[1]))

        return len(visited)


    #Gets possible safe moves out of options
    def all_safe_moves():
        safe = []
        for move, dir in directions.items():
            nr,nc = head[0] + dir[0], head[1] + dir[1]
            next_pos = (nr,nc)
            area = flood_fill(next_pos,avoid,snake_length*2)
            if is_safe(next_pos) and area > snake_length and not opposite(move,direction):
                safe.append((move,next_pos))
        return safe

    # # Sort food by distance
    # food_targets = sorted([(heuristic(head, (f[0], f[1])), (f[0], f[1])) for f in food])
    closer_food = []
    for f in food:
        food_pos = f
        my_path = path_finder(head, food_pos, avoid, tail)
        opp_avoid = opp | body | set(obstacles)  # Adjusted avoid for opponent
        opp_path = path_finder(opp_head, food_pos, opp_avoid, opp_tail)
        my_dist = len(my_path) if my_path else float('999')
        opp_dist = len(opp_path) if opp_path else float('999')
        next_pos = my_path[0] if my_path else None 
        danger_blocks = set(avoid)
        danger_blocks.add(next_pos) if next_pos else None
        if my_dist < opp_dist and bfs(next_pos, tail, danger_blocks):
            closer_food.append((my_dist, food_pos))


    food_targets = sorted(closer_food) if closer_food else None
    best_food = food_targets[0][1] if food_targets else None 
    safe_moves = all_safe_moves()

    ## Randomly jump out of loop to progress 
    if looped and safe_moves:
        print("looped")
        return random.choice(safe_moves)[0]



    food_path = path_finder(head, best_food, avoid, tail) if best_food else None
    
    #Head in direction of the nearest food if the move puts you in a position where you can get back to your tail
    if food_path:
        next_pos = food_path[0]
        danger_blocks = set(avoid)
        danger_blocks.add(next_pos)
        if bfs(next_pos, tail, danger_blocks):
            for move, d in directions.items():
                if (head[0] + d[0], head[1] + d[1]) == next_pos:
                    area = flood_fill(next_pos,avoid,snake_length*2)
                    if is_safe(next_pos) and area > snake_length and not opposite(move,direction):
                        return move

    #If no food is safe then just go back towards the tail
    # print("chase tail")
    for move, next_pos in safe_moves:
        if bfs(next_pos, tail, avoid):
            return move
    
    #No path to food or path to tail, choose a safe path and hope for the best
    if safe_moves:
        return random.choice(safe_moves)[0]

    #Wallahi no safe move found, GG
    print("Good Game")
    return direction

def set_player_name():
    return "Childz"