from dataclasses import dataclass
from typing import List, Tuple, TypedDict, Optional, Set, NamedTuple
from enum import Enum
import collections
import heapq
from copy import deepcopy

class MoveEnum(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    @staticmethod
    def get_direction_vector(move_enum_val: Optional['MoveEnum']) -> Optional[Tuple[int, int]]: # Added Optional
        if move_enum_val == MoveEnum.UP:
            return (-1, 0)
        if move_enum_val == MoveEnum.DOWN:
            return (1, 0)
        if move_enum_val == MoveEnum.LEFT:
            return (0, -1)
        if move_enum_val == MoveEnum.RIGHT:
            return (0, 1)
        return None # If direction is None or invalid

# List of direction tuples
directions = [
    (-1, 0),    # UP
    (1, 0),   # DOWN
    (0, -1),    # LEFT
    (0, 1),   # RIGHT
]

# Ordered MoveEnum values
move_enum_values = [
    MoveEnum.UP,
    MoveEnum.DOWN,
    MoveEnum.LEFT,
    MoveEnum.RIGHT,
]

# Automatically generate direction_move_map using dictionary comprehension
direction_move_map = {direction: move for direction, move in zip(directions, move_enum_values)}

@dataclass
class BoardStateType(TypedDict):
    width: int  # Board width in pixels
    height: int  # Board height in pixels
    rows: int  # Board rows
    cols: int  # Board columns
    food_locations: List[Tuple[int,int]]  # List of food location indices
    obstacle_locations: List[Tuple[int,int]]  # List of obstacle location indices

@dataclass
class CoordinateObjectType(TypedDict):
    row: int
    col: int

@dataclass
class PlayerStateSimulationType(TypedDict):
    id: int
    head_position: tuple[int, int]
    body: List[tuple[int, int]]
    direction: MoveEnum
    score: int
    length: int

@dataclass
class PlayerStateType(TypedDict):
    id: int
    head_position: CoordinateObjectType
    body: List[CoordinateObjectType]
    direction: MoveEnum
    score: int
    length: int

class CoordinateConversionHelper:
    @staticmethod
    def Transform2DTupleToObject(coordinate_tuple) -> CoordinateObjectType:
        if len(coordinate_tuple) != 2:
            raise Exception("Tuple is not of length 2")
        return {
            "row": coordinate_tuple[0],
            "col": coordinate_tuple[1]
        }
    
    @staticmethod
    def TransformDictToTuple(coordinate_dict) -> tuple[int, int]:
        if "row" not in coordinate_dict or "col" not in coordinate_dict:
            raise Exception("Dictionary is not the right shape")
        return (coordinate_dict["row"], coordinate_dict["col"])

class DirectionHelper:
    @staticmethod
    def DetermineDirection(cur_position: tuple[int, int], next_position: tuple[int, int]):
        direction_vector = (next_position[0] - cur_position[0], next_position[1] - cur_position[1])
        return direction_move_map[direction_vector]
    
    @staticmethod
    def is_opposite_direction(direction1: MoveEnum, direction2: MoveEnum) -> bool:
        """
        Check if two directions are opposite.
        """
        vector1 = MoveEnum.get_direction_vector(direction1)
        if vector1 is None:
            raise Exception("Invalid direction provided.")
            

        opposite_vector = (vector1[0] * -1, vector1[1] * -1)

        vector2 = MoveEnum.get_direction_vector(direction2)
        if vector2 is None:
            raise Exception("Invalid direction provided.")
            
        return  vector2 == opposite_vector
    
    @staticmethod
    def get_next_position(head_position: tuple[int, int], direction: MoveEnum) -> tuple[int, int]:
        """
        Get the next position based on the current head position and direction.
        """
        direction_vector = MoveEnum.get_direction_vector(direction)
        if direction_vector is None:
            raise ValueError("Invalid direction provided.")
        
        return (head_position[0] + direction_vector[0], head_position[1] + direction_vector[1])

class SearchHelper:
    @staticmethod
    def _is_valid(r: int, c: int, rows: int, cols: int, obstacles: Set[Tuple[int, int]]) -> bool:
        return 0 <= r < rows and 0 <= c < cols and (r, c) not in obstacles

    @staticmethod
    def manhattan_distance(start: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points."""
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    @staticmethod
    def AStar_shortest_search_to_target(
        start: Tuple[int, int],
        target: Tuple[int, int],
        impassable_cells: Set[Tuple[int, int]],
        rows: int,
        cols: int
    ) -> Optional[List[Tuple[int, int]]]:
        
        if not SearchHelper._is_valid(start[0], start[1], rows, cols, impassable_cells) or \
           not SearchHelper._is_valid(target[0], target[1], rows, cols, impassable_cells):
            return None # Start or target is an obstacle/invalid

        if start == target:
            return [start] # Already at the target

        # priority_queue stores (f_cost, g_cost, (row, col))
        # f_cost = g_cost + h_cost (estimated total cost)
        # g_cost = actual cost from start to current node
        priority_queue = [(0 + SearchHelper.manhattan_distance(start, target), 0, start)]
        
        # came_from maps node to previous node in optimal path
        came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        
        # g_costs maps node to actual cost from start
        g_costs = {start: 0}

        visited = set() # To prevent reprocessing nodes

        while priority_queue:
            f_cost, _, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue
            
            visited.add(current_node)

            if current_node == target:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = came_from[current_node]
                return path[::-1] # Reverse to get path from start to target

            r, c = current_node
            # Directions for neighbors (up, down, left, right)
            neighbors = [
                (r - 1, c), # UP
                (r + 1, c), # DOWN
                (r, c - 1), # LEFT
                (r, c + 1)  # RIGHT
            ]

            for nr, nc in neighbors:
                if SearchHelper._is_valid(nr, nc, rows, cols, impassable_cells):
                    neighbor_node = (nr, nc)
                    
                    # Cost to reach this neighbor is current g_cost + 1 (assuming uniform step cost)
                    new_g_cost = g_costs[current_node] + 1

                    # If this is a shorter path to neighbor_node, update it
                    if neighbor_node not in g_costs or new_g_cost < g_costs[neighbor_node]:
                        g_costs[neighbor_node] = new_g_cost
                        f_cost = new_g_cost + SearchHelper.manhattan_distance(neighbor_node, target)
                        heapq.heappush(priority_queue, (f_cost, new_g_cost, neighbor_node))
                        came_from[neighbor_node] = current_node
        
        return None # No path found

    @staticmethod
    def BFS_shortest_search_to_target(head_position: tuple[int, int], target: tuple[int, int], impassable_squares: set[tuple[int, int]] | list[tuple[int, int]], num_rows: int, num_cols: int) -> list[tuple[int, int]] | None:
        # --- 1. Convert all coordinates to (x, y) tuples for efficient set/deque operations ---
        start_node = head_position
        target_node = target
        
        # Convert impassable squares to a set of tuples for O(1) lookup
        blocked_cells = set(CoordinateConversionHelper.TransformDictToTuple(sq) for sq in impassable_squares) if isinstance(impassable_squares, list) else impassable_squares

        # --- 2. Initialize BFS Queue and Visited Set ---
        # Queue stores tuples: (current_node, path_to_current_node)
        # path_to_current_node is a list of (x, y) tuples
        q = collections.deque([(start_node, [start_node])])
        
        # visited set stores nodes that have been added to the queue to prevent cycles and redundant processing
        visited = {start_node}

        # --- 4. BFS Loop ---
        while q:
            curr_node, path = q.popleft() # Get the current node and the path to it

            # Check if we reached the target
            if curr_node == target_node:
                return path # Found the shortest path, return it

            # Explore neighbors
            for dx, dy in directions:
                next_x, next_y = curr_node[0] + dx, curr_node[1] + dy
                next_node = (next_x, next_y)

                # --- 5. Check Validity of Neighboring Node ---
                # Check boundaries (using pixel coordinates for num_rows/height)
                if not (0 <= next_x < num_rows and 0 <= next_y < num_cols):
                    continue # Out of bounds

                # Check if the square is an impassable obstacle or already visited
                if next_node in blocked_cells or next_node in visited:
                    continue

                # --- 6. Add Valid Neighbor to Queue ---
                visited.add(next_node) # Mark as visited
                q.append((next_node, path + [next_node])) # Add to queue with updated path

        # --- 7. No Path Found ---
        return None # If the queue becomes empty and target was not reached
    
    @staticmethod
    def find_shortest_path_to_tail(head_position: tuple[int, int], tail: tuple[int, int], impassable_squares: set[tuple[int, int]] | list[tuple[int, int]], num_rows:int , num_cols:int) -> list[tuple[int, int]] | None:
        if head_position == tail:
            return [head_position]
        
        return SearchHelper.BFS_shortest_search_to_target(head_position, tail, impassable_squares, num_rows, num_cols)
    
    @staticmethod
    def BFS_count_reachable_cells(start_node: tuple[int, int], impassable_squares: set[tuple[int, int]] | list[tuple[int, int]], num_rows: int, num_cols: int) -> int:
        # Convert impassable_squares to a set for O(1) lookup
        blocked_cells_set = set(impassable_squares) if isinstance(impassable_squares, list) else impassable_squares

        q = collections.deque()
        visited = set()
        
        # Start the BFS from the given node
        if start_node in blocked_cells_set:
            return 0
        
        q.append(start_node)
        visited.add(start_node)
        
        # Define possible moves (dx, dy) in pixel changes
        reachable_count = 0

        while q:
            curr_x, curr_y = q.popleft()
            reachable_count += 1 # Count this cell as reachable

            # Explore neighbors
            for dx, dy in directions:
                next_x, next_y = curr_x + dx, curr_y + dy
                next_node = (next_x, next_y)

                # Check boundaries
                if not (0 <= next_x < num_rows and 0 <= next_y < num_cols):
                    continue # Out of bounds

                # Check if impassable or already visited
                if next_node in blocked_cells_set or next_node in visited:
                    continue

                # Add to queue and mark as visited
                visited.add(next_node)
                q.append(next_node)

        return reachable_count

class BoardState(NamedTuple):
    # Board dimensions
    rows: int
    cols: int
    
    # Locations
    food_locations: Set[Tuple[int, int]] # Use Set for faster lookups
    obstacle_locations: Set[Tuple[int, int]] # Use Set

    # Snake states (already converted to tuples in get_next_move)
    player_snake: PlayerStateSimulationType
    opponent_snake: PlayerStateSimulationType

    # You might want to pre-calculate certain common obstacle sets here too,
    # but let's keep it simple for now.

    # This method is for initial setup from the raw API input
    @classmethod
    def from_api_data(cls, board_state_api: BoardStateType, player_state_api: PlayerStateSimulationType, opponent_state_api: PlayerStateSimulationType):
        return cls(
            rows=board_state_api['rows'],
            cols=board_state_api['cols'],
            food_locations=set(board_state_api['food_locations']),
            obstacle_locations=set(board_state_api['obstacle_locations']),
            player_snake=player_state_api,
            opponent_snake=opponent_state_api
        )

class SnakeSimulator:
    @staticmethod
    def simulate_move(
        snake_state: PlayerStateSimulationType, 
        new_position: Tuple[int, int], 
        board_state: BoardState # Pass the full game_state to access food
    ) -> PlayerStateSimulationType:
        
        sim_snake: PlayerStateSimulationType = deepcopy(snake_state)
        
        sim_snake['direction'] = DirectionHelper.DetermineDirection(
            sim_snake['head_position'], new_position
        )
        sim_snake['head_position'] = new_position
        sim_snake["body"].insert(0, new_position)

        # Check if the new position is food
        if new_position in board_state.food_locations: # Use game_state.food_locations
            sim_snake['length'] += 1
            sim_snake['score'] += 1
        else:
            sim_snake['body'].pop() # Remove tail if no food

        return sim_snake
    
class Pathfinder:
    def __init__(self, game_state: BoardState):
        self.game_state = game_state
    
    def shortest_path_to_target(self, snake: PlayerStateSimulationType, 
                                target: Tuple[int, int], 
                                impassable: Set[Tuple[int, int]],
                               ) -> Optional[List[Tuple[int, int]]]:
        return SearchHelper.AStar_shortest_search_to_target(snake['head_position'], target, impassable, self.game_state.rows, self.game_state.cols)
    
    def _get_impassable_for_snake(self, 
                                  snake: PlayerStateSimulationType, 
                                  exclude_head: bool = False, 
                                  exclude_tail: bool = False,
                                  include_opponent_head: bool = False # For head-to-head scenarios
                                 ) -> Set[Tuple[int, int]]:
        """
        Helper to create the set of impassable cells for a given snake's pathing.
        """
        impassable = set(self.game_state.obstacle_locations)
        
        # Add current snake's body as obstacles
        current_snake_body_set = set(snake['body'])
        if exclude_head:
            current_snake_body_set.discard(snake['head_position'])
        if exclude_tail:
            current_snake_body_set.discard(snake['body'][-1])
        impassable.update(current_snake_body_set)

        # Add opponent's body as obstacles
        opponent_snake = self.game_state.opponent_snake if snake['id'] == self.game_state.player_snake['id'] else self.game_state.player_snake
        opponent_snake_body_set = set(opponent_snake['body'])
        if not include_opponent_head: # Typically exclude opponent's head if it's a target
            opponent_snake_body_set.discard(opponent_snake['head_position'])
        impassable.update(opponent_snake_body_set)

        return impassable


    def shortest_paths_to_food(self, snake: PlayerStateSimulationType) -> List[Optional[List[Tuple[int, int]]]]:
        """
        Finds shortest paths from snake's head to all food locations.
        """
        # For general pathfinding, the snake's body (excluding tail) is an obstacle.
        # The tail is considered movable.
        impassable = self._get_impassable_for_snake(
            snake, exclude_head=True, exclude_tail=False, include_opponent_head=True # Opponent's head is not a pathing target here
        )

        food_locations_list = list(self.game_state.food_locations) # Convert to list for consistent indexing
        if not food_locations_list:
            return [] # No food, no paths

        food_list = []
        for food_pos in food_locations_list:
            result = SearchHelper.AStar_shortest_search_to_target(snake['head_position'], food_pos, impassable, self.game_state.rows, self.game_state.cols)
            # return p.starmap(SearchHelper.BFS_shortest_search_to_target, starmap_args)
            food_list.append(result)
        
        return food_list
        

    def shortest_path_to_enemy_head(self, attacking_snake: PlayerStateSimulationType, target_snake: PlayerStateSimulationType) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from attacking_snake's head to target_snake's head.
        Considers both snake bodies as obstacles.
        """
        # When pathing to an enemy head, your body (excluding tail) is an obstacle.
        # The enemy's body (excluding their head, which is the target) is an obstacle.
        impassable = self._get_impassable_for_snake(
            attacking_snake, exclude_head=True, exclude_tail=True, include_opponent_head=False # Opponent's head is the target
        )
        
        # Add target snake's body (excluding its head) as obstacles
        impassable.update(set(target_snake['body']) - {target_snake['head_position']})

        # return SearchHelper.BFS_shortest_search_to_target(
        #     attacking_snake['head_position'],
        #     target_snake['head_position'],
        #     impassable,
        #     self.game_state.rows,
        #     self.game_state.cols
        # )
        return SearchHelper.AStar_shortest_search_to_target(
            attacking_snake['head_position'],
            target_snake['head_position'],
            impassable,
            self.game_state.rows,
            self.game_state.cols
        )

    def count_reachable_cells(self, snake: PlayerStateSimulationType) -> List[int]:
        """
        Counts reachable cells for each possible next move of the snake.
        """
        head_position = snake["head_position"]
        possible_next_positions = [(head_position[0] + dr, head_position[1] + dc) for dr, dc in directions]
        
        pos_impassable_tuple: List[Tuple[Tuple[int, int], Set[Tuple[int, int]]]] = []
        for pos in possible_next_positions:
            sim_snake = SnakeSimulator.simulate_move(snake, pos, self.game_state) # Simulate first move to ensure snake is valid
            # When counting reachable cells, your own body (excluding tail) and opponent's body are obstacles.
            if sim_snake['head_position'][0] < 0 or sim_snake['head_position'][0] >= self.game_state.rows or \
               sim_snake['head_position'][1] < 0 or sim_snake['head_position'][1] >= self.game_state.cols:
                exclude_head = False
            else:
                exclude_head = True
            impassable = self._get_impassable_for_snake(
                sim_snake, exclude_head=exclude_head, exclude_tail=True # Current head is starting point, tail might be vacated
            )
            # For reachable cells, the full opponent body is an obstacle (don't exclude their head)
            impassable.update(set(self.game_state.opponent_snake['body']))
        
            pos_impassable_tuple.append((pos, impassable))

        results = []
        for pos, impassable in pos_impassable_tuple:
            result = SearchHelper.BFS_count_reachable_cells(pos, impassable, self.game_state.rows, self.game_state.cols)
            results.append(result)
        return results # Returns a list of counts, corresponding to possible_next_positions
        
    def find_shortest_path_to_tail(self, snake: PlayerStateSimulationType) -> Optional[List[Tuple[int, int]]]:
        """
        Finds shortest paths from snake's head to all food locations.
        """
        # For general pathfinding, the snake's body (excluding tail) is an obstacle.
        # The tail is considered movable.
        impassable = self._get_impassable_for_snake(
            snake, exclude_head=True, exclude_tail=True, include_opponent_head=True # Opponent's head is not a pathing target here
        )
        snake_tail = snake['body'][-1]  # The tail is the last element in the body list

        # return SearchHelper.BFS_shortest_search_to_target(snake['head_position'], snake_tail, impassable, self.game_state.rows, self.game_state.cols)
        return SearchHelper.AStar_shortest_search_to_target(snake['head_position'], snake_tail, impassable, self.game_state.rows, self.game_state.cols)

class SafetyChecker:
    def __init__(self, game_state: BoardState, pathfinder: Pathfinder):
        self.game_state = game_state
        self.pathfinder = pathfinder # Use the shared pathfinder instance

    def is_tail_safe(self, snake: PlayerStateSimulationType) -> bool:
        """
        Checks if the snake can reach its own tail from its current head position.
        """
        path_to_tail = self.pathfinder.find_shortest_path_to_tail(snake)
        return path_to_tail is not None

    def is_tail_safe_after_move(self, snake_state: PlayerStateSimulationType, new_position: Tuple[int, int]) -> bool:
        """
        Checks if the snake's tail is safe after a hypothetical move.
        """
        simulated_snake = SnakeSimulator.simulate_move(snake_state, new_position, self.game_state)
        return self.is_tail_safe(simulated_snake)

    def is_move_safe(
        self, 
        player_next_pos: Tuple[int, int], 
        opponent_predicted_move_to_food: Optional[Tuple[int, int]], 
        opponent_predicted_move_to_player: Optional[Tuple[int, int]],
        board_state_override: Optional[BoardState] = None
    ) -> bool:
        """
        Combines tail safety after move and collision with opponent.
        """
        # First, check if the proposed move leads to a collision with walls or self-collision
        # (This check should ideally happen before calling is_move_safe, e.g., in get_next_move or an initial filter)
        # For now, let's assume player_next_pos is within bounds and not immediate self-collision.
        
        current_board = board_state_override if board_state_override else self.game_state
        player_snake = current_board.player_snake
        opponent_snake = current_board.opponent_snake

        # 1. Check for wall and obstacle collision
        if not (0 <= player_next_pos[0] < current_board.rows and
                0 <= player_next_pos[1] < current_board.cols):
            return False # Out of bounds (wall collision)
        if player_next_pos in current_board.obstacle_locations:
            return False # Collision with static obstacle

        # 2. Check for self-collision
        # A snake can move into the position of its tail *if* it's not eating food.
        # For simplicity and general safety, consider any non-tail body segment as collision.
        player_body_set = set(player_snake['body'])
        if len(player_snake['body']) > 1: # If the snake is longer than just a head
            player_body_set.discard(player_snake['body'][-1]) # Tail will move, so it's not an obstacle for self

        if player_next_pos in player_body_set:
            return False # Collision with own body (excluding current tail)

        # 3. Check for opponent collision (body and head)
        if player_next_pos in set(opponent_snake['body']):
            return False # Collision with opponent's body

        # 4. Head-to-head collision prediction (Crucial for aggression)
        if player_next_pos == opponent_snake['head_position']:
            # If we are strictly longer, we *win* the head-to-head.
            # If we are shorter or equal length, it's a loss or mutual destruction.
            if player_snake['length'] > opponent_snake['length']:
                return True # This is safe because we win!
            else:
                return False # Not safe (loss or draw)

        # Predict future head-to-head collision (if opponent predictions are provided)
        # This assumes opponent moves optimally or predictably.
        if opponent_predicted_move_to_food and player_next_pos == opponent_predicted_move_to_food:
            # If opponent is going for food and we intercept their path.
            # If we are strictly longer, we win.
            if player_snake['length'] > opponent_snake['length']: 
                return True 
            else:
                return False 
        
        if opponent_predicted_move_to_player and player_next_pos == opponent_predicted_move_to_player:
            # If opponent is explicitly targeting us.
            # If we are strictly longer, we win.
            if player_snake['length'] > opponent_snake['length']: 
                return True 
            else:
                return False 

        # 5. Check if the move leads to self-trapping (cannot reach tail after move)
        # This is a crucial safety check for all moves.
        if not self.is_tail_safe_after_move(player_snake, player_next_pos):
            return False

        return True # If none of the above, the move is considered safe

class DecisionEngine:
    def __init__(self, board_state: BoardState, pathfinder: Pathfinder, safety_checker: SafetyChecker):
        self.board_state = board_state
        self.pathfinder = pathfinder
        self.safety_checker = safety_checker
        
        # Thresholds for containment strategy - Tunable parameters
        self.CONTAINMENT_MIN_LENGTH_ADVANTAGE = 5 # Player must be this many segments longer than opponent
        self.CONTAINMENT_MIN_BOARD_COVERAGE_PERCENT = 0.10 # Player length must be at least this % of total cells
        self.CONTAINMENT_OPPONENT_MAX_INITIAL_SPACE_FACTOR = 0.5 # Opponent must have at least this % of board initially
        self.CONTAINMENT_OPPONENT_TARGET_SPACE = 10 # Try to reduce opponent space below this many cells
        
        # Scoring weights for containment moves - Simpler and more direct
        self.WEIGHT_OPPONENT_SPACE_REDUCTION = 5.0 # Reward for reducing opponent's accessible cells
        self.WEIGHT_PLAYER_SPACE_MAINTENANCE = 1.0 # Reward for keeping our own accessible cells high
        self.WEIGHT_BLOCK_OPPONENT_FOOD = 10.0     # Bonus for blocking opponent's closest food
        self.WEIGHT_TRAP_OPPONENT = 100.0          # High bonus if opponent has 0 or 1 safe moves
        self.WEIGHT_CLOSER_TO_OPPONENT_TAIL = 5.0  # Bonus for getting closer to opponent's tail (aggressive containment)
        self.WEIGHT_REDUCE_OPPONENT_SAFE_MOVES = 17.5 # Bonus for reducing opponent's available safe moves

        # Scoring weights for food path selection
        self.FOOD_SCORE_PATH_LENGTH_BASE = 100
        self.FOOD_PENALTY_DIRECT_COMPETITION = 65
        self.FOOD_PENALTY_SIGNIFICANTLY_FASTER_OPPONENT = 200 # If opponent is much faster to this food
        self.FOOD_BONUS_CLEARED_BY_OPPONENT = 30 # If opponent takes our contested food, and we find new safe path
        self.FOOD_BONUS_OPPONENT_CLEARS_PATH = 30 # If opponent's move makes our path shorter/safer
        self.FOOD_BONUS_NOT_CONTESTED = 5 # Small bonus for uncontested food
        self.FOOD_BONUS_LONGER_OPPONENT_PATH = 0.35 # Factor for making opponent's path longer
        self.FOOD_BONUS_UNREACHABLE_OPPONENT_FOOD = 2.5 # Factor for making opponent's food unreachable
        self.MIN_SPACE_EXPONENT_SNAKE_LENGTH = 1.1

        # Aggression parameters
        self.AGGRESSION_MIN_LENGTH_ADVANTAGE_FOR_FOOD_STEAL = 5 # Player must be this many segments longer to aggressively steal food
        self.AGGRESSION_FOOD_STEAL_WIN_BONUS = 45 # High bonus for successfully stealing food

    
    def _create_temp_board_state_after_my_move(self, my_next_pos: Tuple[int, int]) -> Optional[BoardState]:
        """
        Simulates the player's snake making a move and returns a new BoardState.
        Returns None if the player's move itself is invalid (wall, immediate own body).
        """
        # Simulate player's move. Food consumption is handled by SnakeSimulator.
        sim_player_snake = SnakeSimulator.simulate_move(
            self.board_state.player_snake,
            my_next_pos,
            self.board_state # Original board state for food location check
        )

        # Basic validity check for the player's simulated move itself.
        # The head of the simulated snake is my_next_pos.
        # We must check against the *original* player body to see if it's a valid step.
        # A snake can move into the space previously occupied by its tail.
        original_player_body_set = set(self.board_state.player_snake['body'])
        if len(self.board_state.player_snake['body']) > 1: # if snake is longer than 1 segment
            original_player_body_set.discard(self.board_state.player_snake['body'][-1]) # Allow moving to old tail

        if not (0 <= my_next_pos[0] < self.board_state.rows and
                0 <= my_next_pos[1] < self.board_state.cols and
                my_next_pos not in original_player_body_set and # Check against original body (sans tail)
                my_next_pos not in self.board_state.obstacle_locations):
            # print(f"[Containment] Invalid base move for player to {my_next_pos}")
            return None

        # Create a new BoardState with the simulated player snake
        temp_board = BoardState(
            rows=self.board_state.rows,
            cols=self.board_state.cols,
            food_locations=self.board_state.food_locations, # Food locations don't change by player's move alone
            obstacle_locations=self.board_state.obstacle_locations,
            player_snake=sim_player_snake, # Use the player's simulated state
            opponent_snake=self.board_state.opponent_snake # Opponent remains in their current state
        )
        return temp_board
        

    def _calculate_dynamic_min_free_space(self, player_snake_length: int) -> int:
        """
        Calculates a dynamic minimum required free space.
        """
        
        # Additional minimum based on snake length (longer snakes need more space)
        min_space_from_length = int(player_snake_length ** self.MIN_SPACE_EXPONENT_SNAKE_LENGTH)
        # Cap the requirement to 33% of the total board size to be safe
        max_allowed_requirement = (self.board_state.rows * self.board_state.cols) // 3
        calculated_space = min(min_space_from_length, max_allowed_requirement)

        # Add a hardcoded minimum to prevent it from going too low on very small boards/short snakes
        return max(calculated_space, 5) # Ensure at least 5 cells of space
    
    def _evaluate_food_path_score(
        self,
        player_path: List[Tuple[int, int]],
        opponent_next_move_to_food_coord: Optional[Tuple[int, int]],
        opponent_closest_food_target: Optional[Tuple[int, int]],
        opponent_sorted_shortest_paths_to_food: List[List[Tuple[int, int]]]
    ) -> float:
        """
        Helper function to score a single player food path.
        """
        current_player_next_pos = player_path[1]
        current_player_food_target = player_path[-1]
        path_score = 0.0

        # 1. Path Length Score (shorter is better)
        path_score += (self.FOOD_SCORE_PATH_LENGTH_BASE - len(player_path))

        # New: If the path is longer than a certain threshold, apply a penalty
        # Simulate player's move to get the state for space calculation
        sim_player_after_move = SnakeSimulator.simulate_move(self.board_state.player_snake, current_player_next_pos, self.board_state)

        # # Calculate player's reachable space *after* this move
        # # Create a temporary pathfinder for this simulated state to get correct impassable
        # temp_pathfinder_for_player_eval = Pathfinder(BoardState(
        #     rows=self.board_state.rows,
        #     cols=self.board_state.cols,
        #     food_locations=self.board_state.food_locations,
        #     obstacle_locations=self.board_state.obstacle_locations,
        #     player_snake=sim_player_after_move,
        #     opponent_snake=self.board_state.opponent_snake # Opponent doesn't move here
        # ))

        # Calculate the dynamic minimum required free space for this turn
        dynamic_min_free_space = self._calculate_dynamic_min_free_space(
            self.board_state.player_snake['length']
        )
        # The impassable set needs to be carefully constructed for flood fill from the new head
        # Your own new body (excluding the head which is the start of BFS) and opponent's body are obstacles.
        player_space_impassable = set(sim_player_after_move['body'])
        player_space_impassable.discard(sim_player_after_move['head_position']) # Don't block our own head
        player_space_impassable.update(self.board_state.opponent_snake['body']) # Opponent is obstacle
        player_space_impassable.update(self.board_state.obstacle_locations) # Static obstacles

        player_space_after_move = SearchHelper.BFS_count_reachable_cells(
            sim_player_after_move['head_position'],
            player_space_impassable,
            self.board_state.rows,
            self.board_state.cols
        )

        # Apply penalty/reward based on the dynamic minimum space
        if player_space_after_move < dynamic_min_free_space:
            # Heavy penalty, scales with how far below the dynamic threshold it is
            path_score -= (dynamic_min_free_space - player_space_after_move) * 50.0 
        else:
            # Reward for having good space (relative to dynamic needs)
            path_score += player_space_after_move * 3.0 

        # 2. Opponent Competition & Prediction (Aggressive Food Stealing)
        if opponent_closest_food_target == current_player_food_target:
            opponent_path_len_to_this_food = len(opponent_sorted_shortest_paths_to_food[0]) if opponent_sorted_shortest_paths_to_food else float('inf')
            
            # If we are strictly faster, this is an aggressive steal opportunity!
            if len(player_path) < opponent_path_len_to_this_food:
                # Add a high bonus for successfully stealing food, especially if we have a length advantage
                if self.board_state.player_snake['length'] >= self.board_state.opponent_snake['length'] + self.AGGRESSION_MIN_LENGTH_ADVANTAGE_FOR_FOOD_STEAL:
                    path_score += self.AGGRESSION_FOOD_STEAL_WIN_BONUS * 2 # Even higher bonus if we are longer
                else:
                    path_score += self.AGGRESSION_FOOD_STEAL_WIN_BONUS # Standard aggressive steal bonus
            
            # If opponent is faster or equal, penalize
            elif len(player_path) >= opponent_path_len_to_this_food: 
                if opponent_path_len_to_this_food + 2 <= len(player_path): # Opponent is at least 2 steps faster
                     path_score -= self.FOOD_PENALTY_SIGNIFICANTLY_FASTER_OPPONENT # Very heavy penalty to abandon this food
                else:
                    path_score -= self.FOOD_PENALTY_DIRECT_COMPETITION # Significant penalty for direct competition we likely lose

        # 3. Scenario: What if opponent takes THEIR closest food? Does it benefit us for OTHER food?
        if opponent_closest_food_target and opponent_next_move_to_food_coord:
            # Create a temporary board state where opponent has eaten their closest food
            sim_opponent_after_food = SnakeSimulator.simulate_move(
                self.board_state.opponent_snake, 
                opponent_next_move_to_food_coord, 
                self.board_state
            )
            
            # Remove the food opponent just ate from the simulated board for pathfinding
            sim_food_locations = set(self.board_state.food_locations)
            if opponent_closest_food_target in sim_food_locations:
                sim_food_locations.remove(opponent_closest_food_target)

            # Create a temporary board_state *after opponent's predicted move*
            temp_board_after_opponent_food_eat = BoardState(
                rows=self.board_state.rows,
                cols=self.board_state.cols,
                food_locations=sim_food_locations,
                obstacle_locations=self.board_state.obstacle_locations,
                player_snake=self.board_state.player_snake, 
                opponent_snake=sim_opponent_after_food 
            )
            temp_pathfinder_after_opponent_move = Pathfinder(temp_board_after_opponent_food_eat)

            # Find the *new* closest food for us after opponent's predicted move
            player_paths_after_opponent_move = temp_pathfinder_after_opponent_move.shortest_paths_to_food(
                temp_board_after_opponent_food_eat.player_snake
            )
            player_sorted_paths_after_opponent = sorted([p for p in player_paths_after_opponent_move if p is not None and len(p) > 0], key=len)

            if player_sorted_paths_after_opponent:
                player_new_closest_food_target = player_sorted_paths_after_opponent[0][-1]
                player_new_closest_path_len = len(player_sorted_paths_after_opponent[0])

                # Case A: Opponent took our current target food (it was contested)
                if current_player_food_target == opponent_closest_food_target:
                    # Reward successfully finding a new, viable path
                    if player_new_closest_food_target != current_player_food_target:
                         path_score += self.FOOD_BONUS_CLEARED_BY_OPPONENT 

                # Case B: Opponent took *their* food, which was *not* our current target.
                # Does it make our path to our current target, or to a new better target, better?
                else:
                    path_to_current_target_after_opponent = temp_pathfinder_after_opponent_move.shortest_path_to_target(
                        temp_board_after_opponent_food_eat.player_snake,
                        current_player_food_target,
                        temp_pathfinder_after_opponent_move._get_impassable_for_snake(temp_board_after_opponent_food_eat.player_snake, exclude_head=False, exclude_tail=True, include_opponent_head=True)
                    )
                    
                    if path_to_current_target_after_opponent and \
                       len(path_to_current_target_after_opponent) < len(player_path): # Our path became shorter
                        path_score += self.FOOD_BONUS_OPPONENT_CLEARS_PATH 
            
            # If our chosen food path *remains viable* and the opponent is occupied with other food, give a small bonus
            if current_player_food_target != opponent_closest_food_target:
                path_score += self.FOOD_BONUS_NOT_CONTESTED

            
        return path_score

    def determine_best_food_path(
        self,
        player_sorted_shortest_paths: List[List[Tuple[int, int]]],
        opponent_sorted_shortest_paths_to_food: List[List[Tuple[int, int]]],
        opponent_shortest_path_to_head: Optional[List[Tuple[int, int]]]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Determines the best safe path to food, considering opponent's position and potential food stealing.
        Prioritizes food that is safe and where the player has an advantage,
        also considering how opponent's move might free up other food.
        """
        if not player_sorted_shortest_paths:
            return None

        # Predict opponent's most likely immediate moves from their paths
        opponent_next_move_to_food_coord = None
        opponent_closest_food_target = None
        if opponent_sorted_shortest_paths_to_food and len(opponent_sorted_shortest_paths_to_food[0]) > 1:
            opponent_next_move_to_food_coord = opponent_sorted_shortest_paths_to_food[0][1]
            opponent_closest_food_target = opponent_sorted_shortest_paths_to_food[0][-1] # The actual food coordinate

        opponent_next_move_to_player_coord = None
        if opponent_shortest_path_to_head and len(opponent_shortest_path_to_head) > 1:
            opponent_next_move_to_player_coord = opponent_shortest_path_to_head[1]

        best_path_score = -float('inf')
        best_overall_path = None

        for player_path in player_sorted_shortest_paths:
            if not player_path or len(player_path) < 2:
                continue # Skip empty or too short paths

            current_player_next_pos = player_path[1]
            
            # 1. Basic Safety Check
            if not self.safety_checker.is_move_safe(
                current_player_next_pos,
                opponent_next_move_to_food_coord,
                opponent_next_move_to_player_coord
            ):
                continue # Immediately unsafe, discard this path
            
            # 2. Ensure tail safety after this move
            sim_player_after_move = SnakeSimulator.simulate_move(self.board_state.player_snake, current_player_next_pos, self.board_state)
            if not self.safety_checker.is_tail_safe(sim_player_after_move):
                continue # This move makes us self-trap, discard

            # Score this path using the dedicated helper function
            current_path_score = self._evaluate_food_path_score(
                player_path,
                opponent_next_move_to_food_coord,
                opponent_closest_food_target,
                opponent_sorted_shortest_paths_to_food
            )
            
            if current_path_score > best_path_score:
                best_path_score = current_path_score
                best_overall_path = player_path
        
        return best_overall_path
    
    def find_best_escape_direction(self, snake: PlayerStateSimulationType) -> MoveEnum:
        """
        Finds the direction that maximizes reachable cells, prioritizing safe moves.
        Uses the global 'directions' and 'direction_move_map' from move_enum.py.
        """
        head_position = snake["head_position"]
        # print("Trying to escape")
        # Prepare arguments for reachable cell calculations
        # We now iterate directly through the 'directions' list
        potential_next_positions = []
        for dr, dc in directions: # Use the list of (dr, dc) tuples
            next_pos = (head_position[0] + dr, head_position[1] + dc)
            potential_next_positions.append(next_pos)

        # Get reachable cell counts for all 4 possible moves
        # Pass the player's current snake to pathfinder; it determines obstacles
        reachable_counts = self.pathfinder.count_reachable_cells(snake)
        
        # Pair up moves with their reachable counts.
        # Ensure that `reachable_counts` order matches `potential_next_positions` order
        # as returned by `pathfinder.count_reachable_cells`.
        moves_with_counts = []
        for i, (dr, dc) in enumerate(directions): # Iterate using original direction order
            next_pos = potential_next_positions[i] # Get the position corresponding to this direction
            count = reachable_counts[i] # Get the count for this position
            move_enum = direction_move_map[(dr, dc)] # Use the map to get the MoveEnum
            moves_with_counts.append((move_enum, next_pos, count))

        # Sort moves by reachable cell count in descending order
        sorted_moves: List[Tuple[MoveEnum,Tuple[int, int], int]] = sorted(moves_with_counts, key=lambda x: x[2], reverse=True)

        best_direction_name = MoveEnum.UP # Default fallback
        max_reachable_cells = -1
        
        # Iterate through sorted moves to find the safest and most open path
        for move_enum, next_pos, count in sorted_moves:
            # First, a basic validity check (walls, self-body, static obstacles).
            # This is partly redundant if pathfinder.count_reachable_cells already excludes these,
            # but it's a good sanity check if the BFS returned 0 for blocked paths.
            if not (0 <= next_pos[0] < self.board_state.rows and
                    0 <= next_pos[1] < self.board_state.cols and
                    next_pos not in (set(snake['body'])-{snake['body'][-1]}) and # Cannot move into own body (except tail, which is handled in BFS)
                    next_pos not in self.board_state.obstacle_locations):
                continue # This move is immediately invalid, skip it

            # Now, check for more complex safety (tail safety after move, opponent collisions)
            if self.safety_checker.is_move_safe(next_pos, None, None): # No opponent prediction for flood fill escape
                # If safe and offers more space, update best
                if count > max_reachable_cells:
                    max_reachable_cells = count
                    best_direction_name = move_enum

        # If no safe move was found through the above strategy, try a very basic fallback
        if max_reachable_cells == -1: # No safe, reachable move found
            # print("[DecisionEngine] No safe escape path with reachable cells found. Attempting move with most spaces")
            for move_enum, next_pos, count in sorted_moves:
                if not (0 <= next_pos[0] < self.board_state.rows and
                    0 <= next_pos[1] < self.board_state.cols and
                    next_pos not in (set(snake['body'])-{snake['body'][-1]}) and # Cannot move into own body (except tail, which is handled in BFS)
                    next_pos not in self.board_state.obstacle_locations):
                    best_direction_name = move_enum
                    continue

                return move_enum

        return best_direction_name


    def decide_next_move(self) -> MoveEnum:
        """
        Main decision logic for the snake.
        Prioritizes: Containment -> Food Collection -> Escape/Maximize Space
        """
        player = self.board_state.player_snake
        opponent = self.board_state.opponent_snake

        # Priority 2: Go for Food
        player_shortest_paths_to_each_food = self.pathfinder.shortest_paths_to_food(player)
        player_sorted_shortest_paths_to_food = sorted(
            [path for path in player_shortest_paths_to_each_food if path is not None and len(path) > 0],
            key=len
        )

        opponent_shortest_paths_to_each_food = self.pathfinder.shortest_paths_to_food(opponent)
        opponent_sorted_shortest_paths_to_food = sorted(
            [path for path in opponent_shortest_paths_to_each_food if path is not None and len(path) > 0],
            key=len
        )

        opponent_shortest_path_to_player_head = self.pathfinder.shortest_path_to_enemy_head(opponent, player)

        # Both players now use the single, advanced determine_best_food_path
        best_food_path = self.determine_best_food_path(
            player_sorted_shortest_paths_to_food,
            opponent_sorted_shortest_paths_to_food,
            opponent_shortest_path_to_player_head
        )

        if best_food_path and len(best_food_path) > 1:
            next_food_coord = best_food_path[1]
            food_move = DirectionHelper.DetermineDirection(player['head_position'], next_food_coord)
            return food_move

        # Priority 3: Escape (Maximize Reachable Space)
        escape_move = self.find_best_escape_direction(player)
        return escape_move

def get_next_move(board_state: BoardStateType, player_state: PlayerStateType, opponent_state: PlayerStateType) -> MoveEnum:
    player_state_internal: PlayerStateSimulationType = {
        'id': player_state['id'],
        'head_position': CoordinateConversionHelper.TransformDictToTuple(player_state['head_position']),
        'body': [CoordinateConversionHelper.TransformDictToTuple(seg) for seg in player_state['body']],
        'length': player_state['length'],
        'score': player_state['score'],
        'direction': player_state['direction'] # This is raw enum, not a coord tuple
    }

    opponent_state_internal: PlayerStateSimulationType = {
        'id': opponent_state['id'],
        'head_position': CoordinateConversionHelper.TransformDictToTuple(opponent_state['head_position']),
        'body': [CoordinateConversionHelper.TransformDictToTuple(seg) for seg in opponent_state['body']],
        'length': opponent_state['length'],
        'score': opponent_state['score'],
        'direction': opponent_state['direction']
    }

    # 2. Convert raw API input to our internal GameState (tuple-based)
    current_game_state = BoardState.from_api_data(board_state, player_state_internal, opponent_state_internal)

    # 3. Instantiate core logic components
    # These instances share the `current_game_state` and the `current_pool`
    pathfinder = Pathfinder(current_game_state)
    safety_checker = SafetyChecker(current_game_state, pathfinder) # SafetyChecker uses Pathfinder
    decision_engine = DecisionEngine(current_game_state, pathfinder, safety_checker) # DecisionEngine uses both
    # 4. Let the DecisionEngine determine the best move
    chosen_move = decision_engine.decide_next_move()

    return chosen_move

def set_player_name():
    return "Bush Wookie"