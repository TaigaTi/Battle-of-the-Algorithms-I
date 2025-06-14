from config import config
import time
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional

GRID_SIZE = config.GRID_SIZE

class SnakeState(Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    EXPLORING = "exploring"
    ESCAPING = "escaping"
    ENDGAME = "endgame"

def convert_to_internal_format(board_state, player_state, opponent_state):
    """Convert the game format to internal AI format"""
    # Convert player state - use 'row' and 'col' instead of 'x' and 'y'
    my_snake = {
        'head': (player_state["head_position"]["row"], player_state["head_position"]["col"]),
        'body': [(segment["row"], segment["col"]) for segment in player_state["body"]]
    }
    
    # Convert opponent state - use 'row' and 'col' instead of 'x' and 'y'
    opponent_snake = {
        'head': (opponent_state["head_position"]["row"], opponent_state["head_position"]["col"]),
        'body': [(segment["row"], segment["col"]) for segment in opponent_state["body"]]
    }
    
    # Convert board state - food_locations is already in (row, col) format
    internal_board = {
        'height': board_state["rows"],  # Use rows/cols directly, not divided by GRID_SIZE
        'width': board_state["cols"],
        'food': board_state.get("food_locations", []),
        'obstacles': board_state.get("obstacle_locations", [])
    }
    
    return internal_board, my_snake, opponent_snake

def get_opposite_direction(direction: str) -> str:
    """Get the opposite direction to prevent 180-degree turns"""
    opposites = {
        'up': 'down',
        'down': 'up',
        'left': 'right',
        'right': 'left'
    }
    return opposites.get(direction, '')

def get_current_direction(snake_body: List[Tuple[int, int]], last_move: str = None) -> str:
    """Determine current direction from snake body or last move"""
    if len(snake_body) < 2:
        # When snake length is 1, use the last move direction
        return last_move
    
    head = snake_body[0]
    neck = snake_body[1]
    
    row_diff = head[0] - neck[0]
    col_diff = head[1] - neck[1]
    
    if row_diff == -1:
        return 'up'
    elif row_diff == 1:
        return 'down'
    elif col_diff == -1:
        return 'left'
    elif col_diff == 1:
        return 'right'
    
    return last_move

def simulate_move_static(snake: Dict, move: str) -> Dict:
    """Static version of move simulation"""
    move_deltas = {
        'up': (-1, 0), 'down': (1, 0),
        'left': (0, -1), 'right': (0, 1)
    }
    
    new_snake = snake.copy()
    dx, dy = move_deltas[move]
    old_head = snake['head']
    new_head = (old_head[0] + dx, old_head[1] + dy)
    
    new_snake['head'] = new_head
    new_snake['body'] = [new_head] + snake['body'][:-1]
    
    return new_snake

def evaluate_position_static(board_state: Dict, my_snake: Dict, opponents: List[Dict], weights: Dict) -> float:
    """Enhanced position evaluation with better wall handling and food prioritization"""
    score = 0.0
    my_head = my_snake['head']
    my_length = len(my_snake['body'])
    
    # Survival evaluation
    accessible_space = calculate_accessible_space_static(my_head, board_state, my_snake, opponents)
    score += weights['survival_weight'] * accessible_space
    
    # Food evaluation - ENHANCED for better food seeking
    food_locations = board_state.get('food', [])
    if food_locations:
        closest_food_dist = min(manhattan_distance_static(my_head, food) for food in food_locations)
        # Stronger food attraction, especially when food is close
        food_bonus = weights['food_priority'] * (25 - closest_food_dist)
        if closest_food_dist <= 3:  # Extra bonus for very close food
            food_bonus += 30
        score += food_bonus
        
        # Bonus for moving towards food even near walls
        closest_food = min(food_locations, key=lambda f: manhattan_distance_static(my_head, f))
        food_direction_bonus = calculate_food_direction_bonus(my_head, closest_food)
        score += food_direction_bonus * 10
    
    # Opponent evaluation
    for opponent in opponents:
        opponent_dist = manhattan_distance_static(my_head, opponent['head'])
        opponent_length = len(opponent['body'])
        length_diff = my_length - opponent_length
        score += length_diff * 10
        
        # Avoid getting too close to larger opponents
        if opponent_length >= my_length and opponent_dist < 4:
            score -= (4 - opponent_dist) * 15
    
    # IMPROVED: Much less aggressive wall penalty - only when trapped or very close
    board_height = board_state.get('height', 11)
    board_width = board_state.get('width', 11)
    
    # Only penalize walls when we're actually trapped or have very little space
    if accessible_space < 10:  # Only when space is limited
        # Light penalty for being on edges
        if my_head[0] == 0 or my_head[0] == board_height - 1:
            score -= 5
        if my_head[1] == 0 or my_head[1] == board_width - 1:
            score -= 5
    
    # Corner penalty (more dangerous)
    if ((my_head[0] == 0 or my_head[0] == board_height - 1) and 
        (my_head[1] == 0 or my_head[1] == board_width - 1)):
        score -= 15
    
    # Center bonus only when we have plenty of space and no immediate food
    center_row = board_height // 2
    center_col = board_width // 2
    distance_to_center = manhattan_distance_static(my_head, (center_row, center_col))
    if accessible_space > 25 and (not food_locations or closest_food_dist > 8):
        score += max(0, 8 - distance_to_center) * 2
    
    return score

def calculate_food_direction_bonus(head: Tuple[int, int], food: Tuple[int, int]) -> float:
    """Calculate bonus for moving towards food"""
    food_distance = manhattan_distance_static(head, food)
    if food_distance == 0:
        return 0
    
    # Direction vector towards food
    row_diff = food[0] - head[0]
    col_diff = food[1] - head[1]
    
    # Normalize to get direction preference
    if abs(row_diff) > abs(col_diff):
        return 1.0 if row_diff != 0 else 0.5
    else:
        return 1.0 if col_diff != 0 else 0.5

def calculate_accessible_space_static(position: Tuple[int, int], board_state: Dict, 
                                    my_snake: Dict, opponents: List[Dict]) -> int:
    """Optimized flood fill for accessible space calculation"""
    visited = set()
    queue = [position]
    obstacles = set(my_snake['body'][:-1])  # Exclude tail as it will move
    
    # Add opponent bodies
    for opponent in opponents:
        obstacles.update(opponent['body'])
    
    # Add static obstacles from the map
    obstacles.update(board_state.get('obstacles', []))
    
    board_height = board_state.get('height', 11)
    board_width = board_state.get('width', 11)
    max_iterations = min(60, board_height * board_width // 3)  # Allow more exploration
    iterations = 0
    
    while queue and iterations < max_iterations:
        current = queue.pop(0)
        iterations += 1
        
        if current in visited or current in obstacles:
            continue
        
        # Check bounds
        if (current[0] < 0 or current[0] >= board_height or 
            current[1] < 0 or current[1] >= board_width):
            continue
        
        visited.add(current)
        
        # Add neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor not in visited:
                queue.append(neighbor)
    
    return len(visited)

def get_safe_moves_static(board_state: Dict, my_snake: Dict, opponents: List[Dict], 
                         current_direction: str = None) -> List[str]:
    """Get safe moves with improved 180-degree prevention"""
    moves = []
    move_deltas = {
        'up': (-1, 0), 'down': (1, 0),
        'left': (0, -1), 'right': (0, 1)
    }
    
    # Build obstacles set
    obstacles = set(my_snake['body'][:-1])  # Exclude tail as it will move
    for opponent in opponents:
        obstacles.update(opponent['body'])
    
    # Add static obstacles from the map
    obstacles.update(board_state.get('obstacles', []))
    
    board_height = board_state.get('height', 11)
    board_width = board_state.get('width', 11)
    my_head = my_snake['head']
    
    # Get opposite direction to prevent 180-degree turns
    forbidden_direction = get_opposite_direction(current_direction) if current_direction else None
    
    for move, (dx, dy) in move_deltas.items():
        new_head = (my_head[0] + dx, my_head[1] + dy)
        
        # Check bounds (walls)
        if (new_head[0] < 0 or new_head[0] >= board_height or
            new_head[1] < 0 or new_head[1] >= board_width):
            continue
        
        # Check collisions with obstacles, bodies, etc.
        if new_head not in obstacles:
            moves.append(move)
    
    # If only 180-degree turn is available and we're not trapped, allow it
    if not moves and forbidden_direction:
        new_head = (my_head[0] + move_deltas[forbidden_direction][0], 
                   my_head[1] + move_deltas[forbidden_direction][1])
        if (0 <= new_head[0] < board_height and 
            0 <= new_head[1] < board_width and 
            new_head not in obstacles):
            moves.append(forbidden_direction)
    
    return moves

def manhattan_distance_static(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Static Manhattan distance calculation"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class OptimizedSnakeAI:
    def __init__(self):
        self.state = SnakeState.EXPLORING
        self.last_move_time = 0
        self.last_direction = None
        self.last_actual_move = None  # Track the actual move made, even for length-1 snake
        self.direction_history = []  # Track recent directions to prevent loops
        self.stuck_counter = 0  # Track if we're stuck in a pattern
        self.move_count = 0  # Track total moves made
        
        # NEW: Enhanced stuck detection and recovery
        self.position_history = []  # Track recent positions
        self.failsafe_move_counter = 0  # Count moves since failsafe was triggered
        self.failsafe_move = None  # The current failsafe move being tried
        self.tried_failsafe_moves = set()  # Keep track of tried failsafe moves
        self.last_stuck_position = None  # Position where we got stuck
        
        self.cached_safe_moves = {}
        
        self.state_weights = {
            SnakeState.AGGRESSIVE: {
                'food_priority': 2.0,  
                'opponent_blocking': 1.8,
                'survival_weight': 1.0,  
                'risk_tolerance': 0.9
            },
            SnakeState.DEFENSIVE: {
                'food_priority': 1.5,  
                'opponent_blocking': 0.3,
                'survival_weight': 2.0,  
                'risk_tolerance': 0.4
            },
            SnakeState.EXPLORING: {
                'food_priority': 1.8,  
                'opponent_blocking': 0.7,
                'survival_weight': 1.0,  
                'risk_tolerance': 0.7
            },
            SnakeState.ESCAPING: {
                'food_priority': 0.5,
                'opponent_blocking': 0.1,
                'survival_weight': 2.0,  
                'risk_tolerance': 0.3
            },
            SnakeState.ENDGAME: {
                'food_priority': 2.5,  
                'opponent_blocking': 1.5,
                'survival_weight': 1.5,  
                'risk_tolerance': 0.8
            }
        }
    
    def detect_position_loops(self, current_position: Tuple[int, int]) -> bool:
        """Detect if we're stuck in a position loop"""
        self.position_history.append(current_position)
        
        # Keep only recent positions
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Check for position repetition 
        if len(self.position_history) >= 6:
            recent_positions = self.position_history[-6:]
            position_counts = {}
            for pos in recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            max_visits = max(position_counts.values())
            if max_visits >= 3:
                return True
        
        return False
    
    def get_failsafe_move(self, safe_moves: List[str], board_state: Dict, my_snake: Dict) -> str:
        """Get a failsafe move to break out of stuck patterns"""
        if not safe_moves:
            return None
        
        # Remove moves we've already tried as failsafe moves
        available_moves = [move for move in safe_moves if move not in self.tried_failsafe_moves]
        
        # If we've tried all moves, reset and try again
        if not available_moves:
            self.tried_failsafe_moves.clear()
            available_moves = safe_moves.copy()
        
        
        food_locations = board_state.get('food', [])
        my_head = my_snake['head']
        
        move_scores = {}
        for move in available_moves:
            score = 0
            
            # Bonus for moves not in recent history
            if move not in self.direction_history[-3:]:
                score += 50
            if move not in self.direction_history[-6:]:
                score += 25
            
            # Bonus for moves towards food
            if food_locations:
                closest_food = min(food_locations, key=lambda f: manhattan_distance_static(my_head, f))
                new_snake = simulate_move_static(my_snake, move)
                new_head = new_snake['head']
                
                old_dist = manhattan_distance_static(my_head, closest_food)
                new_dist = manhattan_distance_static(new_head, closest_food)
                
                if new_dist < old_dist:
                    score += 30
                elif new_dist > old_dist:
                    score -= 10
            
            # Bonus for moves that lead to more accessible space
            new_snake = simulate_move_static(my_snake, move)
            accessible_space = calculate_accessible_space_static(
                new_snake['head'], board_state, new_snake, []
            )
            score += accessible_space * 2
            
            move_scores[move] = score
        
        # Select the move with highest score
        best_move = max(available_moves, key=lambda m: move_scores.get(m, 0))
        self.tried_failsafe_moves.add(best_move)
        
        return best_move
    
    def update_state(self, board_state: Dict, my_snake: Dict, opponent_snake: Dict) -> None:
        """Enhanced state update with improved loop detection"""
        my_length = len(my_snake['body'])
        opponent_length = len(opponent_snake['body'])
        current_position = my_snake['head']
        
        # Get current direction - works for all snake lengths
        current_direction = get_current_direction(my_snake['body'], self.last_actual_move)
        
        # Track direction history for loop detection
        if current_direction:
            self.direction_history.append(current_direction)
            if len(self.direction_history) > 8:  # Keep longer history for length-1 snakes
                self.direction_history.pop(0)
        
        # Enhanced loop detection with position tracking
        direction_loop_detected = False
        position_loop_detected = self.detect_position_loops(current_position)
        
        # Direction-based loop detection
        if len(self.direction_history) >= 4:
            recent_dirs = self.direction_history[-4:]
            # Check for simple back-and-forth pattern
            if (recent_dirs[0] == recent_dirs[2] and recent_dirs[1] == recent_dirs[3] and
                recent_dirs[0] != recent_dirs[1]):
                direction_loop_detected = True
            # Check for length-1 specific oscillation (ABAB pattern)
            elif my_length == 1 and len(self.direction_history) >= 6:
                last_6 = self.direction_history[-6:]
                if (last_6[0] == last_6[2] == last_6[4] and 
                    last_6[1] == last_6[3] == last_6[5] and
                    last_6[0] != last_6[1]):
                    direction_loop_detected = True
        
        # Update stuck counter based on loop detection
        if direction_loop_detected or position_loop_detected:
            self.stuck_counter += 1
            if self.last_stuck_position != current_position:
                self.last_stuck_position = current_position
                # Reset failsafe tracking when we detect stuck at new position
                self.tried_failsafe_moves.clear()
                self.failsafe_move_counter = 0
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        # Quick survival check
        safe_moves = self.get_safe_moves(board_state, my_snake, [opponent_snake], current_direction)
        
        # State determination with enhanced loop detection for length-1
        if len(safe_moves) <= 1:
            self.state = SnakeState.ESCAPING
        elif self.stuck_counter > 2 or (my_length == 1 and self.stuck_counter > 1):
            self.state = SnakeState.AGGRESSIVE  # Be more aggressive when stuck, especially at length 1
        elif my_length > opponent_length + 3:
            self.state = SnakeState.AGGRESSIVE
        elif my_length < opponent_length - 3:
            self.state = SnakeState.DEFENSIVE
        else:
            self.state = SnakeState.EXPLORING
    
    def choose_move_sequential(self, board_state: Dict, my_snake: Dict, opponent_snake: Dict, 
                             safe_moves: List[str], weights: Dict) -> str:
        """Enhanced move evaluation with failsafe move handling"""
        if not safe_moves:
            return 'up'  # Emergency fallback
            
        if len(safe_moves) == 1:
            return safe_moves[0]  # Only one option
        
        #Failsafe move logic - if stuck counter is high, try failsafe moves
        if self.stuck_counter > 3:  
            if self.failsafe_move and self.failsafe_move_counter < 3:
                if self.failsafe_move in safe_moves:
                    self.failsafe_move_counter += 1
                    return self.failsafe_move
                else:
                    # Failsafe move is no longer safe, get new one
                    self.failsafe_move = None
                    self.failsafe_move_counter = 0
            
            
            if self.failsafe_move_counter >= 3 or not self.failsafe_move:
                new_failsafe = self.get_failsafe_move(safe_moves, board_state, my_snake)
                if new_failsafe:
                    print(f"Trying failsafe move: {new_failsafe} (stuck_counter: {self.stuck_counter})")
                    self.failsafe_move = new_failsafe
                    self.failsafe_move_counter = 1
                    return new_failsafe
        else:
            # Reset failsafe tracking when not stuck
            self.failsafe_move = None
            self.failsafe_move_counter = 0
        
        # Regular move selection logic
        best_move = safe_moves[0]
        best_score = float('-inf')
        
        food_locations = board_state.get('food', [])
        my_head = my_snake['head']
        my_length = len(my_snake['body'])
        
        for move in safe_moves:
            new_snake = simulate_move_static(my_snake, move)
            score = evaluate_position_static(board_state, new_snake, [opponent_snake], weights)
            
            # Look-ahead bonus for moves that keep options open
            future_safe_moves = get_safe_moves_static(board_state, new_snake, [opponent_snake], move)
            score += len(future_safe_moves) * 12
            
            # ENHANCED ANTI-LOOP for length-1 snakes
            if my_length == 1 and self.stuck_counter > 0:
                # Strong penalty for recent moves when stuck at length 1
                if move in self.direction_history[-3:]:
                    score -= 40  # Stronger penalty for length-1 loops
                
                # Extra bonus for breaking patterns at length 1
                if len(self.direction_history) >= 2 and move != self.direction_history[-1]:
                    score += 25
                    
                # Special bonus for moves that haven't been tried recently
                if move not in self.direction_history[-4:]:
                    score += 35
            elif self.stuck_counter > 0:
                # Normal anti-loop for longer snakes
                if move in self.direction_history[-2:]:
                    score -= 20
                if len(self.direction_history) >= 2 and move != self.direction_history[-1]:
                    score += 15
            
            # Direction continuity bonus (reduced when stuck or length 1)
            if move == self.last_direction and self.stuck_counter == 0 and my_length > 1:
                score += 8
            
            # ENHANCED: Food direction bonus with length-1 specific logic
            if food_locations:
                closest_food = min(food_locations, key=lambda f: manhattan_distance_static(my_head, f))
                new_head = new_snake['head']
                old_dist = manhattan_distance_static(my_head, closest_food)
                new_dist = manhattan_distance_static(new_head, closest_food)
                
                if new_dist < old_dist:  # Moving closer to food
                    score += 30 if my_length == 1 else 25  # Extra bonus at length 1
                elif new_dist > old_dist:  # Moving away from food
                    score -= 15 if my_length == 1 else 10  # Stronger penalty at length 1
            
            # Length-1 specific: Prefer moves that don't create immediate back-and-forth
            if my_length == 1 and self.last_actual_move:
                opposite_move = get_opposite_direction(self.last_actual_move)
                if move == opposite_move:
                    score -= 30  # Strong penalty for immediate reversals at length 1
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def get_safe_moves(self, board_state: Dict, my_snake: Dict, opponents: List[Dict], 
                      current_direction: str = None) -> List[str]:
        """Cached safe moves calculation"""
        # Create a simple cache key
        cache_key = (
            my_snake['head'],
            tuple(my_snake['body'][:5]),
            tuple(tuple(opp['body'][:5]) for opp in opponents),
            current_direction
        )
        
        if cache_key in self.cached_safe_moves:
            return self.cached_safe_moves[cache_key]
        
        result = get_safe_moves_static(board_state, my_snake, opponents, current_direction)
        
        # Keep cache small
        if len(self.cached_safe_moves) > 10:
            self.cached_safe_moves.clear()
        
        self.cached_safe_moves[cache_key] = result
        return result
    
    def choose_move(self, board_state: Dict, my_snake: Dict, opponent_snake: Dict) -> str:
        """Main move selection with enhanced stuck detection and recovery"""
        start_time = time.time()
        
        try:
            self.move_count += 1
            
            # Get current direction - now works for length-1 snakes too
            current_direction = get_current_direction(my_snake['body'], self.last_actual_move)
            
            # Update AI state with enhanced stuck detection
            self.update_state(board_state, my_snake, opponent_snake)
            weights = self.state_weights[self.state]
            
            # Get safe moves with proper 180-degree prevention
            safe_moves = self.get_safe_moves(board_state, my_snake, [opponent_snake], current_direction)
            
            # IMPROVED: If no safe moves found, try without 180-degree restriction
            if not safe_moves:
                print("No safe moves found! Trying all moves including reversals.")
                safe_moves = get_safe_moves_static(board_state, my_snake, [opponent_snake])
                
                if not safe_moves:
                    print("Still no safe moves! Emergency fallback.")
                    move = 'up'
                else:
                    move = safe_moves[0]
            else:
                # Choose best move (now includes failsafe logic)
                move = self.choose_move_sequential(board_state, my_snake, opponent_snake, safe_moves, weights)
            
            # Track the direction we chose
            self.last_direction = move
            self.last_actual_move = move  # Always track the actual move made
            
            # Track timing
            self.last_move_time = (time.time() - start_time) * 1000
            return move
            
        except Exception as e:
            print(f"AI Error in choose_move: {e}")
            # Ultimate fallback
            current_direction = get_current_direction(my_snake['body'], self.last_actual_move)
            safe_moves = get_safe_moves_static(board_state, my_snake, [opponent_snake], current_direction)
            move = safe_moves[0] if safe_moves else 'up'
            self.last_direction = move
            self.last_actual_move = move
            return move

# Global AI instance
ai = OptimizedSnakeAI()

def get_next_move(board_state, player_state, opponent_state):
    try:
        # Convert game format to internal AI format
        internal_board, my_snake, opponent_snake = convert_to_internal_format(
            board_state, player_state, opponent_state
        )
        
        # Use the AI to choose the best move
        return ai.choose_move(internal_board, my_snake, opponent_snake)
        
    except Exception as e:
        # Emergency fallback with improved safety check
        print(f"AI Critical Error: {e}")
        head_row = player_state["head_position"]["row"]
        head_col = player_state["head_position"]["col"]
        body = player_state["body"]
        
        # Get current direction from body with fallback to last move
        current_direction = None
        if len(body) >= 2:
            head = (body[0]["row"], body[0]["col"])
            neck = (body[1]["row"], body[1]["col"])
            
            row_diff = head[0] - neck[0]
            col_diff = head[1] - neck[1]
            
            if row_diff == -1:
                current_direction = 'up'
            elif row_diff == 1:
                current_direction = 'down'
            elif col_diff == -1:
                current_direction = 'left'
            elif col_diff == 1:
                current_direction = 'right'
        else:
            # For length-1 snake, try to get from AI's last move
            current_direction = ai.last_actual_move if hasattr(ai, 'last_actual_move') else None
        
        def is_safe_move(row, col):
            """Enhanced safety check"""
            # Check bounds
            if row < 0 or row >= board_state["rows"] or col < 0 or col >= board_state["cols"]:
                return False
            
            # Check collision with own body (excluding tail)
            for segment in body[:-1]:
                if segment["row"] == row and segment["col"] == col:
                    return False
            
            # Check collision with opponent body
            for segment in opponent_state["body"]:
                if segment["row"] == row and segment["col"] == col:
                    return False
            
            # Check collision with obstacles
            for obstacle_pos in board_state.get("obstacle_locations", []):
                if obstacle_pos[0] == row and obstacle_pos[1] == col:
                    return False
            
            return True
        
        # Get forbidden direction
        forbidden_direction = get_opposite_direction(current_direction) if current_direction else None
        
        # Try safe moves, avoiding 180-degree turns first
        for move, (dr, dc) in [('right', (0, 1)), ('down', (1, 0)), ('up', (-1, 0)), ('left', (0, -1))]:
            if move == forbidden_direction:
                continue
            if is_safe_move(head_row + dr, head_col + dc):
                return move
        
        # If no other option, allow reversal
        if forbidden_direction:
            dr, dc = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}[forbidden_direction]
            if is_safe_move(head_row + dr, head_col + dc):
                return forbidden_direction
        
        return 'up'  # Last resort

def set_player_name():
    return "LordOfKings"