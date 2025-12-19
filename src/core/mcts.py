"""
Monte Carlo Tree Search (MCTS) implementation.

The MCTS algorithm works by building a search tree through simulation:
1. Selection: Navigate tree using UCB1 formula
2. Expansion: Add child nodes for legal moves
3. Simulation: Random playout to terminal state
4. Backpropagation: Update statistics up the tree
"""

import random
import time
import math
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .game_state import ConnectState
from .config import GameConstants, MCTSConfig


@dataclass
class MoveStatistics:
    """Statistics for a single move option"""
    column: int
    visit_count: int
    q_value: float
    probability: float  # Based on visit count distribution
    ucb_value: float


class Node:
    """
    MCTS tree node.
    
    Each node represents a game state after a move.
    Tracks visit count (N) and total value (Q) for UCB1 calculation.
    """
    
    def __init__(self, move: Optional[int], parent: Optional['Node']):
        """
        Initialize tree node.
        
        Args:
            move: Move that led to this node (None for root)
            parent: Parent node (None for root)
        """
        self.move = move  # Column index of move that led here
        self.parent = parent
        self.N = 0  # Visit count
        self.Q = 0  # Total value (sum of rewards)
        self.children: Dict[int, 'Node'] = {}  # Map from move to child node
        self.outcome = GameConstants.PLAYER_NONE
        self.is_expanded = False
    
    def add_children(self, children: List['Node']) -> None:
        """
        Add child nodes to this node.
        
        Args:
            children: List of child nodes to add
        """
        for child in children:
            self.children[child.move] = child
    
    def value(self, explore: float = MCTSConfig.DEFAULT_EXPLORATION) -> float:
        """
        Calculate UCB1 value for this node.
        
        UCB1 = exploitation + exploration
             = Q/N + c * sqrt(ln(parent.N) / N)
        
        Args:
            explore: Exploration constant (higher = more exploration)
            
        Returns:
            UCB1 value for node selection
        """
        if self.N == 0:
            return 0 if explore == 0 else GameConstants.INF
        
        # Exploitation: average value
        exploitation = self.Q / self.N
        
        # Exploration: bonus for less-visited nodes
        exploration = explore * math.sqrt(math.log(self.parent.N) / self.N)
        
        return exploitation + exploration
    
    def get_q_value(self) -> float:
        """Get average Q value (exploitation term)"""
        return self.Q / self.N if self.N > 0 else 0.0


class MCTS:
    """
    Monte Carlo Tree Search with comprehensive statistics.
    
    Features:
    - Configurable exploration constant
    - Detailed move statistics for logging
    - Softmax temperature for move selection
    - Noise injection for variety
    - Tree reuse between moves via sync_state()
    
    The tree is reused across moves for efficiency. Call sync_state()
    to update the tree when the game state advances.
    """
    
    def __init__(
        self,
        state: Optional[ConnectState] = None,
        exploration: float = MCTSConfig.DEFAULT_EXPLORATION
    ):
        """
        Initialize MCTS.
        
        Args:
            state: Initial game state (creates empty board if None)
            exploration: UCB1 exploration constant
        """
        self.root_state = deepcopy(state) if state else ConnectState()
        self.root = Node(None, None)
        self.exploration = exploration
        
        # Statistics
        self.run_time = 0.0
        self.node_count = 0
        self.num_rollouts = 0
        self.max_depth_reached = 0
    
    # =========================================================================
    # Core MCTS Operations
    # =========================================================================
    
    def select_node(self) -> Tuple[Node, ConnectState]:
        """
        Select a node for expansion using UCB1.
        
        Navigate down the tree, always picking the child with highest UCB1 value.
        Stop when we reach a node that hasn't been fully expanded.
        
        Returns:
            Tuple of (selected node, state at that node)
        """
        node = self.root
        state = deepcopy(self.root_state)
        depth = 0
        
        # Navigate down tree using UCB1
        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value(self.exploration)).value(self.exploration)
            max_nodes = [n for n in children if n.value(self.exploration) == max_value]
            
            # Pick random child among max UCB1 nodes (tie-breaking)
            node = random.choice(max_nodes)
            state.move(node.move)
            depth += 1
            
            # If we hit an unvisited node, select it
            if node.N == 0:
                self.max_depth_reached = max(self.max_depth_reached, depth)
                return node, state
        
        # Try to expand this node
        if self.expand(node, state):
            # Pick random child from newly expanded
            node = random.choice(list(node.children.values()))
            state.move(node.move)
            depth += 1
        
        self.max_depth_reached = max(self.max_depth_reached, depth)
        return node, state
    
    def expand(self, parent: Node, state: ConnectState) -> bool:
        """
        Expand a node by adding all legal moves as children.
        
        Args:
            parent: Node to expand
            state: Game state at parent node
            
        Returns:
            True if expansion occurred, False if terminal state
        """
        if state.game_over():
            return False
        
        # Create child node for each legal move
        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)
        parent.is_expanded = True
        self.node_count += len(children)
        
        return True
    
    def roll_out(self, state: ConnectState) -> int:
        """
        Perform random rollout from state to terminal state.
        
        This is the simulation phase - play randomly until game ends.
        
        Args:
            state: State to roll out from
            
        Returns:
            Game outcome (0=draw, 1=player1, 2=player2)
        """
        rollout_state = deepcopy(state)
        
        while not rollout_state.game_over():
            legal_moves = rollout_state.get_legal_moves()
            rollout_state.move(random.choice(legal_moves))
        
        return rollout_state.get_outcome()
    
    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:
        """
        Backpropagate result through the tree.
        
        Update visit counts and values from leaf to root.
        Reward is 1 for winner, 0 for loser/draw.
        
        Args:
            node: Leaf node to start from
            turn: Player who was to play at leaf
            outcome: Game outcome (0=draw, 1=player1, 2=player2)
        """
        # Determine reward for the player at the leaf
        reward = 0 if outcome == turn else 1
        
        # Propagate up the tree
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            
            # Flip reward at each level (alternating players)
            if outcome == GameConstants.OUTCOME_DRAW:
                reward = 0
            else:
                reward = 1 - reward
    
    def search(self, time_limit: float) -> None:
        """
        Run MCTS search for specified time.
        
        Args:
            time_limit: Maximum search time in seconds
        """
        start_time = time.process_time()
        num_rollouts = 0
        
        while time.process_time() - start_time < time_limit:
            # MCTS iteration: select -> expand -> simulate -> backpropagate
            node, state = self.select_node()
            outcome = self.roll_out(state)
            self.back_propagate(node, state.to_play, outcome)
            num_rollouts += 1
        
        self.run_time = time.process_time() - start_time
        self.num_rollouts = num_rollouts
    
    def search_iterations(self, num_iterations: int) -> None:
        """
        Run MCTS search for specified number of iterations.
        
        Args:
            num_iterations: Number of rollouts to perform
        """
        start_time = time.process_time()
        
        for _ in range(num_iterations):
            node, state = self.select_node()
            outcome = self.roll_out(state)
            self.back_propagate(node, state.to_play, outcome)
        
        self.run_time = time.process_time() - start_time
        self.num_rollouts = num_iterations
    
    # =========================================================================
    # Move Selection
    # =========================================================================
    
    def best_move(self) -> int:
        """
        Select best move based on visit counts.
        
        The move with most visits is most promising.
        
        Returns:
            Column index of best move, or -1 if no legal moves
        """
        if self.root_state.game_over():
            return -1
        
        if not self.root.children:
            return -1
        
        # Select move with highest visit count
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)  # Tie-breaking
        
        return best_child.move
    
    def select_move_with_temperature(self, temperature: float = 1.0) -> int:
        """
        Select move using softmax temperature.
        
        Temperature controls randomness:
        - temperature = 0: Greedy (always best move)
        - temperature = 1: Softmax over visit counts
        - temperature = inf: Uniform random
        
        Args:
            temperature: Temperature parameter
            
        Returns:
            Selected column index
        """
        if self.root_state.game_over():
            return -1
        
        if not self.root.children:
            return -1
        
        if temperature == 0:
            return self.best_move()
        
        moves = []
        visits = []
        
        for move, child in self.root.children.items():
            moves.append(move)
            visits.append(child.N)
        
        if temperature == float('inf'):
            # Uniform random
            return random.choice(moves)
        
        # Softmax with temperature
        visits = [v ** (1.0 / temperature) for v in visits]
        total = sum(visits)
        probs = [v / total for v in visits]
        
        return random.choices(moves, weights=probs, k=1)[0]
    
    def select_move_with_noise(self, noise_level: float = 0.0) -> int:
        """
        Select move with optional noise for variety.
        
        With probability noise_level, pick a random move.
        Otherwise, pick best move.
        
        Args:
            noise_level: Probability of random move (0-1)
            
        Returns:
            Selected column index
        """
        if random.random() < noise_level:
            # Random legal move
            legal = self.root_state.get_legal_moves()
            return random.choice(legal) if legal else -1
        
        return self.best_move()
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def statistics(self) -> Tuple[int, float]:
        """
        Get basic search statistics.
        
        Returns:
            Tuple of (num_rollouts, run_time)
        """
        return self.num_rollouts, self.run_time
    
    def get_move_statistics(self) -> List[MoveStatistics]:
        """
        Get detailed statistics for all legal moves.
        
        Returns:
            List of MoveStatistics for each legal move
        """
        stats = []
        total_visits = sum(child.N for child in self.root.children.values())
        
        for move, child in sorted(self.root.children.items()):
            probability = child.N / total_visits if total_visits > 0 else 0.0
            stats.append(MoveStatistics(
                column=move,
                visit_count=child.N,
                q_value=child.get_q_value(),
                probability=probability,
                ucb_value=child.value(self.exploration) if child.N > 0 else 0.0
            ))
        
        return stats
    
    def get_visit_counts(self) -> Dict[int, int]:
        """
        Get visit counts for all moves.
        
        Returns:
            Dictionary mapping column to visit count
        """
        return {move: child.N for move, child in self.root.children.items()}
    
    def get_q_values(self) -> Dict[int, float]:
        """
        Get Q values for all moves.
        
        Returns:
            Dictionary mapping column to average Q value
        """
        return {move: child.get_q_value() for move, child in self.root.children.items()}
    
    def get_probabilities(self) -> Dict[int, float]:
        """
        Get move probabilities based on visit counts.
        
        Returns:
            Dictionary mapping column to probability
        """
        total = sum(child.N for child in self.root.children.values())
        if total == 0:
            return {}
        return {move: child.N / total for move, child in self.root.children.items()}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get all statistics for logging.
        
        Returns:
            Dictionary with complete search statistics
        """
        move_stats = self.get_move_statistics()
        
        return {
            'num_rollouts': self.num_rollouts,
            'run_time': self.run_time,
            'node_count': self.node_count,
            'max_depth': self.max_depth_reached,
            'exploration': self.exploration,
            'best_move': self.best_move(),
            'visit_counts': self.get_visit_counts(),
            'q_values': self.get_q_values(),
            'probabilities': self.get_probabilities(),
            'move_stats': [
                {
                    'column': ms.column,
                    'visit_count': ms.visit_count,
                    'q_value': ms.q_value,
                    'probability': ms.probability,
                    'ucb_value': ms.ucb_value
                }
                for ms in move_stats
            ]
        }
    
    # =========================================================================
    # Tree Operations (Tree Reuse)
    # =========================================================================
    
    def move(self, move: int) -> None:
        """
        Advance the tree by making a move, reusing the subtree.
        
        If the move exists in children, reuse that subtree.
        Otherwise, create new root.
        
        Args:
            move: Column index of move made
        """
        if move in self.root.children:
            # Reuse subtree
            self.root_state.move(move)
            self.root = self.root.children[move]
            self.root.parent = None
            return
        
        # Move not in tree - create new root
        self.root_state.move(move)
        self.root = Node(None, None)
    
    def sync_state(self, state: ConnectState) -> None:
        """
        Synchronize tree with external game state.
        
        This enables tree reuse when the game state has advanced
        (e.g., opponent made moves, multiple moves since last sync).
        
        Algorithm:
        1. If state is ahead: Apply missing moves to catch up
        2. If state is behind or diverged: Reset tree
        
        Args:
            state: Current game state to sync with
        """
        # Get move counts
        tree_moves = len(self.root_state.move_history)
        state_moves = len(state.move_history)
        
        if state_moves > tree_moves:
            # State is ahead - apply missing moves
            for move_info in state.move_history[tree_moves:]:
                # Extract column from move history
                col = move_info['column'] if isinstance(move_info, dict) else move_info
                self.move(col)
        elif state_moves < tree_moves or self.root_state.compute_hash() != state.compute_hash():
            # State is behind or diverged - reset tree
            self.reset(state)
    
    def reset(self, state: Optional[ConnectState] = None) -> None:
        """
        Reset the search tree.
        
        Args:
            state: New state to set (creates empty board if None)
        """
        self.root_state = deepcopy(state) if state else ConnectState()
        self.root = Node(None, None)
        self.run_time = 0.0
        self.node_count = 0
        self.num_rollouts = 0
        self.max_depth_reached = 0
