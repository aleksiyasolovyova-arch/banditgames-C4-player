"""
Monte Carlo Tree Search implementation with comprehensive statistics tracking.
"""

import random
import time
import math
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from ConnectState import ConnectState
from meta import GameMeta, MCTSMeta


@dataclass
class MoveStatistics:
    """Statistics for a single move option"""
    column: int
    visit_count: int
    q_value: float
    probability: float
    ucb_value: float


class Node:
    """
    MCTS tree node with comprehensive tracking.
    """

    def __init__(self, move: Optional[int], parent: Optional['Node']):
        self.move = move  # Move that led to this node
        self.parent = parent
        self.N = 0  # Visit count
        self.Q = 0  # Total value
        self.children: Dict[int, 'Node'] = {}
        self.outcome = GameMeta.PLAYERS['none']
        self.is_expanded = False

    def add_children(self, children: List['Node']) -> None:
        """Add child nodes"""
        for child in children:
            self.children[child.move] = child

    def value(self, explore: float = MCTSMeta.EXPLORATION) -> float:
        """
        Calculate UCB1 value for node selection.

        Args:
            explore: Exploration constant (higher = more exploration)

        Returns:
            UCB1 value
        """
        if self.N == 0:
            return 0 if explore == 0 else GameMeta.INF

        exploitation = self.Q / self.N
        exploration = explore * math.sqrt(math.log(self.parent.N) / self.N)

        return exploitation + exploration

    def get_q_value(self) -> float:
        """Get average Q value"""
        return self.Q / self.N if self.N > 0 else 0.0


class MCTS:
    """
    Monte Carlo Tree Search with comprehensive statistics.

    Features:
    - Configurable exploration constant
    - Detailed move statistics for logging
    - Softmax temperature for move selection
    - Noise injection for variety
    - Tree reuse between moves
    """

    def __init__(
            self,
            state: ConnectState = None,
            exploration: float = MCTSMeta.EXPLORATION
    ):
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

        Returns:
            Tuple of (selected node, state at that node)
        """
        node = self.root
        state = deepcopy(self.root_state)
        depth = 0

        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value(self.exploration)).value(self.exploration)
            max_nodes = [n for n in children if n.value(self.exploration) == max_value]

            node = random.choice(max_nodes)
            state.move(node.move)
            depth += 1

            if node.N == 0:
                self.max_depth_reached = max(self.max_depth_reached, depth)
                return node, state

        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.move(node.move)
            depth += 1

        self.max_depth_reached = max(self.max_depth_reached, depth)
        return node, state

    def expand(self, parent: Node, state: ConnectState) -> bool:
        """
        Expand a node by adding all legal moves as children.

        Returns:
            True if expansion occurred, False if terminal
        """
        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)
        parent.is_expanded = True
        self.node_count += len(children)

        return True

    def roll_out(self, state: ConnectState) -> int:
        """
        Perform random rollout to terminal state.

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

        Args:
            node: Leaf node to start from
            turn: Player who was to play at leaf
            outcome: Game outcome (0=draw, 1=player1, 2=player2)
        """
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent

            if outcome == GameMeta.OUTCOMES['draw']:
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

        Returns:
            Column index of best move, or -1 if no moves
        """
        if self.root_state.game_over():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

        return best_child.move

    def select_move_with_temperature(self, temperature: float = 1.0) -> int:
        """
        Select move using softmax temperature.

        Args:
            temperature: Temperature parameter (0=greedy, inf=uniform)

        Returns:
            Selected column index
        """
        if self.root_state.game_over():
            return -1

        if temperature == 0:
            return self.best_move()

        visits = []
        moves = []

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
        """Get basic search statistics"""
        return self.num_rollouts, self.run_time

    def get_move_statistics(self) -> List[MoveStatistics]:
        """
        Get detailed statistics for all moves.

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
        """Get visit counts for all moves"""
        return {move: child.N for move, child in self.root.children.items()}

    def get_q_values(self) -> Dict[int, float]:
        """Get Q values for all moves"""
        return {move: child.get_q_value() for move, child in self.root.children.items()}

    def get_probabilities(self) -> Dict[int, float]:
        """Get move probabilities based on visit counts"""
        total = sum(child.N for child in self.root.children.values())
        if total == 0:
            return {}
        return {move: child.N / total for move, child in self.root.children.items()}

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get all statistics for logging.

        Returns:
            Dictionary with all search statistics
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
    # Tree Operations
    # =========================================================================

    def move(self, move: int) -> None:
        """
        Advance the tree by making a move, reusing the subtree.

        Args:
            move: Column index of move made
        """
        if move in self.root.children:
            self.root_state.move(move)
            self.root = self.root.children[move]
            self.root.parent = None
            return

        self.root_state.move(move)
        self.root = Node(None, None)

    def sync_state(self, state: ConnectState) -> None:
        """
        Synchronize tree with external game state by applying missing moves.

        This enables tree reuse when the game state has advanced
        (e.g., opponent made moves).

        Args:
            state: Current game state to sync with
        """
        # Get move counts to determine how many moves to apply
        tree_moves = len(self.root_state.move_history)
        state_moves = len(state.move_history)

        if state_moves > tree_moves:
            # Apply missing moves to catch up
            for move_info in state.move_history[tree_moves:]:
                col = move_info['column'] if isinstance(move_info, dict) else move_info.column
                self.move(col)
        elif state_moves < tree_moves or self.root_state.compute_hash() != state.compute_hash():
            # State is behind or diverged - reset tree
            self.reset(state)

    def reset(self, state: ConnectState = None) -> None:
        """Reset the search tree"""
        self.root_state = deepcopy(state) if state else ConnectState()
        self.root = Node(None, None)
        self.run_time = 0.0
        self.node_count = 0
        self.num_rollouts = 0
        self.max_depth_reached = 0