"""
ConnectState adapter for MCTS algorithm.
Provides the interface between Connect4 game and MCTS.
Includes serialization and comprehensive state tracking.
"""

from typing import List, Dict, Optional, Any
import json
import hashlib
from copy import deepcopy


class ConnectState:
    """
    State representation for MCTS algorithm.

    This class provides:
    1. Complete game state representation
    2. Legal action generation
    3. State transitions
    4. Terminal state detection
    5. Serialization for logging
    """

    def __init__(
        self,
        board: List[List[str]] = None,
        rows: int = 6,
        cols: int = 7,
        connect: int = 4,
        current_player: int = 1,
        empty_token: str = ".",
        player1_token: str = "X",
        player2_token: str = "O",
        move_history: List[Dict] = None
    ):
        self.rows = rows
        self.cols = cols
        self.connect = connect
        self.empty_token = empty_token
        self.player1_token = player1_token
        self.player2_token = player2_token

        if board is None:
            self.board = [[empty_token for _ in range(cols)] for _ in range(rows)]
        else:
            self.board = [row[:] for row in board]

        self.to_play = current_player  # 1 for player1, 2 for player2
        self.move_history = move_history or []
        self._winner_cache = None
        self._legal_moves_cache = None

    # State Queries
    def get_legal_moves(self) -> List[int]:
        """Return list of valid column indices"""
        if self._legal_moves_cache is None:
            self._legal_moves_cache = [
                c for c in range(self.cols)
                if self.board[0][c] == self.empty_token
            ]
        return self._legal_moves_cache

    def game_over(self) -> bool:
        """Check if game is over"""
        return self.check_winner() != 0 or len(self.get_legal_moves()) == 0

    def is_terminal(self) -> bool:
        """Alias for game_over"""
        return self.game_over()

    def get_outcome(self) -> int:
        """
        Get game outcome.

        Returns:
            0: Draw or ongoing
            1: Player 1 wins
            2: Player 2 wins
        """
        winner = self.check_winner()
        if winner != 0:
            return winner

        if len(self.get_legal_moves()) == 0:
            return 0  # Draw

        return 0  # Game ongoing

    def check_winner(self) -> int:
        """
        Check for winner.

        Returns:
            0: No winner
            1: Player 1 wins
            2: Player 2 wins
        """
        if self._winner_cache is not None:
            return self._winner_cache

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == self.empty_token:
                    continue

                token = self.board[row][col]
                for dr, dc in directions:
                    count = 0
                    r, c = row, col

                    while (0 <= r < self.rows and 0 <= c < self.cols and
                           self.board[r][c] == token):
                        count += 1
                        if count >= self.connect:
                            self._winner_cache = 1 if token == self.player1_token else 2
                            return self._winner_cache
                        r += dr
                        c += dc

        self._winner_cache = 0
        return 0

    def get_current_player_token(self) -> str:
        """Get token for current player"""
        return self.player1_token if self.to_play == 1 else self.player2_token

    def get_opponent_token(self) -> str:
        """Get token for opponent"""
        return self.player2_token if self.to_play == 1 else self.player1_token

    # State Transitions
    def move(self, column: int) -> int:
        """
        Make a move in the specified column.

        Args:
            column: Column index to place piece

        Returns:
            Row where piece landed

        Raises:
            ValueError: If move is invalid
        """
        if column not in self.get_legal_moves():
            raise ValueError(f"Invalid move: column {column}")

        token = self.get_current_player_token()

        # Drop piece in column
        row = -1
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][column] == self.empty_token:
                self.board[r][column] = token
                row = r
                break

        # Record move
        self.move_history.append({
            'player': self.to_play,
            'column': column,
            'row': row
        })

        # Switch player
        self.to_play = 3 - self.to_play

        # Invalidate caches
        self._winner_cache = None
        self._legal_moves_cache = None

        return row

    def copy(self) -> 'ConnectState':
        """Create a deep copy of the state"""
        return ConnectState(
            board=deepcopy(self.board),
            rows=self.rows,
            cols=self.cols,
            connect=self.connect,
            current_player=self.to_play,
            empty_token=self.empty_token,
            player1_token=self.player1_token,
            player2_token=self.player2_token,
            move_history=deepcopy(self.move_history)
        )

    def clone(self) -> 'ConnectState':
        """Alias for copy"""
        return self.copy()

    # Evaluation
    def evaluate(self) -> float:
        """
        Evaluate position from current player's perspective.

        Returns:
            Float score: positive = good for current player
        """
        winner = self.check_winner()
        if winner == self.to_play:
            return 1000.0
        elif winner != 0:
            return -1000.0
        elif self.is_terminal():
            return 0.0  # Draw

        # Heuristic evaluation
        score = 0.0
        my_token = self.get_current_player_token()
        opp_token = self.get_opponent_token()

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(self.rows):
            for c in range(self.cols):
                for dr, dc in directions:
                    score += self._evaluate_window(r, c, dr, dc, my_token, opp_token)

        return score

    def _evaluate_window(
        self,
        start_r: int,
        start_c: int,
        dr: int,
        dc: int,
        my_token: str,
        opp_token: str
    ) -> float:
        """Evaluate a window of 'connect' size"""
        # Check if window fits
        end_r = start_r + (self.connect - 1) * dr
        end_c = start_c + (self.connect - 1) * dc

        if not (0 <= end_r < self.rows and 0 <= end_c < self.cols):
            return 0.0

        my_count = 0
        opp_count = 0

        for i in range(self.connect):
            r = start_r + i * dr
            c = start_c + i * dc
            cell = self.board[r][c]

            if cell == my_token:
                my_count += 1
            elif cell == opp_token:
                opp_count += 1

        # Mixed window = no potential
        if my_count > 0 and opp_count > 0:
            return 0.0

        if my_count > 0:
            return 10 ** (my_count - 1)
        elif opp_count > 0:
            return -(10 ** (opp_count - 1))

        return 0.0

    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            'board': self.board,
            'rows': self.rows,
            'cols': self.cols,
            'connect': self.connect,
            'current_player': self.to_play,
            'empty_token': self.empty_token,
            'player1_token': self.player1_token,
            'player2_token': self.player2_token,
            'move_history': self.move_history,
            'legal_moves': self.get_legal_moves(),
            'is_terminal': self.is_terminal(),
            'winner': self.check_winner()
        }

    def to_json(self) -> str:
        """Convert state to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConnectState':
        """Create state from dictionary"""
        return cls(
            board=data['board'],
            rows=data['rows'],
            cols=data['cols'],
            connect=data['connect'],
            current_player=data['current_player'],
            empty_token=data['empty_token'],
            player1_token=data['player1_token'],
            player2_token=data['player2_token'],
            move_history=data.get('move_history', [])
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ConnectState':
        """Create state from JSON string"""
        return cls.from_dict(json.loads(json_str))

    def compute_hash(self) -> str:
        """Compute unique hash for this state (board + player to move)."""
        board_str = json.dumps(self.board, sort_keys=True)
        hash_input = f"{board_str}:{self.to_play}"
        # IMPORTANT: call hexdigest(), not return the function object
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def to_flat_array(self) -> List[int]:
        """
        Convert board to flat array for ML models.

        Returns:
            List of integers: 0=empty, 1=player1, 2=player2
        """
        flat = []
        for row in self.board:
            for cell in row:
                if cell == self.empty_token:
                    flat.append(0)
                elif cell == self.player1_token:
                    flat.append(1)
                else:
                    flat.append(2)
        return flat

    def to_feature_planes(self) -> List[List[List[int]]]:
        """
        Convert board to feature planes for CNN models.

        Returns:
            3 planes: [current_player_pieces, opponent_pieces, empty_cells]
        """
        my_token = self.get_current_player_token()
        opp_token = self.get_opponent_token()

        my_plane = [[1 if cell == my_token else 0 for cell in row] for row in self.board]
        opp_plane = [[1 if cell == opp_token else 0 for cell in row] for row in self.board]
        empty_plane = [[1 if cell == self.empty_token else 0 for cell in row] for row in self.board]

        return [my_plane, opp_plane, empty_plane]


    def __str__(self) -> str:
        """String representation of the board"""
        lines = []
        for row in self.board:
            lines.append(' '.join(row))
        lines.append(' '.join(str(i) for i in range(self.cols)))
        lines.append(f"Player {self.to_play} to move")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"ConnectState(to_play={self.to_play}, moves={len(self.move_history)})"