"""
ConnectState - Game state representation for Connect Four.

This is the core game state class used by MCTS and agents.

"""

from typing import List, Dict, Optional, Any, Tuple
import json
import hashlib
from copy import deepcopy

from .config import GameConstants


class ConnectState:
    """
    Complete game state representation for Connect Four.
    
    This class provides:
    1. Complete game state representation (board, current player, history)
    2. Legal action generation
    3. State transitions (making moves)
    4. Terminal state detection (win/loss/draw)
    5. Serialization for logging and ML
    6. State evaluation heuristics
    
    The state is self-contained and can be used standalone (for self-play)
    or synchronized with external game state (for live games).
    """
    
    def __init__(
        self,
        board: Optional[List[List[str]]] = None,
        rows: int = GameConstants.DEFAULT_ROWS,
        cols: int = GameConstants.DEFAULT_COLS,
        connect: int = GameConstants.DEFAULT_CONNECT,
        current_player: int = GameConstants.PLAYER_ONE,
        empty_token: str = GameConstants.EMPTY_TOKEN,
        player1_token: str = GameConstants.PLAYER1_TOKEN,
        player2_token: str = GameConstants.PLAYER2_TOKEN,
        move_history: Optional[List[Dict]] = None
    ):
        """
        Initialize game state.
        
        Args:
            board: Optional pre-existing board (for cloning)
            rows: Number of rows on board
            cols: Number of columns on board
            connect: Number in a row needed to win
            current_player: Player to move (1 or 2)
            empty_token: Character representing empty cell
            player1_token: Character for player 1
            player2_token: Character for player 2
            move_history: Optional move history
        """
        self.rows = rows
        self.cols = cols
        self.connect = connect
        self.empty_token = empty_token
        self.player1_token = player1_token
        self.player2_token = player2_token
        
        # Initialize board
        if board is None:
            self.board = [[empty_token for _ in range(cols)] for _ in range(rows)]
        else:
            self.board = [row[:] for row in board]  # Deep copy rows
        
        self.to_play = current_player  # 1 for player1, 2 for player2
        self.move_history = move_history or []
        
        # Cache for expensive computations
        self._winner_cache: Optional[int] = None
        self._legal_moves_cache: Optional[List[int]] = None
    
    # =========================================================================
    # State Queries
    # =========================================================================
    
    def get_legal_moves(self) -> List[int]:
        """
        Get list of legal column indices.
        
        Returns:
            List of column indices where a piece can be placed
        """
        if self._legal_moves_cache is None:
            self._legal_moves_cache = [
                c for c in range(self.cols)
                if self.board[0][c] == self.empty_token
            ]
        return self._legal_moves_cache
    
    def is_column_available(self, column: int) -> bool:
        """Check if a column can accept a piece"""
        if not 0 <= column < self.cols:
            return False
        return self.board[0][column] == self.empty_token
    
    def find_landing_row(self, column: int) -> Optional[int]:
        """
        Find the row where a piece would land if dropped in column.
        
        Args:
            column: Column index
            
        Returns:
            Row index where piece lands, or None if column full
        """
        if not self.is_column_available(column):
            return None
        
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][column] == self.empty_token:
                return row
        return None
    
    def game_over(self) -> bool:
        """Check if game is finished"""
        return self.check_winner() != GameConstants.PLAYER_NONE or len(self.get_legal_moves()) == 0
    
    def is_terminal(self) -> bool:
        """Alias for game_over()"""
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
        if winner != GameConstants.PLAYER_NONE:
            return winner
        
        if len(self.get_legal_moves()) == 0:
            return GameConstants.OUTCOME_DRAW
        
        return GameConstants.PLAYER_NONE  # Game ongoing
    
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
                            winner = (GameConstants.PLAYER_ONE if token == self.player1_token 
                                     else GameConstants.PLAYER_TWO)
                            self._winner_cache = winner
                            return winner
                        r += dr
                        c += dc
        
        self._winner_cache = GameConstants.PLAYER_NONE
        return GameConstants.PLAYER_NONE
    
    def get_current_player_token(self) -> str:
        """Get token for current player"""
        return self.player1_token if self.to_play == GameConstants.PLAYER_ONE else self.player2_token
    
    def get_opponent_token(self) -> str:
        """Get token for opponent"""
        return self.player2_token if self.to_play == GameConstants.PLAYER_ONE else self.player1_token
    
    # =========================================================================
    # State Transitions
    # =========================================================================
    
    def move(self, column: int) -> int:
        """
        Make a move in the specified column.
        
        This MODIFIES the current state (not functional).
        
        Args:
            column: Column index to place piece
            
        Returns:
            Row where piece landed
            
        Raises:
            ValueError: If move is invalid
        """
        if column not in self.get_legal_moves():
            raise ValueError(f"Invalid move: column {column} is not available")
        
        token = self.get_current_player_token()
        
        # Drop piece in column
        row = self.find_landing_row(column)
        if row is None:
            raise ValueError(f"No space in column {column}")
        
        self.board[row][column] = token
        
        # Record move
        self.move_history.append({
            'player': self.to_play,
            'column': column,
            'row': row,
            'token': token
        })
        
        # Switch player
        self.to_play = 3 - self.to_play  # Toggles between 1 and 2
        
        # Invalidate caches
        self._winner_cache = None
        self._legal_moves_cache = None
        
        return row
    
    def copy(self) -> 'ConnectState':
        """
        Create a deep copy of the state.
        
        Returns:
            New ConnectState with same game state
        """
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
        """Alias for copy()"""
        return self.copy()
    
    # =========================================================================
    # Evaluation (Heuristic for non-terminal states)
    # =========================================================================
    
    def evaluate(self) -> float:
        """
        Evaluate position from current player's perspective.
        
        Returns positive score for positions favoring current player,
        negative for positions favoring opponent.
        
        Returns:
            Float score: positive = good for current player
        """
        winner = self.check_winner()
        if winner == self.to_play:
            return 1000.0  # Current player wins
        elif winner != GameConstants.PLAYER_NONE:
            return -1000.0  # Opponent wins
        elif self.is_terminal():
            return 0.0  # Draw
        
        # Heuristic evaluation for non-terminal states
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
        """
        Evaluate a window of 'connect' size.
        
        Returns positive score for windows with only our pieces,
        negative for windows with only opponent pieces.
        """
        # Check if window fits on board
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
        
        # Mixed window has no potential
        if my_count > 0 and opp_count > 0:
            return 0.0
        
        # Score based on number of pieces
        if my_count > 0:
            return 10 ** (my_count - 1)
        elif opp_count > 0:
            return -(10 ** (opp_count - 1))
        
        return 0.0
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.
        
        Returns:
            Dictionary representation of state
        """
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
        """
        Create state from dictionary.
        
        Args:
            data: Dictionary with state data
            
        Returns:
            New ConnectState instance
        """
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
        """
        Compute unique hash for this state.
        
        Hash is based on board configuration and current player.
        Used for detecting duplicate states and tree reuse.
        
        Returns:
            SHA256 hash string
        """
        board_str = json.dumps(self.board, sort_keys=True)
        hash_input = f"{board_str}:{self.to_play}"
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    
    # =========================================================================
    # ML Feature Extraction
    # =========================================================================
    
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

    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __str__(self) -> str:
        """
        Pretty print the board.
        
        Returns:
            String representation with column numbers
        """
        lines = []
        for row in self.board:
            lines.append(' '.join(row))
        lines.append(' '.join(str(i) for i in range(self.cols)))
        lines.append(f"Player {self.to_play} to move")
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return f"ConnectState(to_play={self.to_play}, moves={len(self.move_history)})"
