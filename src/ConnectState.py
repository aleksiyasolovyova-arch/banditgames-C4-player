"""
ConnectState adapter for MCTS algorithm
Provides the interface between Connect4 game and MCTS
"""

from typing import List


class ConnectState:
    """
    State representation for MCTS algorithm
    """

    def __init__(self, board: List[List[str]] = None, rows: int = 6, cols: int = 7,
                 connect: int = 4, current_player: int = 1,
                 empty_token: str = ".", player1_token: str = "X", player2_token: str = "O"):
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

    def get_legal_moves(self) -> List[int]:
        """Return list of valid column indices"""
        return [c for c in range(self.cols) if self.board[0][c] == self.empty_token]

    def move(self, column: int) -> None:
        """Make a move in the specified column"""
        if column not in self.get_legal_moves():
            raise ValueError(f"Invalid move: column {column}")

        token = self.player1_token if self.to_play == 1 else self.player2_token

        # Drop piece in column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][column] == self.empty_token:
                self.board[row][column] = token
                break

        # Switch player
        self.to_play = 3 - self.to_play

    def game_over(self) -> bool:
        """Check if game is over"""
        return self.check_winner() != 0 or len(self.get_legal_moves()) == 0

    def get_outcome(self) -> int:
        """Get game outcome: 0 for draw/ongoing, 1 for player1 win, 2 for player2 win"""
        winner = self.check_winner()
        if winner != 0:
            return winner

        if len(self.get_legal_moves()) == 0:
            return 0  # Draw

        return 0  # Game ongoing

    def check_winner(self) -> int:
        """Check for winner: 0 for none, 1 for player1, 2 for player2"""
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
                            return 1 if token == self.player1_token else 2
                        r += dr
                        c += dc

        return 0