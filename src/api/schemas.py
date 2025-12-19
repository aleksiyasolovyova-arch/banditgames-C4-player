"""
Pydantic schemas for API requests and responses.

These schemas match the APIs of:
- Connect4 Backend
- Gameplay Logging Service
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Connect4 Backend Schemas
# =============================================================================

class MakeMoveRequest(BaseModel):
    """Request to make a move in Connect4 Backend"""
    player_id: str = Field(..., alias="playerId")
    column: int
    
    class Config:
        populate_by_name = True


class GameStateResponse(BaseModel):
    """Response from Connect4 Backend with game state"""
    id: str
    board: Dict[str, Any]
    player_one: Dict[str, Any] = Field(..., alias="playerOne")
    player_two: Dict[str, Any] = Field(..., alias="playerTwo")
    current_token: str = Field(..., alias="currentToken")
    current_player: Dict[str, Any] = Field(..., alias="currentPlayer")
    phase: str
    winner: Optional[Dict[str, Any]] = None
    available_columns: List[int] = Field(..., alias="availableColumns")
    
    class Config:
        populate_by_name = True


# =============================================================================
# Gameplay Logging Service Schemas
# =============================================================================

class MoveLogRequest(BaseModel):
    """Request to log a single move to Gameplay Logging Service"""
    
    # Game identification
    game_id: str
    game_type: str = "human_vs_ai"  # or "ai_vs_ai"
    session_id: Optional[str] = None
    
    # Move information
    move_index: int
    player: str  # "player1" or "player2"
    player_type: str  # "human" or "ai"
    action_taken: int  # Column played
    
    # Board state
    board_before: List[List[str]]
    board_after: List[List[str]]
    legal_actions: List[int]
    
    # MCTS statistics (for AI moves)
    skill_level: Optional[str] = None
    mcts_time_limit: Optional[float] = None
    mcts_num_rollouts: Optional[int] = None
    mcts_visit_counts: Optional[Dict[int, int]] = None
    mcts_q_values: Optional[Dict[int, float]] = None
    mcts_best_move: Optional[int] = None
    thinking_time_ms: float
    
    # Game outcome (filled after game ends)
    game_outcome: Optional[str] = None
    total_moves: Optional[int] = None
    game_duration_seconds: Optional[float] = None
    
    # Metadata
    timestamp: str
    is_self_play: bool = False
    
    # Reference agent stats (for comparison)
    reference_agent_move: Optional[int] = None
    reference_agent_stats: Optional[Dict[str, Any]] = None


class LogResponse(BaseModel):
    """Response from logging service"""
    status: str
    message: Optional[str] = None
    games_count: Optional[int] = None
