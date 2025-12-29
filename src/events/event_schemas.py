"""
Event schemas for RabbitMQ messages from Connect4 Backend.

These match the events published by Connect4 Backend:
- game.created
- move.made
- game.finished
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PlayerInfo(BaseModel):
    """Player information in events"""
    id: str
    name: str


class MoveInfo(BaseModel):
    """Move information in move.made event"""
    moveIndex: int
    column: int
    landedAt: Dict[str, int]
    token: str
    playerId: str
    timestamp: str
    thinkingTimeMs: float


class GameStateInfo(BaseModel):
    """Game state information in events"""
    gameId: str
    board: List[List[str]]
    currentToken: str
    moveIndex: int
    availableColumns: List[int]
    phase: str
    playerOneId: str
    playerTwoId: str
    lastMove: Optional[MoveInfo] = None
    timestamp: str


class MoveEvent(BaseModel):
    """
    Event published when a move is made.
    
    Routing key: move.made
    
    This is the primary event the AI Player listens to.
    Contains complete game state before and after the move.
    """
    eventId: str
    eventType: str  # "move.made"
    timestamp: str
    gameId: str
    nextPlayerId: str
    move: MoveInfo
    legalMoves: List[int]  # Legal moves BEFORE this move was made
    preState: GameStateInfo  # State before move
    postState: GameStateInfo  # State after move


def parse_event(event_type: str, data: Dict[str, Any]) -> Any:
    """
    Parse event data into appropriate schema.
    
    Args:
        event_type: Event type string
        data: Event data dictionary
    
    Returns:
        Parsed event object
    
    Raises:
        ValueError: If event type is unknown
    """

    if event_type == "move.made":
        return MoveEvent(**data)
    else:
        raise ValueError(f"Unknown event type: {event_type}")
