"""
API module for Connect4 AI Player.

This module provides HTTP clients and event publishers for communicating with external services:
- Connect4 Backend (for making moves)
- RabbitMQ Event Publisher (for logging moves and statistics)
"""

from .schemas import (
    MakeMoveRequest,
    GameStateResponse,
)
from .backend_client import Connect4BackendClient
from .event_publisher import AIPlayerEventPublisher

__all__ = [
    # Schemas
    'MakeMoveRequest',
    'GameStateResponse',
    
    # Clients
    'Connect4BackendClient',
    'AIPlayerEventPublisher',
]
