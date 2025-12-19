"""
Events module for Connect4 AI Player.

This module handles event-driven communication with Connect4 Backend:
- RabbitMQ listener for move.made events
- Event processing and AI response triggering
- Event schema definitions
"""

from .event_schemas import MoveEvent
from .event_handler import EventHandler
from .rabbitmq_listener import RabbitMQListener

__all__ = [
    # Schemas
    'MoveEvent',
    
    # Handler
    'EventHandler',
    
    # Listener
    'RabbitMQListener',
]
