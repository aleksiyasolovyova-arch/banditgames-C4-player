"""
Core module for Connect4 AI Player.

This module contains the fundamental building blocks:
- Game state representation
- MCTS algorithm implementation
- Configuration management
"""

from .config import (
    GameConstants,
    SkillLevel,
    SkillLevelConfig,
    MCTSConfig,
    DDAConfig,
    LoggingConfig,
    AppConfig,
    game_constants,
    mcts_config,
    dda_config,
    logging_config
)
from .game_state import ConnectState
from .mcts import MCTS, Node, MoveStatistics

__all__ = [
    # Configuration
    'GameConstants',
    'SkillLevel',
    'SkillLevelConfig',
    'MCTSConfig',
    'DDAConfig',
    'LoggingConfig',
    'AppConfig',
    'game_constants',
    'mcts_config',
    'dda_config',
    'logging_config',
    
    # Game State
    'ConnectState',
    
    # MCTS
    'MCTS',
    'Node',
    'MoveStatistics',
]
