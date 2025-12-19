"""
Agents module for Connect4 AI Player.

This module provides:
- MCTS agent wrapper with skill level configuration
- Dynamic Difficulty Adjustment (DDA)
- AI Manager for handling multiple agents (adaptive + reference)
- Result data classes for returning agent decisions
"""

from .agent_result import AgentResult, MoveResult
from .mcts_agent import MCTSAgent
from .difficulty_adjuster import DifficultyAdjuster, PlayerBehavior
from .ai_manager import AIManager

__all__ = [
    'AgentResult',
    'MoveResult',
    'MCTSAgent',
    'DifficultyAdjuster',
    'PlayerBehavior',
    'AIManager',
]
