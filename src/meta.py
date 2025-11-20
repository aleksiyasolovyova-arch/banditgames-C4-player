"""
Configuration for MCTS and Connect4 Game
"""

import math


class GameMeta:
    """Game constants"""
    PLAYERS = {'none': 0, 'player1': 1, 'player2': 2}
    OUTCOMES = {'draw': 0, 'player1': 1, 'player2': 2}
    INF = float('inf')


class MCTSMeta:
    """MCTS configuration for different skill levels"""

    # Default exploration constant
    EXPLORATION = math.sqrt(2)

    # Skill level configurations with varying computation budgets
    SKILL_LEVELS = {
        'easy': {
            'time_limit': 0.1,  # 100ms
            'exploration': 2.0  # Higher = more random
        },
        'medium': {
            'time_limit': 0.5,  # 500ms
            'exploration': math.sqrt(2)
        },
        'hard': {
            'time_limit': 1.0,  # 1 second
            'exploration': 1.0
        },
        'expert': {
            'time_limit': 2.0,  # 2 seconds
            'exploration': 0.5  # Lower = more deterministic
        }
    }

    # Dynamic difficulty thresholds
    DDA_THRESHOLDS = {
        'win_streak': 3,      # Wins to upgrade
        'loss_streak': 3,     # Losses to downgrade
        'win_rate_high': 0.7, # Win rate to upgrade
        'win_rate_low': 0.3   # Win rate to downgrade
    }