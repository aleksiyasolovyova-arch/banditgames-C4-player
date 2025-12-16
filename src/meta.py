"""
Configuration for MCTS and Connect4 Game.
Extended for comprehensive logging and dataset generation.
"""

import math


class GameMeta:
    """Game constants"""
    PLAYERS = {'none': 0, 'player1': 1, 'player2': 2}
    OUTCOMES = {'draw': 0, 'player1': 1, 'player2': 2}
    INF = float('inf')

    # Board dimensions
    DEFAULT_ROWS = 6
    DEFAULT_COLS = 7
    DEFAULT_CONNECT = 4


class MCTSMeta:
    """MCTS configuration for different skill levels"""

    # Default exploration constant (UCB1)
    EXPLORATION = math.sqrt(2)

    # Skill level configurations with varying computation budgets
    SKILL_LEVELS = {
        'easy': {
            'time_limit': 0.1,      # 100ms
            'exploration': 2.0,      # Higher = more random
            'rollout_limit': 100,    # Max rollouts
            'description': 'Beginner-friendly, makes occasional mistakes'
        },
        'medium': {
            'time_limit': 0.5,      # 500ms
            'exploration': math.sqrt(2),
            'rollout_limit': 500,
            'description': 'Balanced difficulty for casual play'
        },
        'hard': {
            'time_limit': 1.0,      # 1 second
            'exploration': 1.0,
            'rollout_limit': 2000,
            'description': 'Challenging for experienced players'
        },
        'expert': {
            'time_limit': 2.0,      # 2 seconds
            'exploration': 0.5,      # Lower = more deterministic
            'rollout_limit': 5000,
            'description': 'Maximum strength for dataset generation'
        }
    }

    # Dynamic difficulty adjustment thresholds
    DDA_THRESHOLDS = {
        'win_streak': 3,        # Consecutive wins to upgrade difficulty
        'loss_streak': 3,       # Consecutive losses to downgrade
        'win_rate_high': 0.7,   # Win rate threshold to upgrade
        'win_rate_low': 0.3     # Win rate threshold to downgrade
    }

    # In-game adaptation parameters
    ADAPTATION = {
        'struggling_threshold': 1.5,    # Time multiplier to detect struggling
        'rushing_threshold': 2.0,       # Max seconds for "rushing" detection
        'comfortable_threshold': 5.0,   # Seconds for "comfortable" pace
        'slow_threshold': 15.0,         # Seconds for "slow" pace

        'struggling_adjustment': 0.7,   # Reduce AI time when player struggles
        'rushing_adjustment': 1.3,      # Increase AI time when player rushes
        'comfortable_adjustment': 1.1,  # Slight increase for comfortable pace
        'slow_adjustment': 0.8          # Reduce for very slow players
    }


class SelfPlayMeta:
    """Configuration for self-play dataset generation"""

    # Default self-play parameters
    DEFAULT_NUM_GAMES = 1000

    # Noise levels for move variety
    NOISE_LEVELS = {
        'none': 0.0,        # Pure MCTS selection
        'low': 0.05,        # 5% random moves
        'medium': 0.1,      # 10% random moves
        'high': 0.2         # 20% random moves
    }

    # Temperature for softmax move selection
    # Higher = more random, Lower = more deterministic
    TEMPERATURES = {
        'deterministic': 0.0,
        'low': 0.5,
        'medium': 1.0,
        'high': 2.0,
        'random': float('inf')
    }

    # Skill level combinations for varied dataset
    SKILL_COMBINATIONS = [
        ('easy', 'easy'),
        ('easy', 'medium'),
        ('medium', 'medium'),
        ('medium', 'hard'),
        ('hard', 'hard'),
        ('hard', 'expert'),
        ('expert', 'expert')
    ]

    # Dataset version naming
    DATASET_VERSIONS = {
        'v1': {'games': 1000, 'description': 'Initial dataset'},
        'v2': {'games': 5000, 'description': 'Extended dataset'},
        'v3': {'games': 10000, 'description': 'Large dataset'},
        'v4': {'games': 50000, 'description': 'Production dataset'}
    }


class LoggingMeta:
    """Configuration for gameplay logging"""

    # What to log for each move
    MOVE_LOG_FIELDS = [
        'game_id',
        'move_index',
        'player',
        'action',  # column played
        'board_before',
        'board_after',
        'legal_actions',
        'utility_before',
        'utility_after',
        'thinking_time_ms',
        'timestamp'
    ]

    # MCTS statistics to log
    MCTS_LOG_FIELDS = [
        'skill_level',
        'time_limit',
        'actual_time',
        'num_rollouts',
        'best_move',
        'visit_counts',
        'q_values',
        'move_probabilities',
        'exploration_constant',
        'time_adjustment'
    ]

    # Fields for training data export
    TRAINING_DATA_FIELDS = [
        'game_id',
        'move_index',
        'board_state',  # Flattened or encoded
        'legal_actions',
        'action_taken',
        'visit_counts',
        'q_values',
        'game_outcome',
        'outcome_reward',
        'player_skill',
        'opponent_skill'
    ]