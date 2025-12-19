"""
Configuration for MCTS and Connect4 Game.
Centralized configuration for skill levels, DDA

This module provides:
- Skill level definitions with time budgets
- Dynamic Difficulty Adjustment (DDA) thresholds
- Logging configuration

"""

import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Game Constants
# =============================================================================

class GameConstants:
    """Core game constants for Connect Four"""
    
    # Board dimensions
    DEFAULT_ROWS: int = 6
    DEFAULT_COLS: int = 7
    DEFAULT_CONNECT: int = 4
    
    # Token representations
    EMPTY_TOKEN: str = "."
    PLAYER1_TOKEN: str = "X"
    PLAYER2_TOKEN: str = "O"
    
    # Player identifiers
    PLAYER_NONE: int = 0
    PLAYER_ONE: int = 1
    PLAYER_TWO: int = 2
    
    # Outcomes
    OUTCOME_DRAW: int = 0
    OUTCOME_PLAYER1: int = 1
    OUTCOME_PLAYER2: int = 2
    
    # Sentinel value
    INF: float = float('inf')


# =============================================================================
# Skill Level Configuration
# =============================================================================

@dataclass(frozen=True)
class SkillLevelConfig:
    """Configuration for a single skill level"""
    
    name: str
    time_limit: float  # seconds
    exploration: float  # UCB1 exploration constant
    rollout_limit: int  # Maximum rollouts
    description: str
    
    def __post_init__(self):
        """Validate configuration"""
        if self.time_limit <= 0:
            raise ValueError(f"time_limit must be positive, got {self.time_limit}")
        if self.exploration < 0:
            raise ValueError(f"exploration must be non-negative, got {self.exploration}")
        if self.rollout_limit <= 0:
            raise ValueError(f"rollout_limit must be positive, got {self.rollout_limit}")


class SkillLevel(str, Enum):
    """Available skill levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class MCTSConfig:
    """
    MCTS configuration for different skill levels.
    
    Skill levels are defined by:
    - time_limit: Computation time budget
    - exploration: UCB1 exploration constant (higher = more random)
    - rollout_limit: Maximum number of simulations
    """
    
    # Default exploration constant (UCB1)
    DEFAULT_EXPLORATION: float = math.sqrt(2)
    
    # Skill level configurations
    SKILL_CONFIGS: Dict[str, SkillLevelConfig] = {
        SkillLevel.EASY: SkillLevelConfig(
            name="easy",
            time_limit=0.1,  # 100ms
            exploration=2.0,  # High exploration = more random
            rollout_limit=100,
            description="Beginner-friendly, makes occasional mistakes"
        ),
        SkillLevel.MEDIUM: SkillLevelConfig(
            name="medium",
            time_limit=0.5,  # 500ms
            exploration=math.sqrt(2),
            rollout_limit=500,
            description="Balanced difficulty for casual play"
        ),
        SkillLevel.HARD: SkillLevelConfig(
            name="hard",
            time_limit=1.0,  # 1 second
            exploration=1.0,
            rollout_limit=2000,
            description="Challenging for experienced players"
        ),
        SkillLevel.EXPERT: SkillLevelConfig(
            name="expert",
            time_limit=2.0,  # 2 seconds
            exploration=0.5,  # Low exploration = more deterministic
            rollout_limit=5000,
            description="Maximum strength for dataset generation and reference"
        )
    }
    
    @classmethod
    def get_config(cls, skill_level: str) -> SkillLevelConfig:
        """
        Get configuration for a skill level.
        
        Args:
            skill_level: Skill level name
            
        Returns:
            SkillLevelConfig for the level
            
        Raises:
            ValueError: If skill level is invalid
        """
        if skill_level not in cls.SKILL_CONFIGS:
            raise ValueError(
                f"Invalid skill level: {skill_level}. "
                f"Valid levels: {list(cls.SKILL_CONFIGS.keys())}"
            )
        return cls.SKILL_CONFIGS[skill_level]
    
    @classmethod
    def get_all_levels(cls) -> List[str]:
        """Get list of all available skill levels"""
        return list(cls.SKILL_CONFIGS.keys())
    
    @classmethod
    def is_valid_level(cls, skill_level: str) -> bool:
        """Check if skill level is valid"""
        return skill_level in cls.SKILL_CONFIGS


# =============================================================================
# Dynamic Difficulty Adjustment (DDA)
# =============================================================================

@dataclass(frozen=True)
class DDAConfig:
    """
    Dynamic Difficulty Adjustment configuration.
    
    DDA adjusts AI skill level during gameplay based on player thinking time.
    The AI adapts in real-time by monitoring how long the player takes to make moves.
    """
    
    # In-game adaptation based on thinking time
    # Thresholds for detecting player behavior patterns
    struggling_threshold: float = 10.0  # Seconds - player is struggling if move takes this long
    rushing_threshold: float = 2.0      # Seconds - player is rushing if move is this fast
    comfortable_threshold: float = 5.0  # Seconds - comfortable thinking pace
    slow_threshold: float = 15.0        # Seconds - very slow/careful play
    
    # AI time budget adjustments based on player pace
    # These are multipliers applied to the AI's time_limit
    struggling_adjustment: float = 0.7   # Reduce AI time (easier) when player struggles
    rushing_adjustment: float = 1.3      # Increase AI time (harder) when player rushes
    comfortable_adjustment: float = 1.0  # No change for comfortable pace
    slow_adjustment: float = 0.9         # Slightly reduce for very slow players
    
    # History tracking for pattern detection
    max_history_length: int = 10  # Track last N moves for averaging thinking times
    min_moves_for_adjustment: int = 3  # Minimum moves before making adjustments
    
    # Adjustment limits to prevent extreme changes
    min_time_multiplier: float = 0.5  # Don't go below 50% of base time
    max_time_multiplier: float = 2.0  # Don't go above 200% of base time
    
    def __post_init__(self):
        """Validate DDA configuration"""
        if self.struggling_threshold <= self.comfortable_threshold:
            raise ValueError("struggling_threshold must be > comfortable_threshold")
        if self.rushing_threshold >= self.comfortable_threshold:
            raise ValueError("rushing_threshold must be < comfortable_threshold")
        if not 0 < self.min_time_multiplier <= 1.0:
            raise ValueError("min_time_multiplier must be between 0 and 1")
        if self.max_time_multiplier <= 1.0:
            raise ValueError("max_time_multiplier must be > 1")
        if self.min_moves_for_adjustment < 1:
            raise ValueError("min_moves_for_adjustment must be positive")


# =============================================================================
# Logging Configuration
# =============================================================================

@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for gameplay logging"""
    
    # Move log fields (what to log for each move)
    move_fields: List[str] = field(default_factory=lambda: [
        'game_id',
        'move_index',
        'player',
        'player_type',
        'action',  # column played
        'board_before',
        'board_after',
        'legal_actions',
        'thinking_time_ms',
        'timestamp'
    ])
    
    # MCTS statistics to log
    mcts_fields: List[str] = field(default_factory=lambda: [
        'skill_level',
        'time_limit',
        'actual_time',
        'num_rollouts',
        'best_move',
        'visit_counts',
        'q_values',
        'move_probabilities',
        'exploration_constant',
        'node_count',
        'max_depth'
    ])
    
    # Training data fields for ML export
    training_fields: List[str] = field(default_factory=lambda: [
        'game_id',
        'game_type',
        'session_id',
        'move_index',
        'board_state',
        'legal_actions',
        'action_taken',
        'visit_counts',
        'q_values',
        'game_outcome',
        'player_skill',
        'opponent_skill',
        'is_self_play'
    ])


# =============================================================================
# Application Configuration
# =============================================================================

@dataclass
class AppConfig:
    """Main application configuration"""
    
    # Service URLs
    connect4_backend_url: str = "http://localhost:8000"
    gameplay_logging_url: str = "http://localhost:8001"
    
    # RabbitMQ configuration
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_exchange: str = "connect4.events"
    
    # AI Player configuration
    default_skill_level: str = "medium"
    enable_dda: bool = True
    enable_reference_agent: bool = True
    reference_agent_skill: str = "expert"
    
    # Performance
    max_concurrent_games: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# =============================================================================
# Global Configuration Instances
# =============================================================================

# Create singleton instances
game_constants = GameConstants()
mcts_config = MCTSConfig()
dda_config = DDAConfig()
logging_config = LoggingConfig()
