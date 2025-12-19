"""
Result data classes for AI agents.

These classes encapsulate the results of agent decision-making,
including the chosen move and all MCTS statistics for logging.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class AgentResult:
    """
    Complete result from an agent's decision-making process.

    Contains:
    - The chosen move
    - MCTS statistics (visit counts, Q-values, etc.)
    - Timing information
    - Skill level information
    """

    # Move selection
    move: int  # Column index chosen

    # Skill configuration
    skill_level: str
    time_limit: float  # Configured time limit (seconds)
    exploration: float  # Exploration constant used

    # MCTS execution
    actual_time: float  # Actual time spent (seconds)
    num_rollouts: int  # Number of simulations performed
    node_count: int  # Total nodes in tree
    max_depth: int  # Maximum search depth reached

    # Move statistics (for ML)
    visit_counts: Dict[int, int]  # Column -> visit count
    q_values: Dict[int, float]  # Column -> average Q value
    probabilities: Dict[int, float]  # Column -> selection probability

    # Additional metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    adjusted_time_limit: Optional[float] = None  # If DDA applied

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'move': self.move,
            'skill_level': self.skill_level,
            'time_limit': self.time_limit,
            'exploration': self.exploration,
            'actual_time': self.actual_time,
            'thinking_time_ms': self.actual_time * 1000,
            'num_rollouts': self.num_rollouts,
            'node_count': self.node_count,
            'max_depth': self.max_depth,
            'visit_counts': self.visit_counts,
            'q_values': self.q_values,
            'probabilities': self.probabilities,
            'timestamp': self.timestamp,
            'adjusted_time_limit': self.adjusted_time_limit
        }

    def get_mcts_stats(self) -> Dict[str, Any]:
        """
        Get MCTS statistics for logging to Gameplay Logging Service.

        Returns complete statistics including all move evaluations.
        """
        return {
            'skill_level': self.skill_level,
            'time_limit': self.time_limit,
            'adjusted_time_limit': self.adjusted_time_limit,
            'actual_time': self.actual_time,
            'thinking_time_ms': self.actual_time * 1000,
            'num_rollouts': self.num_rollouts,
            'node_count': self.node_count,
            'max_depth': self.max_depth,
            'best_move': self.move,
            'visit_counts': self.visit_counts,
            'q_values': self.q_values,
            'move_probabilities': self.probabilities,
            'exploration_constant': self.exploration
        }


@dataclass
class MoveResult:
    """
    Result from AI Manager containing both adaptive and reference agent results.

    This is what the event handler receives and uses for:
    1. The actual move to play (from adaptive agent)
    2. Statistics for logging (from both agents)
    """

    # Adaptive agent (the one that actually plays)
    move: int
    adaptive_result: AgentResult

    # Reference agent (expert, for comparison)
    reference_result: Optional[AgentResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'move': self.move,
            'adaptive_agent': self.adaptive_result.to_dict()
        }

        if self.reference_result:
            result['reference_agent'] = self.reference_result.to_dict()

        return result

    def get_mcts_stats(self) -> Dict[str, Any]:
        """Get MCTS stats from adaptive agent (the one that played)"""
        return self.adaptive_result.get_mcts_stats()

    def get_reference_stats(self) -> Optional[Dict[str, Any]]:
        """Get MCTS stats from reference agent (for ML comparison)"""
        return self.reference_result.get_mcts_stats() if self.reference_result else None
