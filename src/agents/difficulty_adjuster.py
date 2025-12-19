"""
Dynamic Difficulty Adjustment (DDA) for Connect4 AI.

Adjusts AI difficulty in real-time based on player thinking time.
The system monitors how long the player takes to make moves and
adjusts the AI's computation budget accordingly.

Strategy:
- Player struggling (slow moves) → Easier AI (less time)
- Player rushing (fast moves) → Harder AI (more time)
- Player comfortable → Maintain difficulty
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.config import DDAConfig

logger = logging.getLogger(__name__)


class PlayerBehavior(str, Enum):
    """Detected player behavior patterns"""
    STRUGGLING = "struggling"  # Taking very long to move
    RUSHING = "rushing"  # Moving very quickly
    COMFORTABLE = "comfortable"  # Normal pace
    SLOW = "slow"  # Slower than comfortable but not struggling


@dataclass
class MoveTimeRecord:
    """Record of a single move's timing"""
    move_index: int
    thinking_time: float  # seconds
    
    
class DifficultyAdjuster:
    """
    Dynamic Difficulty Adjustment based on player thinking time.
    
    Monitors player move times and adjusts AI time budget to create
    better gameplay experience.
    
    Usage:
        adjuster = DifficultyAdjuster()
        
        # Record player move
        adjuster.record_player_move(thinking_time=8.5)
        
        # Get time multiplier for AI
        multiplier = adjuster.get_time_multiplier()
        
        # AI uses: actual_time = base_time * multiplier
    """
    
    def __init__(self, config: Optional[DDAConfig] = None):
        """
        Initialize difficulty adjuster.
        
        Args:
            config: DDA configuration (uses default if None)
        """
        self.config = config or DDAConfig()
        
        # Move history
        self.move_history: List[MoveTimeRecord] = []
        
        # Current adjustment
        self.current_multiplier = 1.0
        
        logger.info("DifficultyAdjuster initialized")
    
    def record_player_move(self, thinking_time: float, move_index: Optional[int] = None) -> None:
        """
        Record a player's move time.
        
        Args:
            thinking_time: Time taken for move (seconds)
            move_index: Optional move index
        """
        if move_index is None:
            move_index = len(self.move_history)
        
        record = MoveTimeRecord(
            move_index=move_index,
            thinking_time=thinking_time
        )
        
        self.move_history.append(record)
        
        # Keep history limited
        if len(self.move_history) > self.config.max_history_length:
            self.move_history.pop(0)
        
        logger.debug(f"Recorded player move: {thinking_time:.2f}s")
        
        # Update multiplier if we have enough history
        if len(self.move_history) >= self.config.min_moves_for_adjustment:
            self._update_multiplier()
    
    def _update_multiplier(self) -> None:
        """Update time multiplier based on recent move history"""
        # Get recent average thinking time
        recent_times = [m.thinking_time for m in self.move_history[-5:]]
        avg_time = sum(recent_times) / len(recent_times)
        
        # Detect behavior pattern
        behavior = self._detect_behavior(avg_time)
        
        # Adjust multiplier based on behavior
        old_multiplier = self.current_multiplier
        
        if behavior == PlayerBehavior.STRUGGLING:
            # Player struggling → Make AI easier
            self.current_multiplier = self.config.struggling_adjustment
        elif behavior == PlayerBehavior.RUSHING:
            # Player rushing → Make AI harder
            self.current_multiplier = self.config.rushing_adjustment
        elif behavior == PlayerBehavior.SLOW:
            # Player slow → Make AI slightly easier
            self.current_multiplier = self.config.slow_adjustment
        else:  # COMFORTABLE
            # Player comfortable → Keep current difficulty
            self.current_multiplier = self.config.comfortable_adjustment
        
        # Apply limits
        self.current_multiplier = max(
            self.config.min_time_multiplier,
            min(self.config.max_time_multiplier, self.current_multiplier)
        )
        
        if abs(self.current_multiplier - old_multiplier) > 0.01:
            logger.info(
                f"DDA adjustment: {behavior.value} detected "
                f"(avg_time={avg_time:.1f}s) → multiplier={self.current_multiplier:.2f}"
            )
    
    def _detect_behavior(self, avg_thinking_time: float) -> PlayerBehavior:
        """
        Detect player behavior from average thinking time.
        
        Args:
            avg_thinking_time: Average recent thinking time (seconds)
        
        Returns:
            Detected behavior pattern
        """
        if avg_thinking_time >= self.config.struggling_threshold:
            return PlayerBehavior.STRUGGLING
        elif avg_thinking_time >= self.config.slow_threshold:
            return PlayerBehavior.SLOW
        elif avg_thinking_time <= self.config.rushing_threshold:
            return PlayerBehavior.RUSHING
        else:
            return PlayerBehavior.COMFORTABLE
    
    def get_time_multiplier(self) -> float:
        """
        Get current time multiplier for AI.
        
        Returns:
            Time multiplier (AI time = base_time * multiplier)
        """
        return self.current_multiplier
    
    def get_behavior(self) -> Optional[PlayerBehavior]:
        """
        Get current detected player behavior.
        
        Returns:
            Current behavior or None if not enough data
        """
        if len(self.move_history) < self.config.min_moves_for_adjustment:
            return None
        
        recent_times = [m.thinking_time for m in self.move_history[-5:]]
        avg_time = sum(recent_times) / len(recent_times)
        return self._detect_behavior(avg_time)
    
    def get_average_thinking_time(self) -> Optional[float]:
        """
        Get average player thinking time.
        
        Returns:
            Average thinking time or None if no history
        """
        if not self.move_history:
            return None
        
        times = [m.thinking_time for m in self.move_history]
        return sum(times) / len(times)
    
    def reset(self) -> None:
        """Reset adjuster state"""
        self.move_history.clear()
        self.current_multiplier = 1.0
        logger.debug("DifficultyAdjuster reset")
    
    def get_stats(self) -> dict:
        """
        Get statistics for monitoring.
        
        Returns:
            Dictionary with DDA statistics
        """
        return {
            'current_multiplier': self.current_multiplier,
            'move_count': len(self.move_history),
            'avg_thinking_time': self.get_average_thinking_time(),
            'current_behavior': self.get_behavior().value if self.get_behavior() else None
        }
