"""
MCTS Agent - Clean wrapper around MCTS algorithm.

This provides a high-level interface for using MCTS with:
- Skill level configuration
- Tree reuse between moves
- Comprehensive statistics tracking
- Multiple move selection strategies
"""

import logging
from typing import Optional, Tuple

from ..core.game_state import ConnectState
from ..core.mcts import MCTS
from ..core.config import MCTSConfig, SkillLevel
from .agent_result import AgentResult

logger = logging.getLogger(__name__)


class MCTSAgent:
    """
    MCTS-based AI agent.
    
    Wraps the MCTS algorithm with:
    - Skill level configuration
    - Tree reuse via sync_state()
    - Easy-to-use interface
    - Comprehensive result tracking
    
    Usage:
        agent = MCTSAgent(skill_level='medium')
        result = agent.get_move(game_state)
        move = result.move
    """
    
    def __init__(
        self,
        skill_level: str = SkillLevel.MEDIUM,
        noise_level: float = 0.0,
        temperature: float = 0.0
    ):
        """
        Initialize MCTS agent.
        
        Args:
            skill_level: Skill level ('easy', 'medium', 'hard', 'expert')
            noise_level: Probability of random move (0-1, for variety)
            temperature: Softmax temperature for move selection
        
        Raises:
            ValueError: If skill level is invalid
        """
        # Validate and get skill configuration
        if not MCTSConfig.is_valid_level(skill_level):
            raise ValueError(
                f"Invalid skill level: {skill_level}. "
                f"Valid levels: {MCTSConfig.get_all_levels()}"
            )
        
        self.skill_level = skill_level
        self.config = MCTSConfig.get_config(skill_level)
        self.noise_level = noise_level
        self.temperature = temperature
        
        # Create MCTS instance with tree reuse
        self.mcts = MCTS(
            state=None,  # Will be set on first move
            exploration=self.config.exploration
        )
        
        # Track time adjustments for DDA
        self.time_multiplier = 1.0
        
        logger.debug(
            f"Created MCTSAgent: skill={skill_level}, "
            f"time_limit={self.config.time_limit}s, "
            f"exploration={self.config.exploration}"
        )
    
    def get_move(
        self,
        state: ConnectState,
        time_multiplier: float = 1.0
    ) -> AgentResult:
        """
        Get best move for current state.
        
        Args:
            state: Current game state
            time_multiplier: Multiply time limit by this (for DDA)
        
        Returns:
            AgentResult with move and statistics
        """
        # Synchronize MCTS tree with current state
        self.mcts.sync_state(state)
        
        # Apply time adjustment
        adjusted_time_limit = self.config.time_limit * time_multiplier
        self.time_multiplier = time_multiplier
        
        # Run MCTS search
        self.mcts.search(time_limit=adjusted_time_limit)
        
        # Select move based on strategy
        if self.noise_level > 0:
            move = self.mcts.select_move_with_noise(self.noise_level)
        elif self.temperature > 0:
            move = self.mcts.select_move_with_temperature(self.temperature)
        else:
            move = self.mcts.best_move()
        
        # Get comprehensive statistics
        stats = self.mcts.get_comprehensive_stats()
        
        # Create result
        result = AgentResult(
            move=move,
            skill_level=self.skill_level,
            time_limit=self.config.time_limit,
            exploration=self.config.exploration,
            actual_time=self.mcts.run_time,
            num_rollouts=self.mcts.num_rollouts,
            node_count=self.mcts.node_count,
            max_depth=self.mcts.max_depth_reached,
            visit_counts=stats['visit_counts'],
            q_values=stats['q_values'],
            probabilities=stats['probabilities'],
            adjusted_time_limit=adjusted_time_limit if time_multiplier != 1.0 else None
        )
        
        logger.debug(
            f"Agent move: column={move}, rollouts={result.num_rollouts}, "
            f"time={result.actual_time:.3f}s"
        )
        
        return result
    
    def reset(self, state: Optional[ConnectState] = None) -> None:
        """
        Reset the agent's MCTS tree.
        
        Args:
            state: Optional new state to set
        """
        self.mcts.reset(state)
        self.time_multiplier = 1.0
        logger.debug("Agent reset")
    
    def set_time_multiplier(self, multiplier: float) -> None:
        """
        Set time multiplier for next move (DDA).
        
        Args:
            multiplier: Time adjustment factor
        """
        self.time_multiplier = multiplier
        logger.debug(f"Time multiplier set to {multiplier:.2f}")
    
    def __repr__(self) -> str:
        return f"MCTSAgent(skill={self.skill_level}, time={self.config.time_limit}s)"
