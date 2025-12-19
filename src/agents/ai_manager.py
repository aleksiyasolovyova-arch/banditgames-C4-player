"""
AI Manager - Manages multiple MCTS agents for live gameplay.

Handles:
1. Adaptive Agent: Adjusts difficulty based on player performance (DDA)
2. Reference Agent: Always runs at expert level for comparison

The adaptive agent is what actually plays the game.
The reference agent runs in parallel to provide baseline statistics.
"""

import logging
from typing import Optional, Dict

from ..core.game_state import ConnectState
from ..core.config import SkillLevel
from .mcts_agent import MCTSAgent
from .difficulty_adjuster import DifficultyAdjuster
from .agent_result import MoveResult

logger = logging.getLogger(__name__)


class AIManager:
    """
    Manages dual agent system for live games.
    
    Features:
    - Adaptive agent with DDA (adjusts to player skill)
    - Reference agent (always expert, for comparison)
    - Per-game agent instances with tree reuse
    - Tracks player thinking times for DDA
    
    Usage:
        manager = AIManager()
        
        # Get move for a game
        result = manager.get_move(
            game_id="abc123",
            state=current_state,
            player_thinking_time=5.2
        )
        
        move = result.move  # Adaptive agent's choice
        adaptive_stats = result.adaptive_result  # Adaptive agent stats
        reference_stats = result.reference_result  # Expert agent stats
    """
    
    def __init__(
        self,
        default_skill: str = SkillLevel.MEDIUM,
        enable_reference_agent: bool = True,
        enable_dda: bool = True
    ):
        """
        Initialize AI Manager.
        
        Args:
            default_skill: Default skill level for new games
            enable_reference_agent: Whether to run reference agent
            enable_dda: Whether to enable dynamic difficulty adjustment
        """
        self.default_skill = default_skill
        self.enable_reference_agent = enable_reference_agent
        self.enable_dda = enable_dda
        
        # Per-game agents (game_id -> agents)
        self.adaptive_agents: Dict[str, MCTSAgent] = {}
        self.reference_agents: Dict[str, MCTSAgent] = {}
        self.difficulty_adjusters: Dict[str, DifficultyAdjuster] = {}
        
        logger.info(
            f"AIManager initialized: default_skill={default_skill}, "
            f"reference_agent={enable_reference_agent}, dda={enable_dda}"
        )
    
    def get_move(
        self,
        game_id: str,
        state: ConnectState,
        player_thinking_time: Optional[float] = None,
        skill_level: Optional[str] = None
    ) -> MoveResult:
        """
        Get AI move for a game.
        
        Args:
            game_id: Unique game identifier
            state: Current game state
            player_thinking_time: Last player move time (for DDA)
            skill_level: Override skill level (optional)
        
        Returns:
            MoveResult with move and statistics from both agents
        """
        # Ensure agents exist for this game
        self._ensure_agents(game_id, skill_level)
        
        # Record player thinking time for DDA
        if self.enable_dda and player_thinking_time is not None:
            self.difficulty_adjusters[game_id].record_player_move(player_thinking_time)
        
        # Get time multiplier from DDA
        time_multiplier = 1.0
        if self.enable_dda and game_id in self.difficulty_adjusters:
            time_multiplier = self.difficulty_adjusters[game_id].get_time_multiplier()
        
        # Get move from adaptive agent
        adaptive_result = self.adaptive_agents[game_id].get_move(
            state=state,
            time_multiplier=time_multiplier
        )
        
        logger.info(
            f"Game {game_id[:8]}: Adaptive agent move={adaptive_result.move}, "
            f"skill={adaptive_result.skill_level}, "
            f"time_mult={time_multiplier:.2f}, "
            f"rollouts={adaptive_result.num_rollouts}"
        )
        
        # Get move from reference agent (if enabled)
        reference_result = None
        if self.enable_reference_agent:
            reference_result = self.reference_agents[game_id].get_move(
                state=state,
                time_multiplier=1.0  # Reference always uses base time
            )
            
            logger.debug(
                f"Game {game_id[:8]}: Reference agent move={reference_result.move}, "
                f"rollouts={reference_result.num_rollouts}"
            )
        
        return MoveResult(
            move=adaptive_result.move,
            adaptive_result=adaptive_result,
            reference_result=reference_result
        )
    
    def _ensure_agents(self, game_id: str, skill_level: Optional[str] = None) -> None:
        """
        Ensure agents exist for a game.
        
        Creates agents if they don't exist yet.
        
        Args:
            game_id: Game identifier
            skill_level: Optional skill level override
        """
        # Create adaptive agent if needed
        if game_id not in self.adaptive_agents:
            skill = skill_level or self.default_skill
            self.adaptive_agents[game_id] = MCTSAgent(skill_level=skill)
            logger.info(f"Created adaptive agent for game {game_id[:8]}: skill={skill}")
        
        # Create reference agent if needed
        if self.enable_reference_agent and game_id not in self.reference_agents:
            self.reference_agents[game_id] = MCTSAgent(skill_level=SkillLevel.EXPERT)
            logger.info(f"Created reference agent for game {game_id[:8]}: skill=expert")
        
        # Create DDA if needed
        if self.enable_dda and game_id not in self.difficulty_adjusters:
            self.difficulty_adjusters[game_id] = DifficultyAdjuster()
            logger.info(f"Created DDA for game {game_id[:8]}")
    
    def reset_game(self, game_id: str) -> None:
        """
        Reset agents for a game.
        
        Args:
            game_id: Game identifier
        """
        if game_id in self.adaptive_agents:
            self.adaptive_agents[game_id].reset()
        
        if game_id in self.reference_agents:
            self.reference_agents[game_id].reset()
        
        if game_id in self.difficulty_adjusters:
            self.difficulty_adjusters[game_id].reset()
        
        logger.info(f"Reset agents for game {game_id[:8]}")
    
    def cleanup_game(self, game_id: str) -> None:
        """
        Remove agents for a finished game to free memory.
        
        Args:
            game_id: Game identifier
        """
        self.adaptive_agents.pop(game_id, None)
        self.reference_agents.pop(game_id, None)
        self.difficulty_adjusters.pop(game_id, None)
        
        logger.info(f"Cleaned up agents for game {game_id[:8]}")
    
    def get_dda_stats(self, game_id: str) -> Optional[dict]:
        """
        Get DDA statistics for a game.
        
        Args:
            game_id: Game identifier
        
        Returns:
            DDA statistics or None if not available
        """
        if game_id in self.difficulty_adjusters:
            return self.difficulty_adjusters[game_id].get_stats()
        return None
    
    def get_active_games(self) -> list:
        """
        Get list of active game IDs.
        
        Returns:
            List of game IDs with active agents
        """
        return list(self.adaptive_agents.keys())
    
    def __repr__(self) -> str:
        return (
            f"AIManager(active_games={len(self.adaptive_agents)}, "
            f"default_skill={self.default_skill}, "
            f"dda={self.enable_dda})"
        )
