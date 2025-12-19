"""
HTTP client for Gameplay Logging Service.

Handles all communication with the Gameplay Logging Service:
- Logging individual moves (live games)
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime

from .schemas import MoveLogRequest

logger = logging.getLogger(__name__)


class GameplayLoggingClient:
    """
    HTTP client for Gameplay Logging Service.
    
    Provides methods for:
    - Logging individual moves (real-time for live games)
    
    Usage:
        client = GameplayLoggingClient(base_url="http://localhost:8001")
        
        # Log a single move (live game)
        await client.log_move(
            game_id="abc123",
            move_index=5,
            player="player1",
            player_type="ai",
            action_taken=3,
            board_before=[[...]],
            board_after=[[...]],
            mcts_stats={...}
        )
        
        # Bulk upload games (self-play)
        await client.bulk_upload_games(
            session_id="session-123",
            games=[game1, game2, ...]
        )
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 60.0
    ):
        """
        Initialize logging client.
        
        Args:
            base_url: Base URL of Gameplay Logging Service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        logger.info(f"GameplayLoggingClient initialized: {self.base_url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def log_move(
        self,
        game_id: str,
        move_index: int,
        player: str,
        player_type: str,
        action_taken: int,
        board_before: List[List[str]],
        board_after: List[List[str]],
        legal_actions: List[int],
        thinking_time_ms: float,
        mcts_stats: Optional[Dict[str, Any]] = None,
        game_type: str = "human_vs_ai",
        is_self_play: bool = False,
        reference_agent_move: Optional[int] = None,
        reference_agent_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a single move to the logging service.
        
        Args:
            game_id: Game identifier
            move_index: Move number (0-based)
            player: Player identifier ("player1" or "player2")
            player_type: Player type ("human" or "ai")
            action_taken: Column played (0-6)
            board_before: Board state before move
            board_after: Board state after move
            legal_actions: Legal columns before move
            thinking_time_ms: Time taken for move (milliseconds)
            mcts_stats: MCTS statistics (if AI move)
            game_type: Type of game ("human_vs_ai" or "ai_vs_ai")
            is_self_play: Whether this is a self-play game
            reference_agent_move: Reference agent's choice (optional)
            reference_agent_stats: Reference agent's stats (optional)
        
        Returns:
            Response from logging service
        
        Raises:
            httpx.HTTPError: If request fails
        """
        url = "/api/moves"
        
        # Build request
        request = MoveLogRequest(
            game_id=game_id,
            game_type=game_type,
            move_index=move_index,
            player=player,
            player_type=player_type,
            action_taken=action_taken,
            board_before=board_before,
            board_after=board_after,
            legal_actions=legal_actions,
            thinking_time_ms=thinking_time_ms,
            timestamp=datetime.utcnow().isoformat(),
            is_self_play=is_self_play,
            reference_agent_move=reference_agent_move,
            reference_agent_stats=reference_agent_stats
        )
        
        # Add MCTS stats if available
        if mcts_stats:
            request.skill_level = mcts_stats.get('skill_level')
            request.mcts_time_limit = mcts_stats.get('time_limit')
            request.mcts_num_rollouts = mcts_stats.get('num_rollouts')
            request.mcts_visit_counts = mcts_stats.get('visit_counts')
            request.mcts_q_values = mcts_stats.get('q_values')
            request.mcts_best_move = mcts_stats.get('best_move')
        
        logger.debug(
            f"Logging move: game={game_id[:8]}, move={move_index}, "
            f"player={player}, col={action_taken}"
        )
        
        try:
            response = await self.client.post(
                url,
                json=request.dict()
            )
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Move logged successfully: game={game_id[:8]}")
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to log move: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )

    async def close(self) -> None:
        """Close the HTTP client"""
        await self.client.aclose()
        logger.debug("GameplayLoggingClient closed")
    
    def __repr__(self) -> str:
        return f"GameplayLoggingClient(base_url={self.base_url})"

