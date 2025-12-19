"""
HTTP client for Connect4 Backend API.

Handles all communication with the Connect4 Backend:
- Making moves
- Getting game state
"""

import logging
from typing import  Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .schemas import MakeMoveRequest

logger = logging.getLogger(__name__)


class Connect4BackendClient:
    """
    HTTP client for Connect4 Backend.
    
    Provides methods for:
    - Making moves
    - Getting game state
    - Error handling with retries
    
    Usage:
        client = Connect4BackendClient(base_url="http://localhost:8000")
        
        # Make a move
        await client.make_move(
            game_id="abc123",
            player_id="player-uuid",
            column=3
        )
        
        # Get game state
        state = await client.get_game_state(game_id="abc123")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0
    ):
        """
        Initialize backend client.
        
        Args:
            base_url: Base URL of Connect4 Backend
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        logger.info(f"Connect4BackendClient initialized: {self.base_url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def make_move(
        self,
        game_id: str,
        player_id: str,
        column: int
    ) -> Dict[str, Any]:
        """
        Make a move in the game.
        
        Args:
            game_id: Game identifier
            player_id: Player making the move
            column: Column to place piece (0-6)
        
        Returns:
            Updated game state
        
        Raises:
            httpx.HTTPError: If request fails
        """
        url = f"/games/{game_id}/moves"
        
        request = MakeMoveRequest(
            playerId=player_id,
            column=column
        )
        
        logger.debug(f"Making move: game={game_id[:8]}, player={player_id[:8]}, col={column}")
        
        try:
            response = await self.client.post(
                url,
                json=request.dict(by_alias=True)
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Move successful: game={game_id[:8]}, col={column}")
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to make move: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_game_state(self, game_id: str) -> Dict[str, Any]:
        """
        Get current game state.
        
        Args:
            game_id: Game identifier
        
        Returns:
            Game state dictionary
        
        Raises:
            httpx.HTTPError: If request fails
        """
        url = f"/games/{game_id}"
        
        logger.debug(f"Getting game state: game={game_id[:8]}")
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Got game state: game={game_id[:8]}, phase={data.get('phase')}")
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to get game state: {e}")
            raise
    
    async def close(self) -> None:
        """Close the HTTP client"""
        await self.client.aclose()
        logger.debug("Connect4BackendClient closed")
    
    def __repr__(self) -> str:
        return f"Connect4BackendClient(base_url={self.base_url})"

