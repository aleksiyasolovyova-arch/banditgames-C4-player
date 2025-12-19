"""
Event Handler - Business logic for processing events from Connect4 Backend.

This is the core logic that:
1. Receives move.made events with full game state from backend
2. Determines if it's the AI's turn
3. Gets AI move from AI Manager (adaptive + reference agents)
4. Makes move via Backend API
5. Publishes move.logged event with complete MCTS statistics
"""

import logging
from typing import Dict, Any

from ..core.game_state import ConnectState
from ..agents.ai_manager import AIManager
from ..api.backend_client import Connect4BackendClient
from ..api.event_publisher import AIPlayerEventPublisher
from .event_schemas import MoveEvent

logger = logging.getLogger(__name__)


class EventHandler:
    """
    Handles events from Connect4 Backend.

    Main responsibilities:
    1. Process move.made events
    2. Determine if AI should respond
    3. Get AI move and statistics
    4. Execute move via backend
    5. Publish move.logged event with complete statistics
    """

    def __init__(
        self,
        ai_player_id: str,
        ai_manager: AIManager,
        backend_client: Connect4BackendClient,
        event_publisher: AIPlayerEventPublisher
    ):
        """Initialize event handler"""
        self.ai_player_id = ai_player_id
        self.ai_manager = ai_manager
        self.backend_client = backend_client
        self.event_publisher = event_publisher

        # Track which games the AI is playing in
        self.ai_games: Dict[str, Dict[str, Any]] = {}

        logger.info(f"EventHandler initialized: ai_player_id={ai_player_id[:8]}")

    async def handle_move_event(self, event: MoveEvent) -> None:
        """
        Handle move.made event - Main handler for AI response.

        Flow:
        1. Backend sends move.made with complete game state
        2. AI checks if it's their turn
        3. AI calculates move (adaptive + reference agents)
        4. AI executes move via backend API
        5. AI publishes move.logged event with statistics
        """
        game_id = event.gameId

        # Check if AI is in this game
        if game_id not in self.ai_games:
            # Try to register if we missed game.created
            if event.postState.playerOneId == self.ai_player_id:
                self.ai_games[game_id] = {
                    'is_player_one': True,
                    'opponent_id': event.postState.playerTwoId
                }
            elif event.postState.playerTwoId == self.ai_player_id:
                self.ai_games[game_id] = {
                    'is_player_one': False,
                    'opponent_id': event.postState.playerOneId
                }
            else:
                return  # AI not in this game

        # Check if it's AI's turn
        if not self._is_ai_turn(event):
            return

        # Check if game is in progress
        if event.postState.phase != "IN_PROGRESS":
            return

        logger.info(f"Game {game_id[:8]}: AI's turn to move")

        # Get player thinking time for DDA
        player_thinking_time = None
        if event.move.playerId != self.ai_player_id:
            player_thinking_time = event.move.thinkingTimeMs / 1000.0

        # Build game state from backend event
        state_before_ai = self._build_state_from_event(event)

        # Get AI move with complete statistics (adaptive + reference)
        try:
            move_result = self.ai_manager.get_move(
                game_id=game_id,
                state=state_before_ai,
                player_thinking_time=player_thinking_time
            )

            logger.info(
                f"Game {game_id[:8]}: AI chose column {move_result.move}, "
                f"rollouts={move_result.adaptive_result.num_rollouts}"
            )

            if move_result.reference_result:
                logger.debug(
                    f"Game {game_id[:8]}: Reference agent chose column {move_result.reference_result.move}, "
                    f"rollouts={move_result.reference_result.num_rollouts}"
                )

        except Exception as e:
            logger.error(f"Failed to get AI move for game {game_id[:8]}: {e}", exc_info=True)
            return

        # Execute move via backend API
        try:
            await self.backend_client.make_move(
                game_id=game_id,
                player_id=self.ai_player_id,
                column=move_result.move
            )

            logger.info(f"Game {game_id[:8]}: Move executed successfully")

        except Exception as e:
            logger.error(f"Failed to execute move for game {game_id[:8]}: {e}", exc_info=True)
            return

        # Publish move.logged event for Gameplay Logging Service
        try:
            # Check if game still exists (might have been cleaned up if game finished)
            if game_id not in self.ai_games:
                logger.warning(f"Game {game_id[:8]} already cleaned up, skipping move.logged event")
                return

            # Determine player position
            player_position = "player1" if self.ai_games[game_id]['is_player_one'] else "player2"

            # Calculate board state after AI's move
            state_after_ai = state_before_ai.copy()
            state_after_ai.move(move_result.move)

            # Publish with complete MCTS statistics
            self.event_publisher.publish_move_logged(
                game_id=game_id,
                game_type="human_vs_ai",
                move_index=event.postState.moveIndex,
                player=player_position,
                player_type="ai",
                action_taken=move_result.move,
                board_before=event.postState.board,  # From backend
                board_after=state_after_ai.board,    # After AI move
                legal_actions=state_before_ai.get_legal_moves(),
                thinking_time_ms=move_result.adaptive_result.actual_time * 1000,
                mcts_stats=move_result.get_mcts_stats(),
                is_self_play=False,
                reference_agent_move=move_result.reference_result.move if move_result.reference_result else None,
                reference_agent_stats=move_result.get_reference_stats()
            )

            logger.info(f" Published move.logged event for game {game_id[:8]}")

        except Exception as e:
            logger.error(f"Failed to publish move.logged: {e}", exc_info=True)

    def _is_ai_turn(self, event: MoveEvent) -> bool:
        """Check if it's AI's turn"""
        if event.gameId not in self.ai_games:
            return False

        current_player_id = self._get_current_player_id(event.postState)
        return current_player_id == self.ai_player_id

    def _get_current_player_id(self, state_info) -> str:
        """Get current player ID from state"""
        if state_info.currentToken == "X":
            return state_info.playerOneId
        else:
            return state_info.playerTwoId

    def _build_state_from_event(self, event: MoveEvent) -> ConnectState:
        """Build ConnectState from backend event data"""
        post_state = event.postState
        current_player = 1 if post_state.currentToken == "X" else 2

        return ConnectState(
            board=post_state.board,
            current_player=current_player
        )
