"""
Event Handler - Business logic for processing events from Connect4 Backend.

This is the core logic that:
1. Receives move.made events with full game state from backend
2. Determines if it's the AI's turn
3. Gets AI move from AI Manager (adaptive + reference agents)
4. Makes move via Backend API
5. Publishes move.logged event with complete MCTS statistics

SUPPORTS BOTH:
- Primary AI player (for human vs AI games) - uses hardcoded AI_PLAYER_ID
- Temporary AI players (for AI vs AI self-play) - registered dynamically

REFERENCE AGENT OPTIMIZATION:
- Reference agent ONLY runs for human vs AI games (for ML comparison)
- Reference agent DISABLED for AI vs AI games ( performance boost)
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
    2. Determine if AI should respond (supports multiple AI players)
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
        self.primary_ai_player_id = ai_player_id  # Main AI for human vs AI
        self.ai_manager = ai_manager
        self.backend_client = backend_client
        self.event_publisher = event_publisher

        # Track which games the AI is playing in
        self.ai_games: Dict[str, Dict[str, Any]] = {}

        # NEW: Track temporary AI players (for self-play)
        self.managed_players: Dict[str, str] = {}  # player_id -> skill_level

        logger.info(f"EventHandler initialized: primary_ai_player_id={ai_player_id[:8]}")

    def register_temporary_player(self, player_id: str, skill_level: str) -> None:
        """
        Register a temporary AI player for self-play.

        Args:
            player_id: Player UUID
            skill_level: Skill level for this player ('easy', 'medium', 'hard', 'expert')
        """
        self.managed_players[player_id] = skill_level
        logger.info(f"Registered temporary AI player {player_id[:8]} with skill={skill_level}")

    def is_managed_player(self, player_id: str) -> bool:
        """Check if a player ID is managed by this service"""
        return (
            player_id == self.primary_ai_player_id or
            player_id in self.managed_players
        )

    def get_player_skill(self, player_id: str) -> str:
        """
        Get skill level for a player.

        Returns:
            Skill level string, or None for primary AI (uses default from config)
        """
        if player_id == self.primary_ai_player_id:
            return None  # Use default skill from config
        return self.managed_players.get(player_id)

    async def handle_move_event(self, event: MoveEvent) -> None:
        """
        Handle move.made event - Main handler for AI response.
        SUPPORTS BOTH PRIMARY AI (human vs AI) AND TEMPORARY AIs (AI vs AI)

        Flow:
        1. Backend sends move.made with complete game state
        2. AI checks if any managed player should respond
        3. AI calculates move (adaptive + reference agents)
        4. AI executes move via backend API
        5. AI publishes move.logged event with statistics
        """
        game_id = event.gameId

        # FIXED: Check if it's a managed player's turn (using nextPlayerId)
        if not self.is_managed_player(event.nextPlayerId):
            # Not our turn - either opponent's turn or not our game
            return

        # It's our turn!
        our_player_id = event.nextPlayerId
        player_one_id = event.postState.playerOneId
        player_two_id = event.postState.playerTwoId

        # Determine opponent
        opponent_id = player_two_id if our_player_id == player_one_id else player_one_id

        # Register game if first time seeing it
        if game_id not in self.ai_games:
            is_ai_vs_ai = self.is_managed_player(opponent_id)

            self.ai_games[game_id] = {
                'our_player_id': our_player_id,
                'is_player_one': player_one_id == our_player_id,
                'opponent_id': opponent_id,
                'is_ai_vs_ai': is_ai_vs_ai,
                'skill_level': self.get_player_skill(our_player_id)
            }

            logger.info(
                f"Registered game {game_id[:8]}: "
                f"our_player={our_player_id[:8]}, "
                f"opponent={opponent_id[:8]}, "
                f"type={'AI_vs_AI' if is_ai_vs_ai else 'human_vs_AI'}, "
                f"skill={self.ai_games[game_id]['skill_level']}, "
                f"reference_agent={'DISABLED' if is_ai_vs_ai else 'ENABLED'}"
            )

        # Check if game is in progress
        if event.postState.phase != "IN_PROGRESS":
            # Game finished - cleanup
            if game_id in self.ai_games:
                logger.info(f"Game {game_id[:8]} finished, cleaning up")
                del self.ai_games[game_id]
                self.ai_manager.cleanup_game(game_id)
            return

        game_info = self.ai_games[game_id]
        skill_level = game_info['skill_level']

        logger.info(
            f"Game {game_id[:8]}: AI's turn "
            f"(player={our_player_id[:8]}, skill={skill_level})"
        )

        # Get player thinking time for DDA (only for human vs AI)
        player_thinking_time = None
        if not game_info['is_ai_vs_ai']:
            # In human vs AI, get the human's thinking time from the last move
            if event.move.playerId != our_player_id:
                player_thinking_time = event.move.thinkingTimeMs / 1000.0

        # Build game state from backend event
        state_before_ai = self._build_state_from_event(event)

        # Reference agent ONLY for human vs AI games (faster for AI vs AI!)
        use_reference_agent = not game_info['is_ai_vs_ai']

        # Get AI move with configured skill level
        try:
            move_result = self.ai_manager.get_move(
                game_id=game_id,
                state=state_before_ai,
                player_thinking_time=player_thinking_time,  # None for AI vs AI
                skill_level=skill_level,
                use_reference_agent=use_reference_agent # Disable for AI vs AI
            )

            logger.info(
                f"Game {game_id[:8]}: AI chose column {move_result.move}, "
                f"skill={skill_level}, rollouts={move_result.adaptive_result.num_rollouts}"
            )

            if move_result.reference_result:
                logger.debug(
                    f"Game {game_id[:8]}: Reference agent chose column {move_result.reference_result.move}, "
                    f"rollouts={move_result.reference_result.num_rollouts}"
                )
            elif game_info['is_ai_vs_ai']:
                logger.debug(
                    f"Game {game_id[:8]}: Reference agent skipped (AI vs AI game)"
                )

        except Exception as e:
            logger.error(f"Failed to get AI move for game {game_id[:8]}: {e}", exc_info=True)
            return

        # Execute move via backend API
        try:
            await self.backend_client.make_move(
                game_id=game_id,
                player_id=our_player_id,
                column=move_result.move
            )

            logger.info(f"Game {game_id[:8]}: Move executed successfully")

        except Exception as e:
            logger.error(f"Failed to execute move for game {game_id[:8]}: {e}", exc_info=True)
            return

        # Publish move.logged event for Gameplay Logging Service
        try:
            game_type = "ai_vs_ai" if game_info['is_ai_vs_ai'] else "human_vs_ai"
            player_position = "player1" if game_info['is_player_one'] else "player2"

            # Calculate board state after AI's move
            state_after_ai = state_before_ai.copy()
            state_after_ai.move(move_result.move)

            # Publish with complete MCTS statistics
            self.event_publisher.publish_move_logged(
                game_id=game_id,
                game_type=game_type,  # Correctly identifies AI vs AI
                move_index=event.postState.moveIndex,
                player=player_position,
                player_type="ai",
                action_taken=move_result.move,
                board_before=event.postState.board,  # From backend
                board_after=state_after_ai.board,    # After AI move
                legal_actions=state_before_ai.get_legal_moves(),
                thinking_time_ms=move_result.adaptive_result.actual_time * 1000,
                mcts_stats=move_result.get_mcts_stats(),
                is_self_play=game_info['is_ai_vs_ai'],
                reference_agent_move=move_result.reference_result.move if move_result.reference_result else None,
                reference_agent_stats=move_result.get_reference_stats()
            )

            logger.info(
                f" Published move.logged event for game {game_id[:8]} "
                f"(type={game_type}, ref_agent={'yes' if move_result.reference_result else 'no'})"
            )

        except Exception as e:
            logger.error(f"Failed to publish move.logged: {e}", exc_info=True)

    def _build_state_from_event(self, event: MoveEvent) -> ConnectState:
        """Build ConnectState from backend event data"""
        post_state = event.postState
        current_player = 1 if post_state.currentToken == "X" else 2

        return ConnectState(
            board=post_state.board,
            current_player=current_player
        )