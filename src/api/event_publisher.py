"""
RabbitMQ Event Publisher for AI Player Service.

Publishes events for gameplay logging:
- move.logged (individual AI moves with complete MCTS statistics)
"""

import json
import logging
import uuid
from datetime import datetime, UTC
from typing import Dict, Any, List

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

logger = logging.getLogger(__name__)


class AIPlayerEventPublisher:
    """
    RabbitMQ event publisher for AI Player Service.

    Publishes to "ai_player.events" exchange for gameplay logging.
    """

    EXCHANGE = "ai_player.events"

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str
    ):
        """Initialize event publisher"""
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self._connection: pika.BlockingConnection = None
        self._channel: pika.channel.Channel = None
        self._setup_connection()

    def _setup_connection(self) -> None:
        """Setup RabbitMQ connection and exchange"""
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )

            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()

            # Declare exchange
            self._channel.exchange_declare(
                exchange=self.EXCHANGE,
                exchange_type="topic",
                durable=True
            )

            logger.info(f"Connected to RabbitMQ for event publishing: {self.host}:{self.port}")

        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def _ensure_connection(self) -> None:
        """Ensure connection is alive, reconnect if needed"""
        if not self._connection or self._connection.is_closed:
            logger.warning("RabbitMQ connection lost, reconnecting...")
            self._setup_connection()

    def _publish(self, routing_key: str, event: Dict[str, Any]) -> None:
        """Publish event to RabbitMQ"""
        try:
            self._ensure_connection()
            message_body = json.dumps(event, default=str)

            self._channel.basic_publish(
                exchange=self.EXCHANGE,
                routing_key=routing_key,
                body=message_body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent
                    content_type="application/json",
                    timestamp=int(datetime.now(UTC).timestamp())
                )
            )

            logger.debug(f"Published event: {routing_key}")

        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.error(f"Failed to publish event {routing_key}: {e}")
            raise

    def publish_move_logged(
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
        mcts_stats: Dict[str, Any],
        game_type: str = "human_vs_ai",
        is_self_play: bool = False,
        reference_agent_move: int = None,
        reference_agent_stats: Dict[str, Any] = None
    ) -> None:
        """
        Publish move.logged event with complete MCTS statistics.

        This event contains:
        - Game state from backend (board before/after)
        - AI's chosen move
        - Complete MCTS statistics (visit counts, Q-values, rollouts)
        - Reference agent's statistics (for ML comparison)
        """
        event = {
            "eventId": str(uuid.uuid4()),
            "eventType": "move.logged",
            "timestamp": datetime.now(UTC).isoformat(),

            # Game identification
            "gameId": game_id,

            # Move information
            "moveIndex": move_index,
            "player": player,
            "playerType": player_type,
            "actionTaken": action_taken,

            # Board state (from backend)
            "boardBefore": board_before,
            "boardAfter": board_after,
            "legalActions": legal_actions,

            # AI statistics (adaptive agent)
            "thinkingTimeMs": thinking_time_ms,
            "mctsStats": mcts_stats,

            # Reference agent (for comparison/ML)
            "referenceAgentMove": reference_agent_move,
            "referenceAgentStats": reference_agent_stats
        }

        self._publish("move.logged", event)

        # Log details
        logger.info(
            f"Published move.logged: game={game_id[:8]}, "
            f"move={action_taken}, rollouts={mcts_stats.get('num_rollouts', 0)}, "
            f"ref_move={reference_agent_move if reference_agent_move else 'N/A'}"
        )

    def close(self) -> None:
        """Close RabbitMQ connection"""
        if self._connection and not self._connection.is_closed:
            self._connection.close()
            logger.info("RabbitMQ event publisher connection closed")
