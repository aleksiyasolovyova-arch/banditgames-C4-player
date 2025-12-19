"""
RabbitMQ Listener - Consumes events from Connect4 Backend.

Listens to the connect4.events exchange and routes events to the handler.
"""

import logging
import json
import asyncio
from typing import Optional
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
from pika.channel import Channel
from pika.spec import Basic, BasicProperties

from .event_schemas import parse_event
from .event_handler import EventHandler

logger = logging.getLogger(__name__)


class RabbitMQListener:
    """
    RabbitMQ event listener for Connect4 Backend events.
    """

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        exchange: str,
        event_handler: EventHandler,
        queue_name: str = "ai_player_queue"
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.exchange = exchange
        self.event_handler = event_handler
        self.queue_name = queue_name

        self._connection: Optional[AsyncioConnection] = None
        self._channel: Optional[Channel] = None
        self._consumer_tag: Optional[str] = None
        self._closing = False

        logger.info(f"RabbitMQListener initialized: {host}:{port}, exchange={exchange}")

    async def start(self) -> None:
        logger.info("Starting RabbitMQ listener...")

        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )

        self._connection = AsyncioConnection(
            parameters=parameters,
            on_open_callback=self._on_connection_open,
            on_open_error_callback=self._on_connection_open_error,
            on_close_callback=self._on_connection_closed
        )

    def _on_connection_open(self, connection: AsyncioConnection) -> None:
        logger.info("RabbitMQ connection opened")
        self._connection = connection
        self._connection.channel(on_open_callback=self._on_channel_open)

    def _on_connection_open_error(self, connection: AsyncioConnection, error: Exception) -> None:
        logger.error(f"Failed to open RabbitMQ connection: {error}")
        self._reconnect()

    def _on_connection_closed(self, connection: AsyncioConnection, reason: Exception) -> None:
        self._channel = None

        if self._closing:
            logger.info("RabbitMQ connection closed")
        else:
            logger.warning(f"RabbitMQ connection closed unexpectedly: {reason}")
            self._reconnect()

    def _reconnect(self) -> None:
        if not self._closing:
            logger.info("Reconnecting to RabbitMQ in 5 seconds...")
            asyncio.get_event_loop().call_later(5, lambda: asyncio.create_task(self.start()))

    def _on_channel_open(self, channel: Channel) -> None:
        logger.info("RabbitMQ channel opened")
        self._channel = channel
        self._channel.add_on_close_callback(self._on_channel_closed)

        self._channel.exchange_declare(
            exchange=self.exchange,
            exchange_type="topic",
            durable=True,
            callback=self._on_exchange_declared
        )

    def _on_channel_closed(self, channel: Channel, reason: Exception) -> None:
        logger.warning(f"RabbitMQ channel closed: {reason}")
        if not self._closing and self._connection and self._connection.is_open:
            self._connection.close()

    def _on_exchange_declared(self, frame) -> None:
        logger.info(f"Exchange '{self.exchange}' declared")

        self._channel.queue_declare(
            queue=self.queue_name,
            durable=True,
            callback=self._on_queue_declared
        )

    def _on_queue_declared(self, frame) -> None:
        logger.info(f"Queue '{self.queue_name}' declared")

        self._channel.queue_bind(
            queue=self.queue_name,
            exchange=self.exchange,
            routing_key="move.made",
            callback=self._on_queue_bound
        )

    def _on_queue_bound(self, frame) -> None:
        logger.info("Queue bound to exchange with routing key: move.made")

        if self._consumer_tag is None:
            self._consumer_tag = self._channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self._on_message
            )
            logger.info(f"Started consuming from queue: {self.queue_name}")

    # ✅ FIXED SIGNATURE (THIS WAS THE BUG)
    def _on_message(
        self,
        channel: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes
    ) -> None:
        try:
            data = json.loads(body.decode("utf-8"))
            event_type = data.get("eventType")

            logger.debug(
                f"Received event: {event_type}, game={data.get('gameId', 'N/A')[:8]}"
            )

            event = parse_event(event_type, data)

            asyncio.create_task(self._handle_event(event, event_type))

            channel.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    async def _handle_event(self, event: any, event_type: str) -> None:
        try:
            if event_type == "move.made":
                await self.event_handler.handle_move_event(event)
            else:
                logger.warning(f"Unknown event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}", exc_info=True)

    def stop(self) -> None:
        logger.info("Stopping RabbitMQ listener...")
        self._closing = True

        if self._channel and self._channel.is_open:
            self._channel.close()

        if self._connection and self._connection.is_open:
            self._connection.close()

        logger.info("RabbitMQ listener stopped")

