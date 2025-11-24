# MCTS service with in-game difficulty adaptation

import json
import logging
import sys
import time
import requests
import pika
from threading import Thread
from datetime import datetime

sys.path.append('/app')
from ConnectState import ConnectState
from ai_manager import AIManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCTSListener:
    """Enhanced MCTS service with in-game adaptation"""

    def __init__(self, rabbitmq_host='rabbitmq', api_base_url='http://connect4-api:8000'):
        self.rabbitmq_host = rabbitmq_host
        self.api_base_url = api_base_url
        self.ai_manager = AIManager()
        self.connection = None
        self.channel = None
        self.setup_rabbitmq()

    def setup_rabbitmq(self):
        """Setup RabbitMQ consumer"""
        try:
            credentials = pika.PlainCredentials('user', 'password')
            params = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=5672,
                credentials=credentials,
                heartbeat=30,
                blocked_connection_timeout=300
            )            
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()

            # Declare exchange
            self.channel.exchange_declare(
                exchange='connect4_events',
                exchange_type='topic',
                durable=True
            )

            # Create queue for MCTS
            result = self.channel.queue_declare(
                queue='mcts_queue',
                durable=True
            )
            queue_name = result.method.queue

            # Bind to events we care about
            self.channel.queue_bind(
                exchange='connect4_events',
                queue=queue_name,
                routing_key='ai.move.needed'
            )
            self.channel.queue_bind(
                exchange='connect4_events',
                queue=queue_name,
                routing_key='human.move.made'  # NEW: Track human moves
            )
            self.channel.queue_bind(
                exchange='connect4_events',
                queue=queue_name,
                routing_key='game.created'
            )
            self.channel.queue_bind(
                exchange='connect4_events',
                queue=queue_name,
                routing_key='game.ended'
            )
            self.channel.queue_bind(
                exchange='connect4_events',
                queue=queue_name,
                routing_key='ai_vs_ai.created'
            )

            logger.info("MCTS: Connected to RabbitMQ")

        except Exception as e:
            logger.error(f"MCTS: Failed to connect to RabbitMQ: {e}")
            raise

    def calculate_ai_move(self, game_id: str, board: list, current_player: int):
        """Calculate AI move with adaptive difficulty"""
        try:
            logger.info(f"MCTS: Calculating adaptive move for game {game_id}")

            # Create ConnectState from board
            state = ConnectState(
                board=board,
                current_player=current_player,
                rows=6,
                cols=7,
                connect=4
            )

            # Get AI move with dynamic adjustment
            move, stats = self.ai_manager.get_ai_move(game_id, state)

            logger.info(f"MCTS: Calculated move {move} for game {game_id}")
            logger.info(f"MCTS: Adaptive stats: {stats}")

            # Call Connect4 Backend API to make the move
            self.make_ai_move_via_api(game_id, move)

        except Exception as e:
            logger.error(f"MCTS: Error calculating move: {e}")

    def make_ai_move_via_api(self, game_id: str, column: int):
        """Call Connect4 Backend API to make AI move"""
        try:
            # Call the regular move endpoint as player2
            url = f"{self.api_base_url}/games/{game_id}/moves"
            payload = {
                "column": column,
                "player": "player2"
            }

            response = requests.post(url, json=payload)

            if response.status_code == 200:
                logger.info(f"MCTS: Successfully made move via API for game {game_id}")
            else:
                logger.error(f"MCTS: API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"MCTS: Failed to call API: {e}")

    def run_ai_vs_ai_game(self, game_id: str, skill_p1: str, skill_p2: str):
        """Run complete AI vs AI game"""
        logger.info(f"MCTS: Starting AI vs AI game {game_id}")

        # Create two agents
        agent1 = self.ai_manager.get_agent(f"{game_id}_p1", skill_p1)
        agent2 = self.ai_manager.get_agent(f"{game_id}_p2", skill_p2)

        moves = 0
        current_player = 1

        while moves < 42:
            # Get current game state from API
            response = requests.get(f"{self.api_base_url}/games/{game_id}")
            if response.status_code != 200:
                logger.error(f"Failed to get game state: {response.status_code}")
                break

            game_state = response.json()

            if game_state['status'] != 'in_progress':
                logger.info(f"Game {game_id} ended: {game_state.get('winner', 'draw')}")
                break

            # Create state for MCTS
            state = ConnectState(
                board=game_state['board'],
                current_player=current_player,
                rows=6,
                cols=7,
                connect=4
            )

            # Get move from appropriate agent - no in-game adaptation for AI vs AI
            if current_player == 1:
                move = agent1.get_move(state)
                player = "player1"
            else:
                move = agent2.get_move(state)
                player = "player2"

            # Make move via API
            response = requests.post(
                f"{self.api_base_url}/games/{game_id}/moves",
                json={"column": move, "player": player}
            )

            if response.status_code == 200:
                logger.info(f"AI vs AI: {player} moved to column {move}")
                current_player = 2 if current_player == 1 else 1
                moves += 1
            else:
                logger.error(f"Move failed: {response.text}")
                break

            # Small delay to not overwhelm the system
            time.sleep(0.1)

        # Cleanup
        self.ai_manager.cleanup(f"{game_id}_p1")
        self.ai_manager.cleanup(f"{game_id}_p2")
        logger.info(f"MCTS: AI vs AI game {game_id} completed")

    def handle_message(self, ch, method, properties, body):
        """Handle incoming RabbitMQ message"""
        try:
            event = json.loads(body)
            event_type = event.get('event_type')

            logger.info(f"MCTS: Received event: {event_type}")

            if event_type == 'human.move.made':
                # NEW: Track human move timing
                game_id = event['game_id']
                self.ai_manager.record_human_move_time(game_id)
                logger.info(f"MCTS: Recorded human move timing for game {game_id}")

                # Don't calculate AI move here, wait for ai.move.needed event

            elif event_type == 'ai.move.needed':
                # AI needs to make a move (with adaptation)
                game_id = event['game_id']
                board = event['board']
                current_player = event['current_player']

                # Calculate and make move in a separate thread to not block
                Thread(
                    target=self.calculate_ai_move,
                    args=(game_id, board, current_player),
                    daemon=True
                ).start()

            elif event_type == 'game.created':
                # New game created, initialize AI if needed
                game_id = event['game_id']
                config = event.get('config', {})

                if config.get('player2_type') == 'cpu':
                    # Initialize AI agent for this game
                    skill_level = config.get('ai_skill_level', 'medium')
                    self.ai_manager.get_agent(game_id, skill_level)
                    logger.info(f"MCTS: Initialized adaptive AI agent for game {game_id} with base skill {skill_level}")

            elif event_type == 'ai_vs_ai.created':
                # AI vs AI game
                game_id = event['game_id']
                skill_p1 = event.get('skill_level_p1', 'medium')
                skill_p2 = event.get('skill_level_p2', 'medium')

                # Run AI vs AI game in separate thread
                Thread(
                    target=self.run_ai_vs_ai_game,
                    args=(game_id, skill_p1, skill_p2),
                    daemon=True
                ).start()

            elif event_type == 'game.ended':
                # Game ended, update performance
                game_id = event['game_id']
                winner = event.get('winner')

                # Update performance metrics for future games
                if winner:
                    player_won = (winner == 'player1')
                    self.ai_manager.update_performance(game_id, player_won)

                    # Log final statistics
                    if game_id in self.ai_manager.performance:
                        perf = self.ai_manager.performance[game_id]
                        logger.info(f"MCTS: Game {game_id} ended - Win rate: {perf.get_win_rate():.2f}")
                        logger.info(
                            f"MCTS: Average response time: {perf.move_history.get_average_response_time():.1f}s")

                # Cleanup
                self.ai_manager.cleanup(game_id)
                logger.info(f"MCTS: Cleaned up game {game_id}")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error(f"MCTS: Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def start(self):
        """Start consuming messages"""
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue='mcts_queue',
            on_message_callback=self.handle_message
        )

        logger.info("MCTS: Started with enhanced adaptive difficulty")
        logger.info("MCTS: Tracking player response times for in-game adaptation")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("MCTS: Shutting down")
            self.channel.stop_consuming()
            self.connection.close()


if __name__ == "__main__":
    # Wait for RabbitMQ to be ready
    import time
    max_retries = 10
    retry_delay = 5
    for i in range(max_retries):
        try:
            listener = MCTSListener()
            listener.start()
            break
        except pika.exceptions.AMQPConnectionError:
            if i < max_retries - 1:
                print(f"RabbitMQ not ready, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to RabbitMQ after multiple retries")
                raise
