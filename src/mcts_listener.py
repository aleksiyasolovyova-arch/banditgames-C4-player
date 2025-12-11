"""
MCTS Listener Service with comprehensive logging and self-play support.
Consumes RabbitMQ events and makes AI moves with full statistics tracking.
"""

import json
import logging
import os
import sys
import time
import uuid
import requests
import pika
from threading import Thread
from datetime import datetime
from typing import Dict, Any, Optional

from ConnectState import ConnectState
from ai_manager import AIManager, MCTSResult, OracleAgent, OracleResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCTSListener:
    """
    Enhanced MCTS service with comprehensive logging.

    Features:
    - Listens for game events via RabbitMQ
    - Makes AI moves with MCTS
    - Logs comprehensive statistics to backend
    - Supports self-play for dataset generation
    """

    def __init__(
            self,
            rabbitmq_host: str = None,
            rabbitmq_port: int = None,
            rabbitmq_user: str = None,
            rabbitmq_password: str = None,
            api_base_url: str = None
    ):
        # Read from environment variables with sensible defaults
        self.rabbitmq_host = rabbitmq_host or os.getenv('RABBITMQ_HOST', 'rabbitmq')
        self.rabbitmq_port = rabbitmq_port or int(os.getenv('RABBITMQ_PORT', '5672'))
        self.rabbitmq_user = rabbitmq_user or os.getenv('RABBITMQ_USER', 'user')
        self.rabbitmq_password = rabbitmq_password or os.getenv('RABBITMQ_PASSWORD', 'password')
        self.api_base_url = api_base_url or os.getenv('CONNECT4_BACKEND_URL', 'http://connect4_backend:8000')

        self.ai_manager = AIManager()
        self.connection = None
        self.channel = None
        # Oracle agent for computing optimal moves (runs in background)
        self.oracle = OracleAgent(time_limit=3.0, exploration=0.5)
        self.enable_oracle = True  # Can be disabled via env var

        # Log configuration for debugging
        logger.info(f"RabbitMQ config: host={self.rabbitmq_host}, port={self.rabbitmq_port}, user={self.rabbitmq_user}")
        logger.info(f"Backend API URL: {self.api_base_url}")

        self.setup_rabbitmq()

    def setup_rabbitmq(self):
        """Setup RabbitMQ consumer"""
        try:
            credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_password)
            params = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                credentials=credentials,
                heartbeat=30,
                blocked_connection_timeout=300
            )
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()

            # Declare exchanges
            self.channel.exchange_declare(
                exchange='connect4_events',
                exchange_type='topic',
                durable=True
            )
            self.channel.exchange_declare(
                exchange='connect4_ml',
                exchange_type='topic',
                durable=True
            )

            # Create queue for MCTS
            result = self.channel.queue_declare(
                queue='mcts_queue',
                durable=True
            )
            queue_name = result.method.queue

            # Bind to game events
            bindings = [
                ('connect4_events', 'ai.move.needed'),
                ('connect4_events', 'human.move.made'),
                ('connect4_events', 'game.created'),
                ('connect4_events', 'game.ended'),
                ('connect4_events', 'game.ai_vs_ai.created'),
                ('connect4_ml', 'self_play.session.started'),
            ]

            for exchange, routing_key in bindings:
                self.channel.queue_bind(
                    exchange=exchange,
                    queue=queue_name,
                    routing_key=routing_key
                )

            logger.info("MCTS Listener connected to RabbitMQ")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def create_mcts_stats_payload(self, result: MCTSResult) -> Dict[str, Any]:
        """Convert MCTSResult to API payload format"""
        return {
            'skill_level': result.skill_level,
            'base_skill_level': result.base_skill_level,
            'time_limit_seconds': result.time_limit,
            'actual_search_time_seconds': result.actual_time,
            'num_rollouts': result.num_rollouts,
            'nodes_explored': result.nodes_explored,
            'best_move': result.move,
            'move_stats': result.move_stats,
            'time_adjustment_factor': result.time_adjustment,
            'exploration_constant': result.exploration_constant,
            'metadata': {
                'max_depth': result.max_depth,
                'visit_counts': result.visit_counts,
                'q_values': result.q_values,
                'probabilities': result.probabilities
            }
        }

    def calculate_ai_move(
            self,
            game_id: str,
            board: list,
            current_player: int,
            skill_level: Optional[str] = None,
            dda_adjustment: float = 1.0
    ):
        """Calculate and make AI move with logging"""
        try:
            logger.info(f"Calculating move for game {game_id} (DDA: {dda_adjustment:.2f}x)")

            # Initialize agent if skill level specified
            if skill_level:
                self.ai_manager.get_agent(game_id, skill_level)

            # Create state
            state = ConnectState(
                board=board,
                current_player=current_player,
                rows=6,
                cols=7,
                connect=4
            )

            # Use DDA adjustment
            agent = self.ai_manager.get_agent(game_id)
            move, result = agent.get_move(state, time_adjustment=dda_adjustment)

            logger.info(
                f"Move {move} calculated: {result.num_rollouts} rollouts in {result.actual_time:.3f}s")

            # Make move via API with MCTS stats
            self.make_ai_move_via_api(game_id, move, result)

        except Exception as e:
            logger.error(f"Error calculating move: {e}", exc_info=True)

    def run_oracle_analysis(
            self,
            game_id: str,
            board: list,
            current_player: int,
            move_index: int,
            actual_move: int = None
    ):
        """
        Run oracle analysis in background and log to database.

        Args:
            game_id: Game identifier
            board: Current board state
            current_player: Player to move (1 or 2)
            move_index: Current move number
            actual_move: The move that was actually played (if known)
        """
        try:
            from ConnectState import ConnectState

            # Create state for analysis
            state = ConnectState(
                board=board,
                current_player=current_player,
                rows=6,
                cols=7,
                connect=4
            )

            # Run oracle analysis
            logger.info(f" Oracle analyzing game {game_id} move {move_index}...")
            result = self.oracle.analyze(state)

            logger.info(f"Oracle: best={result.best_move}, "
                        f"ranking={result.move_ranking}, "
                        f"rollouts={result.num_rollouts}")

            # Log to database via API
            self.log_oracle_result(
                game_id=game_id,
                move_index=move_index,
                state_hash=state.compute_hash(),
                result=result,
                actual_move=actual_move
            )

        except Exception as e:
            logger.error(f"Oracle analysis failed: {e}", exc_info=True)

    def log_oracle_result(
            self,
            game_id: str,
            move_index: int,
            state_hash: str,
            result: OracleResult,
            actual_move: int = None
    ):
        """Log oracle result to database via API"""
        try:
            url = f"{self.api_base_url}/oracle/log"

            payload = {
                "game_id": game_id,
                "move_index": move_index,
                "state_hash": state_hash,
                "best_move": result.best_move,
                "move_ranking": result.move_ranking,
                "visit_counts": result.visit_counts,
                "q_values": result.q_values,
                "probabilities": result.probabilities,
                "num_rollouts": result.num_rollouts,
                "search_time": result.search_time,
                "actual_move": actual_move
            }

            response = requests.post(url, json=payload, timeout=5)

            if response.status_code == 200:
                logger.debug(f"Oracle result logged for game {game_id} move {move_index}")
            else:
                logger.warning(f"Failed to log oracle result: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to log oracle result: {e}")

    def make_ai_move_via_api(self, game_id: str, column: int, result: MCTSResult):
        """Call backend API to make AI move with statistics"""
        try:
            url = f"{self.api_base_url}/games/{game_id}/moves"

            mcts_stats = self.create_mcts_stats_payload(result)

            # ✅ FIX: Don't hardcode player - let backend auto-detect current player
            payload = {
                "column": column,
                "thinking_time_ms": int(result.actual_time * 1000),
                "mcts_stats": mcts_stats,
            }

            logger.info(f" Sending move: game={game_id}, column={column}")

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                state = response.json()
                logger.info(f" Move accepted: turn={state.get('turn_index')}, status={state.get('status')}")

                # Log if game ended
                if state.get('status') in ['win', 'draw']:
                    logger.info(f"Game {game_id} ended: {state.get('status')}, winner={state.get('winner')}")
            else:
                logger.error(f"Backend rejected move: {response.status_code}")
                logger.error(f"   Response: {response.text}")

        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Backend timeout for game {game_id}")
        except Exception as e:
            logger.error(f" Failed to submit move: {e}", exc_info=True)

    def run_ai_vs_ai_game(
            self,
            game_id: str,
            skill_p1: str,
            skill_p2: str,
            noise_level: float = 0.1,
            temperature: float = 0.5
    ):
        """Run complete AI vs AI game with full logging"""
        logger.info(f"Starting AI vs AI: {game_id} ({skill_p1} vs {skill_p2})")

        # Create agents
        agent1, agent2 = self.ai_manager.create_self_play_agents(
            skill_p1, skill_p2, noise_level, temperature
        )

        moves = 0
        current_player = 1
        game_active = True

        while game_active and moves < 42:
            try:
                # Get current state
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

                # Get move from appropriate agent
                if current_player == 1:
                    move, result = agent1.get_move(state)
                else:
                    move, result = agent2.get_move(state)
                mcts_stats = self.create_mcts_stats_payload(result)

                payload = {
                    "column": move,
                    "thinking_time_ms": int(result.actual_time * 1000),
                    "mcts_stats": mcts_stats
                }

                response = requests.post(
                    f"{self.api_base_url}/games/{game_id}/moves",
                    json=payload
                )

                if response.status_code != 200:
                    logger.error(f"Move failed: {response.text}")
                    break

                moves += 1
                current_player = 2 if current_player == 1 else 1
                time.sleep(0.05)  # Small delay between moves

            except Exception as e:
                logger.error(f"Error in AI vs AI game: {e}", exc_info=True)
                break

        logger.info(f"AI vs AI game {game_id} completed after {moves} moves")

    def run_self_play_session(
            self,
            session_id: str,
            num_games: int,
            skill_levels: list,
            noise_level: float,
            temperature: float,
            vary_skills: bool
    ):
        """Run a self-play session"""
        logger.info(f"Starting self-play session {session_id}: {num_games} games")

        results = {
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0
        }

        start_time = time.time()

        for i in range(num_games):
            try:
                # Vary skill levels if requested
                if vary_skills and len(skill_levels) >= 2:
                    import random
                    skill1 = random.choice(skill_levels)
                    skill2 = random.choice(skill_levels)
                else:
                    skill1 = skill_levels[0] if skill_levels else 'medium'
                    skill2 = skill_levels[1] if len(skill_levels) > 1 else skill1

                # Create game via API
                config = {
                    "player1_type": "cpu",
                    "player2_type": "cpu",
                    "player1_skill_level": skill1,
                    "player2_skill_level": skill2,
                    "noise_level": noise_level,
                    "temperature": temperature
                }

                response = requests.post(
                    f"{self.api_base_url}/games",
                    json=config
                )

                if response.status_code != 200:
                    logger.error(f"Failed to create game: {response.text}")
                    continue

                game_data = response.json()
                game_id = game_data['game_id']

                # Run the game
                self.run_ai_vs_ai_game(
                    game_id, skill1, skill2, noise_level, temperature
                )

                # Get final result
                response = requests.get(f"{self.api_base_url}/games/{game_id}")
                if response.status_code == 200:
                    final_state = response.json()
                    winner = final_state.get('winner')

                    if winner == 'player1':
                        results['agent1_wins'] += 1
                    elif winner == 'player2':
                        results['agent2_wins'] += 1
                    else:
                        results['draws'] += 1

                # Log progress every 10 games
                if (i + 1) % 10 == 0:
                    logger.info(f"Self-play progress: {i + 1}/{num_games} games")

            except Exception as e:
                logger.error(f"Error in self-play game {i}: {e}")

        duration = time.time() - start_time

        logger.info(f"Self-play session {session_id} completed:")
        logger.info(f"  Games: {num_games}")
        logger.info(f"  Agent1 wins: {results['agent1_wins']}")
        logger.info(f"  Agent2 wins: {results['agent2_wins']}")
        logger.info(f"  Draws: {results['draws']}")
        logger.info(f"  Duration: {duration:.1f}s")

        # Publish completion event
        self.publish_session_completed(session_id, num_games, results, duration)

    def publish_session_completed(
            self,
            session_id: str,
            num_games: int,
            results: Dict,
            duration: float
    ):
        """Publish self-play session completed event"""
        try:
            event = {
                'event_id': str(uuid.uuid4()),
                'event_type': 'self_play.session.ended',
                'timestamp': datetime.utcnow().isoformat(),
                'session_id': session_id,
                'total_games': num_games,
                'agent1_wins': results['agent1_wins'],
                'agent2_wins': results['agent2_wins'],
                'draws': results['draws'],
                'duration_seconds': duration
            }

            self.channel.basic_publish(
                exchange='connect4_ml',
                routing_key='self_play.session.ended',
                body=json.dumps(event),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type='application/json'
                )
            )

        except Exception as e:
            logger.error(f"Failed to publish session completed: {e}")

    def handle_message(self, ch, method, properties, body):
        """Handle incoming RabbitMQ message"""
        try:
            event = json.loads(body)
            event_type = event.get('event_type')

            logger.debug(f"Received event: {event_type}")

            if event_type == 'human.move.made':
                game_id = event['game_id']
                self.ai_manager.record_human_move_time(game_id)

            elif event_type == 'ai.move.needed':
                game_id = event['game_id']
                board = event['board']
                current_player = event['current_player']
                skill_level = event.get('skill_level')
                dda_adjustment = event.get('dda_adjustment', 1.0)
                move_index = event.get('move_index', 0)

                Thread(
                    target=self.calculate_ai_move,
                    args=(game_id, board, current_player, skill_level, dda_adjustment),
                    daemon=True
                ).start()

                if self.enable_oracle:
                    Thread(
                        target=self.run_oracle_analysis,
                        args=(game_id, board, current_player, move_index),
                        daemon=True
                    ).start()

            elif event_type == 'game.created':
                game_id = event['game_id']
                config = event.get('config', {})

                if config.get('player2_type') == 'cpu':
                    skill = config.get('player2_skill_level', 'medium')
                    noise = config.get('noise_level', 0.0)
                    temp = config.get('temperature', 0.0)
                    self.ai_manager.get_agent(game_id, skill, noise, temp)
                    logger.info(f"Initialized AI agent: {game_id} ({skill})")

            elif event_type == 'ai_vs_ai.created':
                game_id = event['game_id']
                skill_p1 = event.get('skill_level_p1', 'medium')
                skill_p2 = event.get('skill_level_p2', 'medium')

                Thread(
                    target=self.run_ai_vs_ai_game,
                    args=(game_id, skill_p1, skill_p2),
                    daemon=True
                ).start()

            elif event_type == 'self_play.session.started':
                session_id = event['session_id']
                config = event.get('config', {})

                Thread(
                    target=self.run_self_play_session,
                    args=(
                        session_id,
                        config.get('num_games', 100),
                        config.get('skill_levels', ['medium']),
                        config.get('noise_level', 0.1),
                        config.get('temperature', 0.5),
                        config.get('vary_skills', True)
                    ),
                    daemon=True
                ).start()

            elif event_type == 'game.ended':
                game_id = event['game_id']
                winner = event.get('winner')

                if winner:
                    player_won = (winner == 'player1')
                    self.ai_manager.update_performance(game_id, player_won)

                self.ai_manager.cleanup(game_id)
                logger.debug(f"Cleaned up game: {game_id}")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def start(self):
        """Start consuming messages"""
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue='mcts_queue',
            on_message_callback=self.handle_message
        )

        logger.info("MCTS Listener started")
        logger.info("Listening for: ai.move.needed, game.created, game.ended, self_play.session.started")

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Shutting down")
            self.channel.stop_consuming()
            self.connection.close()


if __name__ == "__main__":
    max_retries = 10
    retry_delay = 5

    for i in range(max_retries):
        try:
            listener = MCTSListener()
            listener.start()
            break
        except pika.exceptions.AMQPConnectionError:
            if i < max_retries - 1:
                logger.warning(f"RabbitMQ not ready, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error("Failed to connect to RabbitMQ")
                raise