"""
Self-Play Runner for Connect4 Dataset Generation.
Runs games between AI agents and logs comprehensive data for ML training.
"""

import os
import json
import logging
import uuid
import time
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from ConnectState import ConnectState
from src.ai_manager import AIManager, AIAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single game"""
    game_id: str
    winner: Optional[str]
    total_moves: int
    player1_skill: str
    player2_skill: str
    duration_seconds: float
    moves: List[Dict]


@dataclass
class SessionResult:
    """Result of a self-play session"""
    session_id: str
    total_games: int
    player1_wins: int
    player2_wins: int
    draws: int
    total_moves: int
    duration_seconds: float
    games: List[GameResult]


class SelfPlayRunner:
    """
    Run self-play games for dataset generation.

    Features:
    - Configurable skill level combinations
    - Noise and temperature for move variety
    - Comprehensive move logging
    - Parallel game execution
    - Progress tracking
    """

    def __init__(
            self,
            api_base_url: str = None,
            use_api: bool = False
    ):
        self.api_base_url = api_base_url or os.getenv(
            'API_BASE_URL', 'http://localhost:8000'
        )
        self.use_api = use_api
        self.ai_manager = AIManager()

    def create_agents(
            self,
            skill1: str,
            skill2: str,
            noise_level: float,
            temperature: float
    ) -> Tuple[AIAgent, AIAgent]:
        """Create two agents for self-play"""
        agent1 = AIAgent(skill1, noise_level, temperature)
        agent2 = AIAgent(skill2, noise_level, temperature)
        return agent1, agent2

    def play_single_game(
            self,
            skill1: str = 'medium',
            skill2: str = 'medium',
            noise_level: float = 0.1,
            temperature: float = 0.5,
            game_id: Optional[str] = None
    ) -> GameResult:
        """
        Play a single self-play game.

        Args:
            skill1: Skill level for player 1
            skill2: Skill level for player 2
            noise_level: Noise for move variety
            temperature: Softmax temperature
            game_id: Optional game ID

        Returns:
            GameResult with full game data
        """
        game_id = game_id or str(uuid.uuid4())

        # Create agents
        agent1, agent2 = self.create_agents(
            skill1, skill2, noise_level, temperature
        )

        # Initialize game state
        state = ConnectState()
        moves = []
        start_time = time.time()

        while not state.is_terminal():
            current_player = state.to_play

            # Get move from appropriate agent
            if current_player == 1:
                move, result = agent1.get_move(state)
                player_str = 'player1'
            else:
                move, result = agent2.get_move(state)
                player_str = 'player2'

            # Record board before move
            board_before = [row[:] for row in state.board]
            legal_before = state.get_legal_moves()

            # Make the move
            row = state.move(move)

            # Record move with statistics
            move_record = {
                'move_index': len(moves),
                'player': player_str,
                'column': move,
                'row': row,
                'board_before': board_before,
                'board_after': [row[:] for row in state.board],
                'legal_actions': legal_before,
                'mcts_stats': result.to_dict() if result else None,
                'timestamp': datetime.utcnow().isoformat()
            }
            moves.append(move_record)

        # Determine winner
        outcome = state.get_outcome()
        if outcome == 1:
            winner = 'player1'
        elif outcome == 2:
            winner = 'player2'
        else:
            winner = None  # Draw

        duration = time.time() - start_time

        return GameResult(
            game_id=game_id,
            winner=winner,
            total_moves=len(moves),
            player1_skill=skill1,
            player2_skill=skill2,
            duration_seconds=duration,
            moves=moves
        )

    def play_game_via_api(
            self,
            skill1: str,
            skill2: str,
            noise_level: float,
            temperature: float
    ) -> Optional[GameResult]:
        """
        Play game using backend API (with database logging).

        Args:
            skill1: Skill level for player 1
            skill2: Skill level for player 2
            noise_level: Noise level
            temperature: Temperature

        Returns:
            GameResult or None if failed
        """
        try:
            # Create game via API
            response = requests.post(
                f"{self.api_base_url}/games",
                json={
                    "player1_type": "cpu",
                    "player2_type": "cpu",
                    "player1_skill_level": skill1,
                    "player2_skill_level": skill2,
                    "noise_level": noise_level,
                    "temperature": temperature
                }
            )

            if response.status_code != 200:
                logger.error(f"Failed to create game: {response.text}")
                return None

            game_data = response.json()
            game_id = game_data['game_id']

            # Create agents
            agent1, agent2 = self.create_agents(
                skill1, skill2, noise_level, temperature
            )

            moves = []
            start_time = time.time()
            current_player = 1

            while True:
                # Get current state
                response = requests.get(f"{self.api_base_url}/games/{game_id}")
                if response.status_code != 200:
                    break

                state_data = response.json()

                if state_data['status'] != 'in_progress':
                    break

                # Create state for MCTS
                state = ConnectState(
                    board=state_data['board'],
                    current_player=current_player
                )

                # Get move
                if current_player == 1:
                    move, result = agent1.get_move(state)
                    player_str = 'player1'
                else:
                    move, result = agent2.get_move(state)
                    player_str = 'player2'

                # Make move via API
                mcts_payload = result.to_dict() if result else None

                response = requests.post(
                    f"{self.api_base_url}/games/{game_id}/moves",
                    json={
                        "column": move,
                        "player": player_str,
                        "thinking_time_ms": int(result.actual_time * 1000) if result else 0
                    }
                )

                if response.status_code != 200:
                    logger.error(f"Move failed: {response.text}")
                    break

                move_data = response.json()

                moves.append({
                    'move_index': len(moves),
                    'player': player_str,
                    'column': move,
                    'mcts_stats': mcts_payload
                })

                current_player = 2 if current_player == 1 else 1
                time.sleep(0.01)  # Small delay

            # Get final state
            response = requests.get(f"{self.api_base_url}/games/{game_id}")
            final_state = response.json()

            return GameResult(
                game_id=game_id,
                winner=final_state.get('winner'),
                total_moves=len(moves),
                player1_skill=skill1,
                player2_skill=skill2,
                duration_seconds=time.time() - start_time,
                moves=moves
            )

        except Exception as e:
            logger.error(f"Error in API game: {e}")
            return None

    def run_session(
            self,
            num_games: int,
            skill_levels: List[str] = None,
            vary_skills: bool = True,
            noise_level: float = 0.1,
            temperature: float = 0.5,
            num_workers: int = 1,
            progress_callback=None
    ) -> SessionResult:
        """
        Run a self-play session.

        Args:
            num_games: Number of games to play
            skill_levels: Available skill levels
            vary_skills: Vary skill levels across games
            noise_level: Noise for move variety
            temperature: Temperature for move selection
            num_workers: Number of parallel workers
            progress_callback: Callback for progress updates

        Returns:
            SessionResult with all game data
        """
        session_id = str(uuid.uuid4())
        skill_levels = skill_levels or ['easy', 'medium', 'hard', 'expert']

        logger.info(f"Starting session {session_id}: {num_games} games")

        games: List[GameResult] = []
        start_time = time.time()

        # Generate skill combinations for each game
        game_configs = []
        for i in range(num_games):
            if vary_skills:
                skill1 = random.choice(skill_levels)
                skill2 = random.choice(skill_levels)
            else:
                skill1 = skill_levels[0]
                skill2 = skill_levels[1] if len(skill_levels) > 1 else skill1

            game_configs.append((skill1, skill2))

        if num_workers > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for skill1, skill2 in game_configs:
                    future = executor.submit(
                        self.play_single_game,
                        skill1, skill2, noise_level, temperature
                    )
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    games.append(result)

                    if progress_callback:
                        progress_callback(i + 1, num_games)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i + 1}/{num_games} games")
        else:
            # Sequential execution
            for i, (skill1, skill2) in enumerate(game_configs):
                if self.use_api:
                    result = self.play_game_via_api(
                        skill1, skill2, noise_level, temperature
                    )
                else:
                    result = self.play_single_game(
                        skill1, skill2, noise_level, temperature
                    )

                if result:
                    games.append(result)

                if progress_callback:
                    progress_callback(i + 1, num_games)

                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{num_games} games")

        # Compute statistics
        player1_wins = sum(1 for g in games if g.winner == 'player1')
        player2_wins = sum(1 for g in games if g.winner == 'player2')
        draws = sum(1 for g in games if g.winner is None)
        total_moves = sum(g.total_moves for g in games)

        duration = time.time() - start_time

        logger.info(f"Session {session_id} complete:")
        logger.info(f"  Games: {len(games)}")
        logger.info(f"  P1 wins: {player1_wins}, P2 wins: {player2_wins}, Draws: {draws}")
        logger.info(f"  Total moves: {total_moves}")
        logger.info(f"  Duration: {duration:.1f}s")

        return SessionResult(
            session_id=session_id,
            total_games=len(games),
            player1_wins=player1_wins,
            player2_wins=player2_wins,
            draws=draws,
            total_moves=total_moves,
            duration_seconds=duration,
            games=games
        )

    def save_session_to_json(
            self,
            result: SessionResult,
            output_path: str
    ) -> str:
        """
        Save session result to JSON file.

        Args:
            result: Session result
            output_path: Output file path

        Returns:
            Output path
        """
        data = {
            'session_id': result.session_id,
            'total_games': result.total_games,
            'player1_wins': result.player1_wins,
            'player2_wins': result.player2_wins,
            'draws': result.draws,
            'total_moves': result.total_moves,
            'duration_seconds': result.duration_seconds,
            'games': [asdict(g) for g in result.games]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Session saved to {output_path}")
        return output_path


def main():
    """CLI for self-play runner"""
    import argparse

    parser = argparse.ArgumentParser(description='Run Connect4 self-play')
    parser.add_argument('--games', type=int, default=100, help='Number of games')
    parser.add_argument('--skills', nargs='+', default=['medium', 'hard'],
                        help='Skill levels to use')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--output', type=str, default='self_play_results.json',
                        help='Output file')
    parser.add_argument('--use-api', action='store_true', help='Use backend API')

    args = parser.parse_args()

    runner = SelfPlayRunner(use_api=args.use_api)

    result = runner.run_session(
        num_games=args.games,
        skill_levels=args.skills,
        vary_skills=True,
        noise_level=args.noise,
        temperature=args.temperature,
        num_workers=args.workers
    )

    runner.save_session_to_json(result, args.output)


if __name__ == "__main__":
    main()