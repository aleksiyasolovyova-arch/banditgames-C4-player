"""
Self-Play Data Generation Script

Generates AI vs AI games for training data using existing MCTS skill configurations.

Usage:
    python scripts/generate_self_play_data.py --num-games 100
"""

import asyncio
import argparse
import random
import uuid
import httpx
from typing import List, Dict
import logging
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# If running from src/scripts/, go up two levels
if script_dir.name == 'scripts' and script_dir.parent.name == 'src':
    project_root = script_dir.parent.parent

sys.path.insert(0, str(project_root))

# Import existing skill level configuration
from src.core.config import SkillLevel, MCTSConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfPlayOrchestrator:
    """
    Orchestrates AI vs AI games for dataset generation.

    Uses existing SkillLevel enum and MCTSConfig for consistency.
    """

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        ai_service_url: str = "http://localhost:8002",
        skill_distribution: Dict[str, float] = None,
    ):
        self.backend_url = backend_url.rstrip('/')
        self.ai_service_url = ai_service_url.rstrip('/')

        # Use existing skill levels from your config
        # Default distribution based on skill configs
        self.skill_distribution = skill_distribution or {
            SkillLevel.EASY.value: 0.25,      # 25%
            SkillLevel.MEDIUM.value: 0.35,    # 35%
            SkillLevel.HARD.value: 0.30,      # 30%
            SkillLevel.EXPERT.value: 0.10     # 10%
        }

        # Validate that all skills in distribution are valid
        self._validate_skill_distribution()

        # HTTP clients
        self.backend_client = httpx.AsyncClient(base_url=self.backend_url, timeout=30.0)
        self.ai_client = httpx.AsyncClient(base_url=self.ai_service_url, timeout=30.0)

        logger.info(f"SelfPlayOrchestrator initialized")
        logger.info(f"Backend: {self.backend_url}")
        logger.info(f"AI Service: {self.ai_service_url}")
        logger.info(f"\nSkill distribution:")
        for skill, prob in self.skill_distribution.items():
            config = MCTSConfig.get_config(skill)
            logger.info(
                f"  {skill:8s}: {prob*100:5.1f}% "
                f"(time={config.time_limit}s, rollouts≤{config.rollout_limit})"
            )
        logger.info(f"{'='*70}\n")

    def _validate_skill_distribution(self) -> None:
        """Validate that skill distribution uses valid skill levels"""
        for skill in self.skill_distribution.keys():
            if not MCTSConfig.is_valid_level(skill):
                raise ValueError(
                    f"Invalid skill level in distribution: {skill}. "
                    f"Valid levels: {MCTSConfig.get_all_levels()}"
                )

        # Check that probabilities sum to 1.0 (with small tolerance)
        total = sum(self.skill_distribution.values())
        if not (0.99 <= total <= 1.01):
            logger.warning(
                f"Skill distribution probabilities sum to {total:.3f}, not 1.0. "
                f"Normalizing..."
            )
            # Normalize
            for skill in self.skill_distribution:
                self.skill_distribution[skill] /= total

    def random_skill(self) -> str:
        """
        Pick random skill level based on distribution.

        Returns:
            Skill level string (e.g., 'easy', 'medium', 'hard', 'expert')
        """
        skills = list(self.skill_distribution.keys())
        weights = list(self.skill_distribution.values())
        return random.choices(skills, weights=weights)[0]

    async def register_ai_player(self, player_id: str, skill_level: str) -> None:
        """
        Register an AI player with the AI Player Service.

        Args:
            player_id: Unique player UUID
            skill_level: One of SkillLevel enum values
        """
        # Validate skill level using your existing config
        if not MCTSConfig.is_valid_level(skill_level):
            raise ValueError(f"Invalid skill level: {skill_level}")

        try:
            response = await self.ai_client.post(
                "/api/register-player",
                json={
                    "player_id": player_id,
                    "skill_level": skill_level
                }
            )
            response.raise_for_status()

            # Get config details for logging
            config = MCTSConfig.get_config(skill_level)
            logger.debug(
                f"Registered AI player {player_id[:8]} with skill={skill_level} "
                f"(time={config.time_limit}s, rollouts≤{config.rollout_limit})"
            )

        except httpx.HTTPError as e:
            logger.error(f"Failed to register AI player: {e}")
            raise

    async def create_game(
        self,
        game_id: str,
        player_one_id: str,
        player_one_name: str,
        player_two_id: str,
        player_two_name: str
    ) -> str:
        """
        Create a game via Connect4 Backend

        FIXED: Now includes gameId in the request body
        """
        try:
            response = await self.backend_client.post(
                "/games",
                json={
                    "gameId": game_id,  # ✅ FIXED: Added missing gameId field
                    "playerOne": {
                        "id": player_one_id,
                        "name": player_one_name
                    },
                    "playerTwo": {
                        "id": player_two_id,
                        "name": player_two_name
                    },
                    "rows": 6,
                    "cols": 7
                }
            )
            response.raise_for_status()
            game_data = response.json()
            return game_data['id']

        except httpx.HTTPError as e:
            logger.error(f"Failed to create game: {e}")
            raise

    async def make_first_move(self, game_id: str, player_id: str, column: int) -> None:
        """Make the first move to start the game"""
        try:
            response = await self.backend_client.post(
                f"/games/{game_id}/moves",
                json={
                    "playerId": player_id,
                    "column": column
                }
            )
            response.raise_for_status()
            logger.debug(f"Made first move in game {game_id[:8]}: col={column}")

        except httpx.HTTPError as e:
            logger.error(f"Failed to make first move: {e}")
            raise

    async def wait_for_game_completion(
        self,
        game_id: str,
        game_number: int,
        poll_interval: float = 10.0
    ) -> Dict[str, any]:
        """
        Wait for a game to complete by polling its status.

        Args:
            game_id: Game ID to monitor
            game_number: Game number for logging
            poll_interval: Seconds between status checks

        Returns:
            Final game state dictionary
        """
        start_time = asyncio.get_event_loop().time()
        move_count = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            try:
                # Get game status
                response = await self.backend_client.get(f"/games/{game_id}")
                response.raise_for_status()
                game_state = response.json()

                phase = game_state.get("phase")
                current_moves = len(game_state.get("moves", []))

                # Log progress if moves increased
                if current_moves > move_count:
                    move_count = current_moves
                    logger.debug(
                        f"[Game {game_number:4d}] Progress: {move_count} moves "
                        f"({elapsed:.1f}s elapsed)"
                    )

                # Check if game finished
                if phase == "FINISHED":
                    winner = game_state.get("winner")
                    winner_name = winner.get("name") if winner else "Draw"

                    logger.info(
                        f"[Game {game_number:4d}] Completed! "
                        f"Winner: {winner_name}, "
                        f"Moves: {move_count}, "
                        f"Duration: {elapsed:.1f}s"
                    )

                    return {
                        "status": "completed",
                        "game_id": game_id,
                        "winner": winner_name,
                        "total_moves": move_count,
                        "duration_seconds": elapsed
                    }

                # Wait before next poll
                await asyncio.sleep(poll_interval)

            except httpx.HTTPError as e:
                logger.error(f"[Game {game_number:4d}] Error checking status: {e}")
                await asyncio.sleep(poll_interval)
                continue

    async def generate_single_game(self, game_number: int) -> Dict[str, str]:
        """
        Generate a single AI vs AI game and wait for completion.

        Returns:
            Dictionary with game_id and metadata
        """
        # Generate unique IDs
        game_id = str(uuid.uuid4())  # ✅ FIXED: Generate game_id here
        ai_1_id = str(uuid.uuid4())
        ai_2_id = str(uuid.uuid4())

        # Random skill levels using your existing config
        skill_1 = self.random_skill()
        skill_2 = self.random_skill()

        logger.info(
            f"[Game {game_number:4d}] Creating: {skill_1:6s} vs {skill_2:6s} "
            f"(AI_1={ai_1_id[:8]}, AI_2={ai_2_id[:8]})"
        )

        # Step 1: Register both AI players with AI Player Service
        await self.register_ai_player(ai_1_id, skill_1)
        await self.register_ai_player(ai_2_id, skill_2)

        # Step 2: Create game via Backend (with game_id)
        returned_game_id = await self.create_game(
            game_id=game_id,  # ✅ FIXED: Pass game_id to create_game
            player_one_id=ai_1_id,
            player_one_name=f"AI-{skill_1}",
            player_two_id=ai_2_id,
            player_two_name=f"AI-{skill_2}"
        )

        logger.info(f"[Game {game_number:4d}] Created game {returned_game_id[:8]}")

        # Step 3: Make first move to start the game
        first_column = random.randint(0, 6)
        await self.make_first_move(game_id, ai_1_id, first_column)

        logger.info(
            f"[Game {game_number:4d}] Started with first move: col={first_column}"
        )

        # Step 4: WAIT FOR GAME TO COMPLETE
        completion_result = await self.wait_for_game_completion(
            game_id, game_number
        )

        # Return comprehensive metadata
        return {
            "game_id": game_id,
            "game_number": game_number,
            "player_1_id": ai_1_id,
            "player_1_skill": skill_1,
            "player_2_id": ai_2_id,
            "player_2_skill": skill_2,
            "first_column": first_column,
            **completion_result  # Includes status, winner, moves, duration
        }

    async def generate_games(
        self,
        num_games: int,
    ) -> List[Dict[str, str]]:
        """
        Generate multiple AI vs AI games concurrently (no batching).

        Args:
            num_games: Total number of games to generate

        Returns:
            List of game metadata dictionaries
        """
        logger.info(f"STARTING SELF-PLAY DATA GENERATION")
        logger.info(f"Total games to generate: {num_games}")
        logger.info(f"Running ALL games concurrently (no batching)")

        all_games = []
        skill_counter = {skill: 0 for skill in self.skill_distribution.keys()}
        completed_count = 0
        timeout_count = 0

        # Create ALL games concurrently
        logger.info(f"Starting all {num_games} games concurrently...\n")

        tasks = [
            self.generate_single_game(game_num + 1)
            for game_num in range(num_games)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and collect successful games
        for result in results:
            if isinstance(result, dict):
                all_games.append(result)

                # Count statistics
                if result.get("status") == "completed":
                    completed_count += 1
                    skill_counter[result['player_1_skill']] += 1
                    skill_counter[result['player_2_skill']] += 1
                elif result.get("status") == "timeout":
                    timeout_count += 1
            else:
                logger.error(f"Game generation failed: {result}")

        successful = sum(1 for r in results if isinstance(r, dict))
        logger.info(
            f"\nAll games complete: "
            f"{successful}/{num_games} games processed "
            f"({completed_count} completed, {timeout_count} timeouts)\n"
        )

        # Print final statistics
        self._print_final_stats(all_games, skill_counter, completed_count, timeout_count)

        return all_games

    def _print_final_stats(
        self,
        games: List[Dict[str, str]],
        skill_counter: Dict[str, int],
        completed_count: int,
        timeout_count: int
    ) -> None:
        """Print final statistics about generated games"""
        total_players = sum(skill_counter.values())

        logger.info(f"GENERATION COMPLETE")
        logger.info(f"Total games processed: {len(games)}")
        logger.info(f"Successfully completed: {completed_count}")
        logger.info(f"Timeouts: {timeout_count}")
        logger.info(f"Total AI players: {total_players}")

        if completed_count > 0:
            avg_moves = sum(
                g.get('total_moves', 0)
                for g in games
                if g.get('status') == 'completed'
            ) / completed_count

            avg_duration = sum(
                g.get('duration_seconds', 0)
                for g in games
                if g.get('status') == 'completed'
            ) / completed_count

            logger.info(f"Average moves per game: {avg_moves:.1f}")
            logger.info(f"Average duration: {avg_duration:.1f}s")

        logger.info(f"SKILL LEVEL DISTRIBUTION")

        for skill in sorted(skill_counter.keys()):
            count = skill_counter[skill]
            percentage = (count / total_players) * 100 if total_players > 0 else 0
            config = MCTSConfig.get_config(skill)

            logger.info(
                f"{skill:8s}: {count:4d} players ({percentage:5.1f}%) "
                f"[time={config.time_limit}s, rollouts≤{config.rollout_limit}]"
            )

        logger.info(f"{'='*70}\n")

    async def close(self):
        """Close HTTP clients"""
        await self.backend_client.aclose()
        await self.ai_client.aclose()


async def main():
    parser = argparse.ArgumentParser(
        description="Generate AI vs AI games for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 games with default distribution (5 concurrent)
  python scripts/generate_self_play_data.py --num-games 100
  
  # Generate 1000 games with custom distribution
  python scripts/generate_self_play_data.py --num-games 1000 \\
    --easy 0.3 --medium 0.4 --hard 0.2 --expert 0.1
  
  # Generate 50 games with only 2 concurrent (slower but safer)
  python scripts/generate_self_play_data.py --num-games 50 --batch-size 2
        """
    )

    parser.add_argument(
        '--num-games',
        type=int,
        default=100,
        help='Number of games to generate (default: 100)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of games to run concurrently (default: 5, reduced from 10)'
    )

    parser.add_argument(
        '--max-game-duration',
        type=int,
        default=120,
        help='Maximum seconds per game before timeout (default: 120)'
    )

    parser.add_argument(
        '--backend-url',
        type=str,
        default='http://localhost:8000',
        help='Connect4 Backend URL (default: http://localhost:8000)'
    )

    parser.add_argument(
        '--ai-service-url',
        type=str,
        default='http://localhost:8002',
        help='AI Player Service URL (default: http://localhost:8002)'
    )

    # Skill distribution arguments
    parser.add_argument(
        '--easy',
        type=float,
        help='Probability for easy skill level (0.0-1.0)'
    )

    parser.add_argument(
        '--medium',
        type=float,
        help='Probability for medium skill level (0.0-1.0)'
    )

    parser.add_argument(
        '--hard',
        type=float,
        help='Probability for hard skill level (0.0-1.0)'
    )

    parser.add_argument(
        '--expert',
        type=float,
        help='Probability for expert skill level (0.0-1.0)'
    )

    args = parser.parse_args()

    # Build skill distribution from arguments
    skill_distribution = None
    if any([args.easy, args.medium, args.hard, args.expert]):
        skill_distribution = {}

        if args.easy is not None:
            skill_distribution[SkillLevel.EASY.value] = args.easy
        if args.medium is not None:
            skill_distribution[SkillLevel.MEDIUM.value] = args.medium
        if args.hard is not None:
            skill_distribution[SkillLevel.HARD.value] = args.hard
        if args.expert is not None:
            skill_distribution[SkillLevel.EXPERT.value] = args.expert

        # Fill in missing skills with 0.0
        for skill in SkillLevel:
            if skill.value not in skill_distribution:
                skill_distribution[skill.value] = 0.0

    # Create orchestrator
    orchestrator = SelfPlayOrchestrator(
        backend_url=args.backend_url,
        ai_service_url=args.ai_service_url,
        skill_distribution=skill_distribution,
    )

    try:
        # Generate games
        games = await orchestrator.generate_games(
            num_games=args.num_games,
        )

        # Print summary
        completed = sum(1 for g in games if g.get('status') == 'completed')
        logger.info(f"SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Games requested: {args.num_games}")
        logger.info(f"Games completed: {completed}")
        logger.info(f"Success rate: {(completed/args.num_games)*100:.1f}%")
        logger.info(f"\n All games completed!")
        logger.info(f"📊 All move.logged events published to Gameplay Logging Service")
        logger.info(f"🔍 Check logger service stats: curl http://localhost:8010/stats")

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())