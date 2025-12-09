"""
AI Manager for Connect4 with Enhanced Dynamic Difficulty and Logging.
Adapts difficulty both between games AND during games based on player behavior.
Provides comprehensive statistics for ML training data collection.
"""

import time
import threading
import logging
from typing import Dict, Optional, Tuple, List, Any
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from mcts import MCTS, MoveStatistics
from ConnectState import ConnectState
from meta import MCTSMeta, SelfPlayMeta

logger = logging.getLogger(__name__)


@dataclass
class MCTSResult:
    """Complete result from MCTS search"""
    move: int
    skill_level: str
    base_skill_level: str
    time_limit: float
    actual_time: float
    num_rollouts: int
    nodes_explored: int
    max_depth: int
    exploration_constant: float
    time_adjustment: float
    visit_counts: Dict[int, int]
    q_values: Dict[int, float]
    probabilities: Dict[int, float]
    move_stats: List[Dict]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class OracleResult:
    """Result from Oracle analysis"""
    best_move: int
    visit_counts: Dict[int, int]
    q_values: Dict[int, float]
    probabilities: Dict[int, float]
    num_rollouts: int
    search_time: float
    move_ranking: List[int]  # Moves ranked by visit count

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OracleAgent:
    """
    Maximum-strength MCTS agent for computing optimal moves.

    Used alongside the regular AI to capture what the "best" move
    would have been at each position for ML training comparison.

    This agent:
    - Always runs at maximum strength (no DDA)
    - Does not reuse trees (fresh analysis each time)
    - Returns comprehensive statistics for all legal moves
    """

    def __init__(
        self,
        time_limit: float = 3.0,
        exploration: float = 0.5  # Lower = more exploitation
    ):
        self.time_limit = time_limit
        self.exploration = exploration

    def analyze(self, state: ConnectState) -> OracleResult:
        """
        Analyze a position and return the optimal move with full statistics.

        Args:
            state: Board position to analyze

        Returns:
            OracleResult with best move and statistics for all moves
        """
        mcts = MCTS(state, exploration=self.exploration)

        start_time = time.time()
        mcts.search(self.time_limit)
        search_time = time.time() - start_time

        visit_counts = mcts.get_visit_counts()
        q_values = mcts.get_q_values()
        probabilities = mcts.get_probabilities()

        # Rank moves by visit count (most visited = best)
        move_ranking = sorted(visit_counts.keys(), key=lambda m: visit_counts[m], reverse=True)
        best_move = move_ranking[0] if move_ranking else -1

        return OracleResult(
            best_move=best_move,
            visit_counts=visit_counts,
            q_values=q_values,
            probabilities=probabilities,
            num_rollouts=mcts.num_rollouts,
            search_time=search_time,
            move_ranking=move_ranking
        )


class AIAgent:
    """
    Wrapper for MCTS agent with dynamic skill level.

    Provides:
    - Configurable skill levels
    - Noise injection for variety
    - Temperature-based move selection
    - Comprehensive statistics tracking
    - Tree reuse between moves
    """

    def __init__(
        self,
        skill_level: str = 'medium',
        noise_level: float = 0.0,
        temperature: float = 0.0
    ):
        self.skill_level = skill_level
        self.base_skill = skill_level
        self.config = MCTSMeta.SKILL_LEVELS[skill_level].copy()
        self.noise_level = noise_level
        self.temperature = temperature
        self.mcts: Optional[MCTS] = None
        self.last_result: Optional[MCTSResult] = None

    def get_move(
        self,
        state: ConnectState,
        time_adjustment: float = 1.0
    ) -> Tuple[int, MCTSResult]:
        """
        Get best move with comprehensive statistics.

        Args:
            state: Current game state
            time_adjustment: Multiplier for search time (DDA)

        Returns:
            Tuple of (move, MCTSResult with all statistics)
        """
        exploration = self.config.get('exploration', MCTSMeta.EXPLORATION)

        # Tree reuse: only create new MCTS if none exists, otherwise sync state
        if self.mcts is None:
            self.mcts = MCTS(state, exploration=exploration)
        else:
            self.mcts.sync_state(state)
            self.mcts.exploration = exploration

        base_time = self.config['time_limit']
        adjusted_time = base_time * time_adjustment
        adjusted_time = max(0.05, min(3.0, adjusted_time))

        logger.info(f"\n{'=' * 70}")
        logger.info(f"🔍 AIAgent.get_move() - Before MCTS Search")
        logger.info(f"{'=' * 70}")
        logger.info(f"Skill Level:      {self.skill_level}")
        logger.info(f"Base Time Limit:  {base_time:.3f}s ({base_time * 1000:.1f}ms)")
        logger.info(f"Time Adjustment:  {time_adjustment:.2f}x")
        logger.info(f"Adjusted Time:    {adjusted_time:.3f}s ({adjusted_time * 1000:.1f}ms)")
        logger.info(f"Exploration:      {exploration:.3f}")
        logger.info(f"{'=' * 70}")

        start_time = time.time()
        self.mcts.search(adjusted_time)
        actual_time = time.time() - start_time

        if self.noise_level > 0:
            move = self.mcts.select_move_with_noise(self.noise_level)
        elif self.temperature > 0:
            move = self.mcts.select_move_with_temperature(self.temperature)
        else:
            move = self.mcts.best_move()

        stats = self.mcts.get_comprehensive_stats()

        logger.info(f"\n🔍 AIAgent.get_move() - After MCTS Search")
        logger.info(f"Requested time:   {adjusted_time:.3f}s ({adjusted_time * 1000:.1f}ms)")
        logger.info(f"Actual time:      {actual_time:.3f}s ({actual_time * 1000:.1f}ms)")
        logger.info(f"Rollouts:         {stats.get('num_rollouts', 0)}")
        logger.info(f"Best move:        Column {move}")
        logger.info(f"Max depth:        {stats.get('max_depth', 0)}")
        visit_counts = stats.get('visit_counts', {})
        logger.info(f"Visit counts:     {[visit_counts.get(i, 0) for i in range(7)]}")
        logger.info(f"{'=' * 70}\n")

        # Advance tree with selected move for next turn's reuse
        self.mcts.move(move)

        result = MCTSResult(
            move=move,
            skill_level=self.skill_level,
            base_skill_level=self.base_skill,
            time_limit=adjusted_time,
            actual_time=actual_time,
            num_rollouts=stats.get('num_rollouts', 0),
            nodes_explored=stats.get('node_count', 0),
            max_depth=stats.get('max_depth', 0),
            exploration_constant=exploration,
            time_adjustment=time_adjustment,
            visit_counts=stats.get('visit_counts', {}),
            q_values=stats.get('q_values', {}),
            probabilities=stats.get('probabilities', {}),
            move_stats=stats.get('move_stats', [])
        )
        self.last_result = result
        return move, result

    def get_stats(self) -> Tuple[int, float]:
        """Get basic search statistics"""
        if self.mcts:
            return self.mcts.statistics()
        return 0, 0.0

    def set_skill(self, skill_level: str):
        """Update skill level"""
        if skill_level in MCTSMeta.SKILL_LEVELS:
            self.skill_level = skill_level
            self.config = MCTSMeta.SKILL_LEVELS[skill_level].copy()

    def reset(self):
        """Reset the agent for a new game"""
        self.mcts = None


class MoveHistory:
    """Track move timing and patterns for in-game adaptation"""

    def __init__(self):
        self.move_times: List[Tuple[datetime, float]] = []
        self.last_move_time: Optional[datetime] = None

    def add_move(self, response_time: float):
        """Add a move with its response time"""
        self.move_times.append((datetime.now(), response_time))
        self.last_move_time = datetime.now()

    def get_average_response_time(self, last_n: int = 3) -> float:
        """Get average response time for last N moves"""
        if not self.move_times:
            return 10.0

        recent_moves = self.move_times[-last_n:]
        return sum(t[1] for t in recent_moves) / len(recent_moves)

    def is_player_struggling(self) -> bool:
        """Check if player is taking longer to think"""
        if len(self.move_times) < 3:
            return False

        if len(self.move_times) >= 5:
            early_avg = sum(t[1] for t in self.move_times[:3]) / 3
            recent_avg = sum(t[1] for t in self.move_times[-2:]) / 2
            return recent_avg > early_avg * MCTSMeta.ADAPTATION['struggling_threshold']
        return False

    def is_player_rushing(self) -> bool:
        """Check if player is making very fast moves"""
        if len(self.move_times) < 2:
            return False

        recent_avg = self.get_average_response_time(2)
        return recent_avg < MCTSMeta.ADAPTATION['rushing_threshold']

    def reset(self):
        """Reset move history for new game"""
        self.move_times = []
        self.last_move_time = None


class PlayerPerformance:
    """Enhanced player performance tracking for DDA"""

    def __init__(self, window_size: int = 10):
        self.results = deque(maxlen=window_size)
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.move_history = MoveHistory()
        self.current_game_moves = 0
        self.game_start_time = datetime.now()
        self.total_games = 0
        self.total_wins = 0

    def add_result(self, won: bool):
        """Add game result"""
        self.results.append(1 if won else 0)
        self.total_games += 1

        if won:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.total_wins += 1
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def get_win_rate(self) -> float:
        """Calculate recent win rate"""
        if not self.results:
            return 0.5
        return sum(self.results) / len(self.results)

    def get_overall_win_rate(self) -> float:
        """Get overall win rate"""
        if self.total_games == 0:
            return 0.5
        return self.total_wins / self.total_games

    def should_increase_difficulty(self) -> bool:
        """Check if difficulty should increase"""
        thresholds = MCTSMeta.DDA_THRESHOLDS
        return (self.consecutive_wins >= thresholds['win_streak'] or
                self.get_win_rate() > thresholds['win_rate_high'])

    def should_decrease_difficulty(self) -> bool:
        """Check if difficulty should decrease"""
        thresholds = MCTSMeta.DDA_THRESHOLDS
        return (self.consecutive_losses >= thresholds['loss_streak'] or
                self.get_win_rate() < thresholds['win_rate_low'])

    def add_move_timing(self, response_time: float):
        """Track move timing"""
        self.move_history.add_move(response_time)
        self.current_game_moves += 1

    def get_in_game_adjustment(self) -> float:
        """Calculate in-game difficulty adjustment"""
        base_adjustment = 1.0
        adapt = MCTSMeta.ADAPTATION

        avg_time = self.move_history.get_average_response_time()

        if self.move_history.is_player_struggling():
            base_adjustment *= adapt['struggling_adjustment']
        elif self.move_history.is_player_rushing():
            base_adjustment *= adapt['rushing_adjustment']
        elif avg_time < adapt['comfortable_threshold']:
            base_adjustment *= adapt['comfortable_adjustment']
        elif avg_time > adapt['slow_threshold']:
            base_adjustment *= adapt['slow_adjustment']

        # Win rate adjustment
        win_rate = self.get_win_rate()
        if win_rate > 0.8:
            base_adjustment *= 1.4
        elif win_rate > 0.6:
            base_adjustment *= 1.2
        elif win_rate < 0.2:
            base_adjustment *= 0.6
        elif win_rate < 0.4:
            base_adjustment *= 0.8

        # Game length adjustment
        if self.current_game_moves > 30:
            base_adjustment *= 1.05

        return base_adjustment

    def reset_for_new_game(self):
        """Reset per-game tracking"""
        self.move_history.reset()
        self.current_game_moves = 0
        self.game_start_time = datetime.now()


class AIManager:
    """
    Enhanced AI Manager with comprehensive logging support.

    Features:
    - Dynamic difficulty adjustment (between games and in-game)
    - Multiple concurrent agents
    - Comprehensive statistics tracking
    - Self-play support
    """

    SKILL_ORDER = ['easy', 'medium', 'hard', 'expert']

    def __init__(self):
        self.agents: Dict[str, AIAgent] = {}
        self.performance: Dict[str, PlayerPerformance] = {}
        self.current_skill: Dict[str, str] = {}
        self.last_human_move_time: Dict[str, datetime] = {}
        self.lock = threading.Lock()

    def get_agent(
        self,
        game_id: str,
        skill: Optional[str] = None,
        noise_level: float = 0.0,
        temperature: float = 0.0
    ) -> AIAgent:
        """Get or create AI agent for game"""
        with self.lock:
            if skill:
                self.agents[game_id] = AIAgent(skill, noise_level, temperature)
                self.current_skill[game_id] = skill
                if game_id not in self.performance:
                    self.performance[game_id] = PlayerPerformance()
            elif game_id not in self.agents:
                self.agents[game_id] = AIAgent('medium', noise_level, temperature)
                self.current_skill[game_id] = 'medium'
                self.performance[game_id] = PlayerPerformance()

            return self.agents[game_id]

    def record_human_move_time(self, game_id: str):
        """Record when human made their move"""
        current_time = datetime.now()

        if game_id in self.last_human_move_time:
            response_time = (current_time - self.last_human_move_time[game_id]).total_seconds()

            if game_id in self.performance:
                self.performance[game_id].add_move_timing(response_time)
                logger.debug(f"Human response time: {response_time:.1f}s for game {game_id}")

        self.last_human_move_time[game_id] = current_time

    def get_ai_move(
        self,
        game_id: str,
        state: ConnectState,
        use_dda: bool = True
    ) -> Tuple[int, MCTSResult]:
        """
        Get AI move with full statistics.

        Args:
            game_id: Game identifier
            state: Current game state
            use_dda: Whether to apply dynamic difficulty adjustment

        Returns:
            Tuple of (move, MCTSResult)
        """
        agent = self.get_agent(game_id)

        # Calculate DDA adjustment
        time_adjustment = 1.0
        if use_dda and game_id in self.performance:
            time_adjustment = self.performance[game_id].get_in_game_adjustment()
            logger.debug(f"DDA adjustment: {time_adjustment:.2f}x for game {game_id}")

        # Get move with statistics
        move, result = agent.get_move(state, time_adjustment)

        return move, result

    def update_performance(self, game_id: str, player_won: bool):
        """Update performance and adjust difficulty for next game"""
        if game_id not in self.performance:
            self.performance[game_id] = PlayerPerformance()

        perf = self.performance[game_id]
        perf.add_result(player_won)

        # Check for difficulty adjustment
        current = self.current_skill.get(game_id, 'medium')
        current_idx = self.SKILL_ORDER.index(current)

        if perf.should_increase_difficulty() and current_idx < len(self.SKILL_ORDER) - 1:
            new_skill = self.SKILL_ORDER[current_idx + 1]
            self.get_agent(game_id, new_skill)
            logger.info(f"Difficulty increased to {new_skill} for game {game_id}")

        elif perf.should_decrease_difficulty() and current_idx > 0:
            new_skill = self.SKILL_ORDER[current_idx - 1]
            self.get_agent(game_id, new_skill)
            logger.info(f"Difficulty decreased to {new_skill} for game {game_id}")

        # Reset for new game
        perf.reset_for_new_game()

    def create_self_play_agents(
        self,
        skill1: str = 'medium',
        skill2: str = 'medium',
        noise_level: float = 0.1,
        temperature: float = 0.5
    ) -> Tuple[AIAgent, AIAgent]:
        """
        Create two agents for self-play.

        Args:
            skill1: Skill level for agent 1
            skill2: Skill level for agent 2
            noise_level: Noise for move variety
            temperature: Temperature for softmax selection

        Returns:
            Tuple of (agent1, agent2)
        """
        agent1 = AIAgent(skill1, noise_level, temperature)
        agent2 = AIAgent(skill2, noise_level, temperature)
        return agent1, agent2

    def get_performance_stats(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a game/player"""
        if game_id not in self.performance:
            return None

        perf = self.performance[game_id]
        return {
            'total_games': perf.total_games,
            'total_wins': perf.total_wins,
            'overall_win_rate': perf.get_overall_win_rate(),
            'recent_win_rate': perf.get_win_rate(),
            'consecutive_wins': perf.consecutive_wins,
            'consecutive_losses': perf.consecutive_losses,
            'current_skill': self.current_skill.get(game_id, 'medium'),
            'avg_response_time': perf.move_history.get_average_response_time()
        }

    def cleanup(self, game_id: str):
        """Clean up resources for finished game"""
        with self.lock:
            if game_id in self.agents:
                self.agents[game_id].reset()
            self.agents.pop(game_id, None)
            # Keep performance for future games
            self.current_skill.pop(game_id, None)
            self.last_human_move_time.pop(game_id, None)

    def cleanup_all(self):
        """Clean up all resources"""
        with self.lock:
            self.agents.clear()
            self.performance.clear()
            self.current_skill.clear()
            self.last_human_move_time.clear()