"""
AI Manager for Connect4 with Enhanced Dynamic Difficulty
Adapts difficulty both between games AND during games based on player behavior
"""

import time
import threading
from typing import Dict, Optional, Tuple, List
from collections import deque
from datetime import datetime, timedelta

from mcts import MCTS
from ConnectState import ConnectState
from meta import MCTSMeta


class AIAgent:
    """Wrapper for MCTS agent with dynamic skill level"""

    def __init__(self, skill_level: str = 'medium'):
        self.skill_level = skill_level
        self.base_skill = skill_level  # Remember original skill
        self.config = MCTSMeta.SKILL_LEVELS[skill_level].copy()
        self.mcts = None

    def get_move(self, state: ConnectState, time_adjustment: float = 1.0) -> int:
        """Get best move with dynamic time adjustment"""
        self.mcts = MCTS(state)

        # Adjust computation time based on in-game analysis
        adjusted_time = self.config['time_limit'] * time_adjustment
        adjusted_time = max(0.05, min(3.0, adjusted_time))  # Clamp between 50ms and 3s

        self.mcts.search(adjusted_time)
        return self.mcts.best_move()

    def get_stats(self) -> Tuple[int, float]:
        """Get search statistics"""
        if self.mcts:
            return self.mcts.statistics()
        return 0, 0


class MoveHistory:
    """Track move timing and patterns for in-game adaptation"""

    def __init__(self):
        self.move_times = []  # List of (timestamp, response_time) tuples
        self.last_move_time = None

    def add_move(self, response_time: float):
        """Add a move with its response time"""
        self.move_times.append((datetime.now(), response_time))
        self.last_move_time = datetime.now()

    def get_average_response_time(self, last_n: int = 3) -> float:
        """Get average response time for last N moves"""
        if not self.move_times:
            return 10.0  # Default to 10 seconds

        recent_moves = self.move_times[-last_n:]
        avg_time = sum(t[1] for t in recent_moves) / len(recent_moves)
        return avg_time

    def is_player_struggling(self) -> bool:
        """Check if player is taking longer to think (might be struggling)"""
        if len(self.move_times) < 3:
            return False

        # Compare last 2 moves to previous average
        if len(self.move_times) >= 5:
            early_avg = sum(t[1] for t in self.move_times[:3]) / 3
            recent_avg = sum(t[1] for t in self.move_times[-2:]) / 2

            # If recent moves are 50% slower, player might be struggling
            return recent_avg > early_avg * 1.5
        return False

    def is_player_rushing(self) -> bool:
        """Check if player is making very fast moves"""
        if len(self.move_times) < 2:
            return False

        recent_avg = self.get_average_response_time(2)
        return recent_avg < 2.0  # Less than 2 seconds per move


class PlayerPerformance:
    """Enhanced player performance tracking"""

    def __init__(self, window_size: int = 10):
        self.results = deque(maxlen=window_size)
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.move_history = MoveHistory()
        self.current_game_moves = 0
        self.game_start_time = datetime.now()

    def add_result(self, won: bool):
        """Add game result"""
        self.results.append(1 if won else 0)

        if won:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def get_win_rate(self) -> float:
        """Calculate recent win rate"""
        if not self.results:
            return 0.5
        return sum(self.results) / len(self.results)

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
        """Track how long player took to make a move"""
        self.move_history.add_move(response_time)
        self.current_game_moves += 1

    def get_in_game_adjustment(self) -> float:
        """
        Calculate in-game difficulty adjustment based on player behavior
        Returns a multiplier for AI computation time (0.5 = easier, 2.0 = harder)
        """
        base_adjustment = 1.0

        # Adjust based on response time
        avg_time = self.move_history.get_average_response_time()

        if self.move_history.is_player_struggling():
            # Player taking longer = struggling, make AI easier
            base_adjustment *= 0.7
        elif self.move_history.is_player_rushing():
            # Player moving very fast = confident, make AI harder
            base_adjustment *= 1.3
        elif avg_time < 5.0:
            # Quick moves (< 5 seconds) = player is comfortable
            base_adjustment *= 1.1
        elif avg_time > 15.0:
            # Very slow moves (> 15 seconds) = player needs help
            base_adjustment *= 0.8

        # Adjust based on win rate (more gradual than between-game adjustment)
        win_rate = self.get_win_rate()
        if win_rate > 0.8:  # Dominating
            base_adjustment *= 1.4
        elif win_rate > 0.6:  # Winning comfortably
            base_adjustment *= 1.2
        elif win_rate < 0.2:  # Struggling badly
            base_adjustment *= 0.6
        elif win_rate < 0.4:  # Losing often
            base_adjustment *= 0.8

        # Adjust based on game length (longer games = closer match)
        if self.current_game_moves > 30:
            # Long game = evenly matched, slight increase
            base_adjustment *= 1.05

        return base_adjustment


class AIManager:
    """Enhanced AI Manager with in-game adaptation"""

    SKILL_ORDER = ['easy', 'medium', 'hard', 'expert']

    def __init__(self):
        self.agents: Dict[str, AIAgent] = {}
        self.performance: Dict[str, PlayerPerformance] = {}
        self.current_skill: Dict[str, str] = {}
        self.last_human_move_time: Dict[str, datetime] = {}
        self.lock = threading.Lock()

    def get_agent(self, game_id: str, skill: Optional[str] = None) -> AIAgent:
        """Get or create AI agent for game"""
        with self.lock:
            if skill:
                # Create new agent with specified skill
                self.agents[game_id] = AIAgent(skill)
                self.current_skill[game_id] = skill
                if game_id not in self.performance:
                    self.performance[game_id] = PlayerPerformance()
            elif game_id not in self.agents:
                # Create default agent
                self.agents[game_id] = AIAgent('medium')
                self.current_skill[game_id] = 'medium'
                self.performance[game_id] = PlayerPerformance()

            return self.agents[game_id]

    def record_human_move_time(self, game_id: str):
        """Record when human made their move"""
        current_time = datetime.now()

        if game_id in self.last_human_move_time:
            # Calculate response time
            response_time = (current_time - self.last_human_move_time[game_id]).total_seconds()

            if game_id in self.performance:
                self.performance[game_id].add_move_timing(response_time)
                logger.info(f"Human response time: {response_time:.1f}s for game {game_id}")

        self.last_human_move_time[game_id] = current_time

    def get_ai_move(self, game_id: str, state: ConnectState) -> Tuple[int, Dict]:
        """Get AI move with dynamic in-game adjustment"""
        agent = self.get_agent(game_id)

        # Calculate in-game difficulty adjustment
        time_adjustment = 1.0
        if game_id in self.performance:
            time_adjustment = self.performance[game_id].get_in_game_adjustment()
            logger.info(f"In-game adjustment: {time_adjustment:.2f}x for game {game_id}")

        start_time = time.time()
        move = agent.get_move(state, time_adjustment)
        elapsed = time.time() - start_time

        rollouts, _ = agent.get_stats()

        return move, {
            'skill_level': agent.skill_level,
            'base_skill': agent.base_skill,
            'time': elapsed,
            'time_adjustment': time_adjustment,
            'rollouts': rollouts,
            'adaptive_difficulty': True
        }

    def update_performance(self, game_id: str, player_won: bool):
        """Update performance and adjust base difficulty for next game"""
        if game_id not in self.performance:
            self.performance[game_id] = PlayerPerformance()

        perf = self.performance[game_id]
        perf.add_result(player_won)

        # Check for difficulty adjustment (between games)
        current = self.current_skill.get(game_id, 'medium')
        current_idx = self.SKILL_ORDER.index(current)

        if perf.should_increase_difficulty() and current_idx < len(self.SKILL_ORDER) - 1:
            new_skill = self.SKILL_ORDER[current_idx + 1]
            self.get_agent(game_id, new_skill)
            logger.info(f"Base difficulty increased to {new_skill} for next game")

        elif perf.should_decrease_difficulty() and current_idx > 0:
            new_skill = self.SKILL_ORDER[current_idx - 1]
            self.get_agent(game_id, new_skill)
            logger.info(f"Base difficulty decreased to {new_skill} for next game")

    def create_concurrent_agents(self, num_agents: int, skill_levels: list = None) -> Dict[str, AIAgent]:
        """Create multiple agents for concurrent play"""
        if skill_levels is None:
            skill_levels = ['medium'] * num_agents

        agents = {}
        for i, skill in enumerate(skill_levels[:num_agents]):
            agent_id = f"agent_{i}"
            agents[agent_id] = AIAgent(skill)

        return agents

    def cleanup(self, game_id: str):
        """Clean up resources for finished game"""
        with self.lock:
            self.agents.pop(game_id, None)
            self.performance.pop(game_id, None)
            self.current_skill.pop(game_id, None)
            self.last_human_move_time.pop(game_id, None)


# Add logger
import logging
logger = logging.getLogger(__name__)