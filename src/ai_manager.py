"""
AI Manager for Connect4
Handles multiple AI agents and dynamic difficulty adjustment
"""

import time
import threading
from typing import Dict, Optional, Tuple
from collections import deque

from mcts import MCTS
from ConnectState import ConnectState
from meta import MCTSMeta


class AIAgent:
    """Wrapper for MCTS agent with skill level"""

    def __init__(self, skill_level: str = 'medium'):
        self.skill_level = skill_level
        self.config = MCTSMeta.SKILL_LEVELS[skill_level]
        self.mcts = None

    def get_move(self, state: ConnectState) -> int:
        """Get best move for current state"""
        self.mcts = MCTS(state)
        self.mcts.search(self.config['time_limit'])
        return self.mcts.best_move()

    def get_stats(self) -> Tuple[int, float]:
        """Get search statistics"""
        if self.mcts:
            return self.mcts.statistics()
        return 0, 0


class PlayerPerformance:
    """Track player performance for dynamic difficulty"""

    def __init__(self, window_size: int = 10):
        self.results = deque(maxlen=window_size)
        self.consecutive_wins = 0
        self.consecutive_losses = 0

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


class AIManager:
    """Manages AI agents with dynamic difficulty adjustment"""

    SKILL_ORDER = ['easy', 'medium', 'hard', 'expert']

    def __init__(self):
        self.agents: Dict[str, AIAgent] = {}
        self.performance: Dict[str, PlayerPerformance] = {}
        self.current_skill: Dict[str, str] = {}
        self.lock = threading.Lock()

    def get_agent(self, game_id: str, skill: Optional[str] = None) -> AIAgent:
        """Get or create AI agent for game"""
        with self.lock:
            if skill:
                # Create new agent with specified skill
                self.agents[game_id] = AIAgent(skill)
                self.current_skill[game_id] = skill
            elif game_id not in self.agents:
                # Create default agent
                self.agents[game_id] = AIAgent('medium')
                self.current_skill[game_id] = 'medium'
                self.performance[game_id] = PlayerPerformance()

            return self.agents[game_id]

    def get_ai_move(self, game_id: str, state: ConnectState) -> Tuple[int, Dict]:
        """Get AI move with statistics"""
        agent = self.get_agent(game_id)

        start_time = time.time()
        move = agent.get_move(state)
        elapsed = time.time() - start_time

        rollouts, _ = agent.get_stats()

        return move, {
            'skill_level': agent.skill_level,
            'time': elapsed,
            'rollouts': rollouts
        }

    def update_performance(self, game_id: str, player_won: bool):
        """Update performance and adjust difficulty if needed"""
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
            print(f"Difficulty increased to {new_skill} for game {game_id}")

        elif perf.should_decrease_difficulty() and current_idx > 0:
            new_skill = self.SKILL_ORDER[current_idx - 1]
            self.get_agent(game_id, new_skill)
            print(f"Difficulty decreased to {new_skill} for game {game_id}")

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