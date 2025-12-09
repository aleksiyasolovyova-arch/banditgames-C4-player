"""
Dataset Exporter for Connect4 ML Training Data.
Exports game data to Parquet format with DVC versioning support.
"""

import os
import json
import logging
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

logger = logging.getLogger(__name__)


class DatasetExporter:
    """
    Export game data to Parquet format for ML training.

    Features:
    - Export moves with board states and MCTS statistics
    - Multiple feature representations (flat, planes)
    - DVC versioning support
    - MinIO remote storage support
    """

    def __init__(
            self,
            db_connection,
            output_dir: str = 'data/datasets',
            dvc_remote: str = None
    ):
        self.db = db_connection
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dvc_remote = dvc_remote

    def fetch_training_data(
            self,
            limit: Optional[int] = None,
            skill_levels: Optional[List[str]] = None,
            date_from: Optional[str] = None,
            date_to: Optional[str] = None,
            include_mcts_stats: bool = True
    ) -> pd.DataFrame:
        """
        Fetch training data from database.

        Args:
            limit: Maximum number of moves to fetch
            skill_levels: Filter by skill levels
            date_from: Start date filter
            date_to: End date filter
            include_mcts_stats: Include MCTS statistics

        Returns:
            DataFrame with training data
        """
        query = """
        SELECT 
            m.game_id,
            m.move_index,
            m.player,
            m.column_played as action,
            m.row_placed,
            m.board_before,
            m.board_after,
            m.thinking_time_ms,
            m.utility_before,
            m.utility_after,
            gs.legal_actions,
            gs.state_hash,
            g.status as game_status,
            g.winner,
            g.player1_skill_level,
            g.player2_skill_level,
            g.total_moves,
            g.duration_seconds
        """

        if include_mcts_stats:
            query += """,
            ms.best_move as mcts_best_move,
            ms.visit_counts,
            ms.q_values,
            ms.num_rollouts,
            ms.skill_level as mcts_skill_level,
            ms.time_adjustment_factor
            """

        query += """
        FROM moves m
        JOIN games g ON m.game_id = g.game_id
        LEFT JOIN game_states gs ON m.state_id = gs.state_id
        """

        if include_mcts_stats:
            query += "LEFT JOIN mcts_statistics ms ON m.move_id = ms.move_id\n"

        query += "WHERE g.status IN ('win', 'draw')\n"

        params = []

        if skill_levels:
            query += "AND (g.player1_skill_level = ANY(%s) OR g.player2_skill_level = ANY(%s))\n"
            params.extend([skill_levels, skill_levels])

        if date_from:
            query += "AND g.created_at >= %s\n"
            params.append(date_from)

        if date_to:
            query += "AND g.created_at <= %s\n"
            params.append(date_to)

        query += "ORDER BY g.created_at, m.move_index\n"

        if limit:
            query += f"LIMIT {limit}"

        # Execute query
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn, params=params if params else None)

        return df

    def process_board_state(self, board_json: Any) -> np.ndarray:
        """
        Convert board JSON to numpy array.

        Args:
            board_json: Board as JSON or list

        Returns:
            Flattened numpy array (42 elements for 6x7 board)
        """
        if isinstance(board_json, str):
            board = json.loads(board_json)
        else:
            board = board_json

        if board is None:
            return np.zeros(42, dtype=np.int8)

        flat = []
        for row in board:
            for cell in row:
                if cell == '.':
                    flat.append(0)
                elif cell == 'X':
                    flat.append(1)
                else:  # 'O'
                    flat.append(2)

        return np.array(flat, dtype=np.int8)

    def compute_outcome_reward(
            self,
            player: str,
            winner: Optional[str],
            game_status: str
    ) -> float:
        """Compute outcome-based reward for a move"""
        if game_status == 'draw':
            return 0.0
        if winner == player:
            return 1.0
        elif winner is not None:
            return -1.0
        return 0.0

    def prepare_training_dataframe(
            self,
            df: pd.DataFrame,
            include_board_features: bool = True
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for ML training.

        Args:
            df: Raw DataFrame from database
            include_board_features: Include flattened board as features

        Returns:
            Processed DataFrame ready for training
        """
        processed = df.copy()

        # Compute outcome rewards
        processed['outcome_reward'] = processed.apply(
            lambda row: self.compute_outcome_reward(
                row['player'], row['winner'], row['game_status']
            ),
            axis=1
        )

        # Convert player to numeric
        processed['player_num'] = processed['player'].map({
            'player1': 1, 'player2': 2
        })

        # Process board states if requested
        if include_board_features:
            # Create board feature columns
            board_features = processed['board_before'].apply(self.process_board_state)

            # Expand into separate columns
            board_df = pd.DataFrame(
                board_features.tolist(),
                columns=[f'cell_{i}' for i in range(42)],
                index=processed.index
            )
            processed = pd.concat([processed, board_df], axis=1)

        # Process MCTS statistics
        if 'visit_counts' in processed.columns:
            processed['has_mcts_stats'] = processed['visit_counts'].notna()

            # Extract visit counts for each column
            def extract_visit_count(vc, col):
                if pd.isna(vc) or vc is None:
                    return 0
                if isinstance(vc, str):
                    vc = json.loads(vc)
                return vc.get(str(col), 0)

            for col in range(7):
                processed[f'visit_count_{col}'] = processed['visit_counts'].apply(
                    lambda x: extract_visit_count(x, col)
                )

        # Convert legal actions to binary mask
        def legal_to_mask(legal):
            if pd.isna(legal) or legal is None:
                return [1] * 7  # All legal if unknown
            if isinstance(legal, str):
                legal = json.loads(legal)
            mask = [0] * 7
            for col in legal:
                mask[col] = 1
            return mask

        legal_masks = processed['legal_actions'].apply(legal_to_mask)
        legal_df = pd.DataFrame(
            legal_masks.tolist(),
            columns=[f'legal_{i}' for i in range(7)],
            index=processed.index
        )
        processed = pd.concat([processed, legal_df], axis=1)

        return processed

    def export_to_parquet(
            self,
            version: str,
            limit: Optional[int] = None,
            skill_levels: Optional[List[str]] = None,
            date_from: Optional[str] = None,
            date_to: Optional[str] = None,
            include_mcts_stats: bool = True,
            include_board_features: bool = True,
            compression: str = 'snappy'
    ) -> Dict[str, Any]:
        """
        Export training data to Parquet file.

        Args:
            version: Dataset version (e.g., 'v1', 'v2')
            limit: Maximum number of moves
            skill_levels: Filter by skill levels
            date_from: Start date
            date_to: End date
            include_mcts_stats: Include MCTS statistics
            include_board_features: Include flattened board features
            compression: Parquet compression algorithm

        Returns:
            Export result with file info
        """
        logger.info(f"Starting dataset export: version={version}")

        # Fetch data
        df = self.fetch_training_data(
            limit=limit,
            skill_levels=skill_levels,
            date_from=date_from,
            date_to=date_to,
            include_mcts_stats=include_mcts_stats
        )

        if len(df) == 0:
            logger.warning("No data to export")
            return {
                'success': False,
                'error': 'No data found matching criteria'
            }

        logger.info(f"Fetched {len(df)} moves from {df['game_id'].nunique()} games")

        # Process data
        processed_df = self.prepare_training_dataframe(
            df, include_board_features=include_board_features
        )

        # Define output path
        filename = f"connect4_dataset_{version}.parquet"
        filepath = self.output_dir / filename

        # Write to Parquet
        table = pa.Table.from_pandas(processed_df)
        pq.write_table(
            table,
            filepath,
            compression=compression
        )

        # Calculate file info
        file_size = filepath.stat().st_size

        # Compute checksum
        with open(filepath, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Get statistics
        num_games = processed_df['game_id'].nunique()
        num_moves = len(processed_df)

        # Get skill level distribution
        skill_dist = {}
        if 'player1_skill_level' in processed_df.columns:
            skill_dist = processed_df['player1_skill_level'].value_counts().to_dict()

        result = {
            'success': True,
            'export_id': str(uuid.uuid4()),
            'version': version,
            'file_path': str(filepath),
            'file_size_bytes': file_size,
            'checksum': checksum,
            'num_games': num_games,
            'num_moves': num_moves,
            'skill_distribution': skill_dist,
            'columns': list(processed_df.columns),
            'created_at': datetime.utcnow().isoformat()
        }

        logger.info(f"Export complete: {filepath} ({file_size / 1024 / 1024:.2f} MB)")

        return result

    def create_dvc_file(self, parquet_path: str, version: str) -> str:
        """
        Create DVC tracking file for dataset.

        Args:
            parquet_path: Path to Parquet file
            version: Dataset version

        Returns:
            Path to DVC file
        """
        dvc_content = f"""# Connect4 Training Dataset {version}
# Generated: {datetime.utcnow().isoformat()}
# Track with: dvc add {parquet_path}

outs:
- md5: null  # Will be filled by dvc add
  size: null
  path: {parquet_path}
"""

        dvc_path = f"{parquet_path}.dvc"
        with open(dvc_path, 'w') as f:
            f.write(dvc_content)

        return dvc_path

    def export_with_dvc(
            self,
            version: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Export dataset and set up DVC tracking.

        Args:
            version: Dataset version
            **kwargs: Arguments passed to export_to_parquet

        Returns:
            Export result with DVC info
        """
        # Export to Parquet
        result = self.export_to_parquet(version, **kwargs)

        if not result['success']:
            return result

        # Create DVC file
        dvc_path = self.create_dvc_file(result['file_path'], version)
        result['dvc_file_path'] = dvc_path

        # Generate DVC commands
        result['dvc_commands'] = {
            'add': f"dvc add {result['file_path']}",
            'push': "dvc push",
            'pull': "dvc pull"
        }

        return result


class DatasetStats:
    """Compute and report dataset statistics"""

    @staticmethod
    def compute_stats(parquet_path: str) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a dataset.

        Args:
            parquet_path: Path to Parquet file

        Returns:
            Dictionary of statistics
        """
        df = pd.read_parquet(parquet_path)

        stats = {
            'total_moves': len(df),
            'total_games': df['game_id'].nunique(),
            'avg_moves_per_game': len(df) / df['game_id'].nunique(),
        }

        # Outcome distribution
        if 'winner' in df.columns:
            winner_counts = df.drop_duplicates('game_id')['winner'].value_counts()
            stats['outcomes'] = winner_counts.to_dict()

        # Skill level distribution
        if 'player1_skill_level' in df.columns:
            stats['skill_levels'] = df['player1_skill_level'].value_counts().to_dict()

        # Action distribution
        if 'action' in df.columns:
            stats['action_distribution'] = df['action'].value_counts().to_dict()

        # MCTS statistics
        if 'num_rollouts' in df.columns:
            mcts_df = df[df['num_rollouts'].notna()]
            if len(mcts_df) > 0:
                stats['mcts'] = {
                    'moves_with_stats': len(mcts_df),
                    'avg_rollouts': mcts_df['num_rollouts'].mean(),
                    'max_rollouts': mcts_df['num_rollouts'].max()
                }

        return stats

    @staticmethod
    def generate_report(parquet_path: str, output_path: str = None) -> str:
        """
        Generate human-readable dataset report.

        Args:
            parquet_path: Path to Parquet file
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        stats = DatasetStats.compute_stats(parquet_path)

        report = f"""
# Connect4 Dataset Report
Generated: {datetime.utcnow().isoformat()}
File: {parquet_path}

## Overview
- Total Moves: {stats['total_moves']:,}
- Total Games: {stats['total_games']:,}
- Average Moves/Game: {stats['avg_moves_per_game']:.1f}

## Game Outcomes
"""
        if 'outcomes' in stats:
            for outcome, count in stats['outcomes'].items():
                report += f"- {outcome}: {count}\n"

        report += "\n## Skill Level Distribution\n"
        if 'skill_levels' in stats:
            for skill, count in stats['skill_levels'].items():
                report += f"- {skill}: {count}\n"

        report += "\n## Action Distribution\n"
        if 'action_distribution' in stats:
            for action, count in stats['action_distribution'].items():
                report += f"- Column {action}: {count} ({count / stats['total_moves'] * 100:.1f}%)\n"

        if 'mcts' in stats:
            report += f"""
## MCTS Statistics
- Moves with MCTS Stats: {stats['mcts']['moves_with_stats']:,}
- Average Rollouts: {stats['mcts']['avg_rollouts']:.0f}
- Max Rollouts: {stats['mcts']['max_rollouts']:,}
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report