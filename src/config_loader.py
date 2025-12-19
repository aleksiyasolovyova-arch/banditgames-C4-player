"""
Configuration loader for Connect4 AI Player.

Loads configuration from environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Environment variables can be set in .env file or system environment.
    """

    # Pydantic Settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Service Information
    service_name: str = "connect4-ai-player"
    version: str = "1.0.0"

    ai_player_id: str = Field(..., validation_alias="AI_PLAYER_ID")

    # AI Configuration
    default_skill_level: str = "medium"
    enable_dda: bool = True
    enable_reference_agent: bool = True

    # Connect4 Backend
    connect4_backend_url: str = "http://localhost:8000"

    # Gameplay Logging Service
    gameplay_logging_url: str = "http://localhost:8001"

    # RabbitMQ Configuration
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "user"
    rabbitmq_password: str = "password"
    rabbitmq_exchange: str = "connect4.events"

    # Logging
    log_level: str = "INFO"

    # FastAPI
    host: str = "0.0.0.0"
    port: int = 8002


# Global config instance
config = Config()

