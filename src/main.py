"""
Main FastAPI application for Connect4 AI Player.

This service:
1. Listens to RabbitMQ events from Connect4 Backend
2. Responds to moves with AI decisions
3. Logs statistics to Gameplay Logging Service
4. Provides health check and status endpoints
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config_loader import config
from .utils.logger import setup_logging
from .agents.ai_manager import AIManager
from .api.backend_client import Connect4BackendClient
from .api.event_publisher import AIPlayerEventPublisher
from .events.event_handler import EventHandler
from .events.rabbitmq_listener import RabbitMQListener

# Setup logging
setup_logging(level=config.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown:
    - Initialize AI Manager
    - Initialize API clients
    - Start RabbitMQ listener
    - Cleanup on shutdown
    """
    logger.info(f"Starting {config.service_name} v{config.version}...")

    try:
        # STEP 1: Initialize AI Components
        logger.info("STEP 1: Initializing AI Components")

        ai_manager = AIManager(
            default_skill=config.default_skill_level,
            enable_reference_agent=config.enable_reference_agent,
            enable_dda=config.enable_dda
        )
        logger.info(" AI Manager initialized")

        # Initialize API clients
        backend_client = Connect4BackendClient(
            base_url=config.connect4_backend_url
        )
        logger.info(f" Connect4 Backend client initialized: {config.connect4_backend_url}")

        # Initialize Event Publisher for logging
        event_publisher = AIPlayerEventPublisher(
            host=config.rabbitmq_host,
            port=config.rabbitmq_port,
            username=config.rabbitmq_user,
            password=config.rabbitmq_password
        )
        logger.info("Event Publisher initialized for gameplay logging")

        # Initialize Event Handler
        event_handler = EventHandler(
            ai_player_id=config.ai_player_id,
            ai_manager=ai_manager,
            backend_client=backend_client,
            event_publisher=event_publisher
        )
        logger.info("Event Handler initialized")
        logger.info("")

        # STEP 2: Initialize RabbitMQ Listener
        logger.info("STEP 2: Connecting to RabbitMQ")

        rabbitmq_listener = RabbitMQListener(
            host=config.rabbitmq_host,
            port=config.rabbitmq_port,
            username=config.rabbitmq_user,
            password=config.rabbitmq_password,
            exchange=config.rabbitmq_exchange,
            event_handler=event_handler
        )
        logger.info(" RabbitMQ Listener initialized")

        # Store in app state for access in endpoints
        app.state.ai_manager = ai_manager
        app.state.backend_client = backend_client
        app.state.event_publisher = event_publisher
        app.state.event_handler = event_handler
        app.state.rabbitmq_listener = rabbitmq_listener

        # Start RabbitMQ listener in background task
        listener_task = asyncio.create_task(rabbitmq_listener.start())
        app.state.listener_task = listener_task

        logger.info(f" {config.service_name} STARTED SUCCESSFULLY!")
        logger.info(f"AI Player ID: {config.ai_player_id[:8]}...")
        logger.info(f"Default Skill: {config.default_skill_level}")
        logger.info(f"DDA Enabled: {config.enable_dda}")
        logger.info(f"Reference Agent Enabled: {config.enable_reference_agent}")
        logger.info(f"Listening on: {config.host}:{config.port}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"STARTUP FAILED: {e}")
        logger.error("", exc_info=True)
        raise

    yield  # Application is running

    # Shutdown
    logger.info("")
    logger.info(f"Shutting down {config.service_name}...")

    try:
        # Stop RabbitMQ listener
        if hasattr(app.state, 'rabbitmq_listener'):
            app.state.rabbitmq_listener.stop()
            logger.info("RabbitMQ listener stopped")

        # Close API clients
        if hasattr(app.state, 'backend_client'):
            await app.state.backend_client.close()
            logger.info(" Backend client closed")

        if hasattr(app.state, 'event_publisher'):
            app.state.event_publisher.close()
            logger.info(" Event publisher closed")

        # Cancel listener task
        if hasattr(app.state, 'listener_task'):
            app.state.listener_task.cancel()
            try:
                await app.state.listener_task
            except asyncio.CancelledError:
                pass

        logger.info(" Shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI app
app = FastAPI(
    title="Connect4 AI Player",
    description=(
        "AI Player service for Connect Four. "
        "Listens to game events and responds with intelligent moves using MCTS."
    ),
    version=config.version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": config.service_name,
        "version": config.version,
        "status": "running",
        "ai_player_id": config.ai_player_id[:8] + "..."
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns service status and component health.
    """
    ai_manager = getattr(app.state, 'ai_manager', None)
    event_handler = getattr(app.state, 'event_handler', None)

    active_games = 0
    if ai_manager:
        active_games = len(ai_manager.get_active_games())

    return {
        "status": "healthy",
        "service": config.service_name,
        "version": config.version,
        "ai_player_id": config.ai_player_id[:8] + "...",
        "components": {
            "ai_manager": ai_manager is not None,
            "event_handler": event_handler is not None,
            "backend_client": hasattr(app.state, 'backend_client'),
            "event_publisher": hasattr(app.state, 'event_publisher'),
            "rabbitmq_listener": hasattr(app.state, 'rabbitmq_listener')
        },
        "active_games": active_games,
        "config": {
            "default_skill": config.default_skill_level,
            "dda_enabled": config.enable_dda,
            "reference_agent_enabled": config.enable_reference_agent
        }
    }


@app.get("/status")
async def status():
    """
    Detailed status endpoint.

    Returns detailed information about active games and AI performance.
    """
    ai_manager = getattr(app.state, 'ai_manager', None)
    event_handler = getattr(app.state, 'event_handler', None)

    if not ai_manager or not event_handler:
        return {"error": "Services not initialized"}

    active_games = ai_manager.get_active_games()

    # Get DDA stats for each active game
    dda_stats = {}
    for game_id in active_games:
        stats = ai_manager.get_dda_stats(game_id)
        if stats:
            dda_stats[game_id] = stats

    return {
        "service": config.service_name,
        "version": config.version,
        "ai_player_id": config.ai_player_id[:8] + "...",
        "active_games_count": len(active_games),
        "active_games": active_games[:10],  # Limit to 10 for brevity
        "dda_stats": dda_stats,
        "config": {
            "default_skill": config.default_skill_level,
            "dda_enabled": config.enable_dda,
            "reference_agent_enabled": config.enable_reference_agent,
            "backend_url": config.connect4_backend_url
        }
    }


# =============================================================================
# Run Application
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level=config.log_level.lower()
    )

