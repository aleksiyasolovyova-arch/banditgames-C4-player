# RabbitMQ consumer:
# - listens to connect4.events (move.made, game.created, game.finished)
# - checks if it's AI's turn
# - calls ai_manager to decide