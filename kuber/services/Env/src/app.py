from flask import Flask
from flask_cors import CORS
import logging
import json
from config import parse_args
from game.manager import GameManager
from routes.snake import snake_bp
from routes.state import state_bp
from routes.errors import errors_bp
from collections import defaultdict
import logger


# Аргументы командной строки
args = parse_args()

if args.reward_config:
    regular_dict = json.loads(args.reward_config)
    reward_config = defaultdict(int, regular_dict)  # int() returns 0
else:
    reward_config = defaultdict(int)

# Создание объекта game_manager (ПАРАМЕТРЫ ИЗ args!)
game_manager = GameManager(
    grid_width=args.grid_width,
    grid_height=args.grid_height,
    vision_radius=args.vision_radius,
    vision_display_cols=args.vision_display_cols,
    vision_display_rows=args.vision_display_rows,
    fps=args.fps, 
    seed = args.seed,
    reward_config = reward_config,
    max_snakes=args.max_snakes if hasattr(args, 'max_snakes') else 10
)

# Создание приложения
app = Flask(__name__)

if not logger.is_setup():
    logger.setup_as_default(
        flask_app=app
    )

CORS(app)

# Передаём game_manager в app.config
app.config["game_manager"] = game_manager

# Регистрируем blueprints
app.register_blueprint(snake_bp)
app.register_blueprint(state_bp)
app.register_blueprint(errors_bp)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )

