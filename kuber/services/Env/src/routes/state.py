from flask import Blueprint, jsonify, current_app
from threading import Lock

state_bp = Blueprint('state', __name__)

game_over_lock = Lock()  # Можно убрать, если используется глобально

@state_bp.route('/reset', methods=['POST'])
def reset():
    """
    Сброс игры — удаляет всех змей и еду, завершает все стримы, создаёт новую еду.
    """
    
    game_manager = current_app.config["game_manager"]
    game_manager.reset_game()
    return jsonify({'message': 'Game reset successfully'})

@state_bp.route('/', methods=['GET'])
def home():
    game_manager = current_app.config["game_manager"]
    """
    Корневой эндпоинт — проверка доступности API и статус игры.
    """
    return jsonify({'message': 'Snake Vision Stream API', 'game_over': game_manager.GAME_OVER})

