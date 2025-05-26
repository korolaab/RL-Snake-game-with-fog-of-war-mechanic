from flask import Blueprint, jsonify
from threading import Lock

state_bp = Blueprint('state', __name__)

game_over_lock = Lock()  # Можно убрать, если используется глобально

@state_bp.route('/state', methods=['GET'])
def state():
    """
    Возвращает текущее состояние поля, всех змей, еды, их видимость и глобальный статус.
    """
    grid, visions, statuses, global_game_over = get_game_state()
    return jsonify({
        'grid': grid,
        'visions': visions,
        'status': statuses,
        'global_game_over': global_game_over
    })

@state_bp.route('/reset', methods=['POST'])
def reset():
    """
    Сброс игры — удаляет всех змей и еду, завершает все стримы, создаёт новую еду.
    """
    reset_game()
    return jsonify({'message': 'Game reset successfully'})

@state_bp.route('/', methods=['GET'])
def home():
    """
    Корневой эндпоинт — проверка доступности API и статус игры.
    """
    return jsonify({'message': 'Snake Vision Stream API', 'game_over': is_game_over()})

