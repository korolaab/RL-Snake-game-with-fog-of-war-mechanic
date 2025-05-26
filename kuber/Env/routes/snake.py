from flask import Blueprint, Response, request, jsonify
from game.snake import SnakeGame
import json
import time
import threading

snake_bp = Blueprint('snake', __name__)

@snake_bp.route('/snake/<sid>', methods=['GET'])
def stream_vision(sid):
    if len(game_manager.snakes) >= game_manager.MAX_SNAKES and sid not in game_manager.snakes:
        return jsonify({'error': 'server full'}), 503
    if sid not in game_manager.snakes:
        game_manager.snakes[sid] = SnakeGame(sid)
        game_manager.snake_locks[sid] = threading.Lock()

    def gen():
        while True:
            with game_manager.game_over_lock:
                is_game_over = game_manager.GAME_OVER
            with game_manager.snake_locks[sid]:
                vis = game_manager.snakes[sid].get_visible_cells()
            yield json.dumps({'snake_id': sid, 'visible_cells': vis, 'game_over': is_game_over}) + '\n'
            if is_game_over:
                break
            time.sleep(1.0 / game_manager.snakes[sid].fps)
    return Response(gen(), mimetype='application/x-ndjson', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})

@snake_bp.route('/snake/<sid>/move', methods=['POST'])
def move_snake(sid):
    if sid not in game_manager.snakes:
        return jsonify({'error': 'not found'}), 404
    with game_manager.game_over_lock:
        if game_manager.GAME_OVER:
            return jsonify({'snake_id': sid, 'game_over': True})
    data = request.get_json(force=True)
    cmd = data.get('move')
    if cmd not in ('left', 'right'):
        return jsonify({'error': 'Invalid move'}), 400
    with game_manager.snake_locks[sid]:
        game_manager.snakes[sid].turn(cmd)
    return jsonify({'snake_id': sid, 'game_over': False})

