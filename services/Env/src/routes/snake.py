from flask import Blueprint, Response, request, jsonify, current_app
from game.snake import SnakeGame
import json
import time
import datetime
import threading
import logging

snake_bp = Blueprint('snake', __name__)


@snake_bp.route('/snake/<sid>', methods=['GET'])
def get_snake_state(sid):
    """Get current snake state - single JSON response (no streaming)"""
    game_manager = current_app.config["game_manager"]
    if len(game_manager.snakes) >= game_manager.MAX_SNAKES and sid not in game_manager.snakes:
        return jsonify({'error': 'server full'}), 503
    if sid not in game_manager.snakes:
        game_manager.add_snake(sid)

    with game_manager.game_over_lock:
        is_game_over = game_manager.GAME_OVER
    
    with game_manager.snake_locks[sid]:
        vis = game_manager.snakes[sid].get_visible_cells()
        reward = game_manager.snakes[sid].reward
        snake_length = len(game_manager.snakes[sid].snake)
    
    payload = {
        'snake_id': sid,
        'visible_cells': vis,
        'reward': reward,
        'snake_length': snake_length,
        'game_over': is_game_over,
        'episode': game_manager.episode_number,
        'frame': game_manager.frame_number,
        'datetime': datetime.datetime.now().isoformat()
    }
    
    logging.debug({"event": "get_snake_state", "snake_id": sid, "episode": game_manager.episode_number, "frame": game_manager.frame_number})
    
    return jsonify(payload)

@snake_bp.route('/snake/<sid>/move', methods=['POST'])
def move_snake(sid):
    """Process move and advance game one step (synchronous control)"""
    game_manager = current_app.config["game_manager"]
    if sid not in game_manager.snakes:
        return jsonify({'error': 'not found'}), 404
    
    with game_manager.game_over_lock:
        if game_manager.GAME_OVER:
            return jsonify({'snake_id': sid, 'game_over': True})
    
    data = request.get_json(force=True)
    cmd = data.get('move')
    
    # Accept 'forward' as valid (no-op) or left/right turns
    if cmd not in ('left', 'right', 'forward'):
        return jsonify({'error': 'Invalid move'}), 400
    
    # Set turn command for this frame if it's a turn (forward is no-op)
    if cmd in ('left', 'right'):
        game_manager.set_turn_command(sid, cmd)
    
    # CRITICAL: Advance game one step
    game_manager.step_game_once()
    
    # Return success with current game state
    with game_manager.game_over_lock:
        is_game_over = game_manager.GAME_OVER
    
    logging.info({"event": "move_processed", "snake_id": sid, "move": cmd, "frame": game_manager.frame_number})
    
    return jsonify({
        'snake_id': sid, 
        'game_over': is_game_over,
        'success': True
    })
