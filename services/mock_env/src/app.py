from flask import Flask, Response, request, jsonify
import json
import time
import threading
from collections import defaultdict

app = Flask(__name__)

# Mock data storage
snake_counters = defaultdict(int)  # Track how many times each snake has been called
snake_locks = defaultdict(threading.Lock)

# Sample visible cells data (you can modify this as needed)
SAMPLE_VISIBLE_CELLS = {
    '10,5': 'EMPTY', '9,6': 'EMPTY', '9,5': 'EMPTY', '9,4': 'EMPTY', 
    '8,7': 'EMPTY', '8,6': 'EMPTY', '8,5': 'EMPTY', '8,4': 'EMPTY', 
    '8,3': 'EMPTY', '7,8': 'EMPTY', '7,7': 'EMPTY', '7,6': 'EMPTY', 
    '7,5': 'EMPTY', '7,4': 'EMPTY', '7,3': 'EMPTY', '7,2': 'EMPTY', 
    '6,9': 'EMPTY', '6,8': 'EMPTY', '6,7': 'EMPTY', '6,6': 'EMPTY', 
    '6,5': 'EMPTY', '6,4': 'EMPTY', '6,3': 'EMPTY', '6,2': 'EMPTY', 
    '6,1': 'EMPTY', '5,10': 'EMPTY', '5,9': 'EMPTY', '5,8': 'EMPTY', 
    '5,7': 'BODY', '5,6': 'BODY', '5,5': 'HEAD', '5,4': 'EMPTY', 
    '5,3': 'EMPTY', '5,2': 'EMPTY', '5,1': 'EMPTY', '5,0': 'EMPTY', 
    '4,9': 'EMPTY', '4,8': 'EMPTY', '4,7': 'EMPTY', '4,6': 'EMPTY', 
    '4,5': 'EMPTY', '4,4': 'EMPTY', '4,3': 'EMPTY', '4,2': 'EMPTY', 
    '4,1': 'EMPTY', '3,8': 'EMPTY', '3,7': 'EMPTY', '3,6': 'EMPTY', 
    '3,5': 'EMPTY', '3,4': 'EMPTY', '3,3': 'EMPTY', '3,2': 'EMPTY', 
    '2,7': 'EMPTY', '2,6': 'EMPTY', '2,5': 'EMPTY', '2,4': 'EMPTY', 
    '2,3': 'EMPTY', '1,6': 'EMPTY', '1,5': 'EMPTY', '1,4': 'EMPTY', 
    '0,5': 'EMPTY'
}

@app.route('/snake/<sid>', methods=['GET'])
def stream_vision(sid):
    """
    Mock endpoint that returns 10 normal responses followed by game_over=True
    """
    def generate():
        with snake_locks[sid]:
            snake_counters[sid] += 1
            current_count = snake_counters[sid]
        
        # Return 10 normal responses, then game over
        for i in range(10):
            response_data = {
                'snake_id': sid,
                'visible_cells': SAMPLE_VISIBLE_CELLS.copy(),
                'reward': 1,
                'game_over': False
            }
            yield json.dumps(response_data) + '\n'
            time.sleep(0.1)  # Small delay between responses
        
        # Final response with game_over=True
        final_response = {
            'snake_id': sid,
            'visible_cells': SAMPLE_VISIBLE_CELLS.copy(),
            'reward': 1,
            'game_over': True
        }
        yield json.dumps(final_response) + '\n'
    
    return Response(
        generate(), 
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

@app.route('/snake/<sid>/move', methods=['POST'])
def move_snake(sid):
    """
    Mock endpoint for snake movement
    """
    try:
        data = request.get_json(force=True)
        cmd = data.get('move')
        
        if cmd not in ('left', 'right'):
            return jsonify({'error': 'Invalid move'}), 400
        
        # Mock response - always successful for valid moves
        return jsonify({
            'snake_id': sid,
            'game_over': False
        })
    
    except Exception as e:
        return jsonify({'error': 'Invalid JSON'}), 400

@app.route('/snake/<sid>/reset', methods=['POST'])
def reset_snake(sid):
    """
    Optional: Reset the counter for a snake (useful for testing)
    """
    with snake_locks[sid]:
        snake_counters[sid] = 0
    
    return jsonify({
        'snake_id': sid,
        'message': 'Snake reset successfully'
    })

@app.route('/', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({'status': 'healthy', 'service': 'snake-mock-api'})

if __name__ == '__main__':
    print("Starting Snake Game Mock API...")
    print("Available endpoints:")
    print("  GET  /snake/<sid>       - Stream vision (10 responses + game_over)")
    print("  POST /snake/<sid>/move  - Move snake (expects {'move': 'left'|'right'})")
    print("  POST /snake/<sid>/reset - Reset snake counter")
    print("  GET  /          - Health check")
    print("\nExample usage:")
    print("  curl http://localhost:5000/snake/test123")
    print("  curl -X POST http://localhost:5000/snake/test123/move -H 'Content-Type: application/json' -d '{\"move\":\"left\"}'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
