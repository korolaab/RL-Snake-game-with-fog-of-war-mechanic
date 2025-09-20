# State Management: Environment ↔ Inference

This document describes how game state is stored, processed, and transmitted between the Environment and Inference services in the Snake RL system.

## Overview

The system uses a **distributed state architecture** where:
- **Environment Service**: Maintains authoritative game state
- **Inference Service**: Receives partial observations via HTTP streaming
- **State Processing**: Converts observations to neural network inputs

## Environment State Storage

### Global Game State

**File**: [`services/Env/src/game/manager.py:13-44`](/app/services/Env/src/game/manager.py#L13-L44)

```python
class GameManager:
    def __init__(self, ...):
        self.GRID_WIDTH = grid_width        # 11 (world width)
        self.GRID_HEIGHT = grid_height      # 11 (world height)  
        self.VISION_RADIUS = vision_radius  # 5 (snake vision range)
        self.MAX_SNAKES = max_snakes        # Maximum concurrent snakes
        self.FOODS = set()                  # Global food positions: {(x,y), ...}
        self.snakes = {}                    # All snakes: {snake_id: SnakeGame}
        self.snake_locks = {}               # Thread locks: {snake_id: Lock}
        self.GAME_OVER = False              # Global game over flag
        self.episode_number = 0             # Current episode counter
        self.frame_number = 0               # Current frame counter
```

#### World Grid Representation

**File**: [`services/Env/src/game/manager.py:46-64`](/app/services/Env/src/game/manager.py#L46-L64)

The environment maintains a complete 11×11 world grid:

```python
def state(self):
    # Create full world grid dictionary
    grid = {f"{x},{y}": [] for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)}
    # grid = {"0,0": [], "0,1": [], ..., "10,10": []}  # 121 cells total
    
    # Add snake positions to grid
    for sid, game in self.snakes.items():
        for i, p in enumerate(game.snake):
            cell = f"{p[0]},{p[1]}"         # "x,y" coordinate string
            typ = 'HEAD' if i == 0 else 'BODY'
            grid[cell].append({'type': typ, 'snake_id': sid})
    
    # Add food positions
    for food in self.FOODS:
        cell = f"{food[0]},{food[1]}"
        grid[cell].append({'type': 'FOOD', 'snake_id': None})
    
    # Fill empty cells
    for cell, v in grid.items():
        if not v:
            grid[cell] = [{'type': 'EMPTY'}]
    
    return grid, visions, statuses, self.GAME_OVER
```

**Grid Cell Structure**:
```python
# Example world state:
{
    "5,5": [{'type': 'HEAD', 'snake_id': 'adam'}],
    "4,5": [{'type': 'BODY', 'snake_id': 'adam'}], 
    "3,5": [{'type': 'BODY', 'snake_id': 'adam'}],
    "8,3": [{'type': 'FOOD', 'snake_id': None}],
    "2,7": [{'type': 'EMPTY'}],
    # ... 121 total cells
}
```

### Snake-Specific State

**File**: [`services/Env/src/game/snake.py:6-26`](/app/services/Env/src/game/snake.py#L6-L26)

Each snake maintains its own state:

```python
class SnakeGame:
    def __init__(self, snake_id, game):
        self.snake_id = snake_id                    # Unique identifier
        self.grid_width = game.GRID_WIDTH           # 11
        self.grid_height = game.GRID_HEIGHT         # 11
        self.vision_radius = game.VISION_RADIUS     # 5
        self.direction = (1, 0)                     # Current direction vector
        self.snake = []                             # Body positions: [(x,y), ...]
        self.ticks = 0                              # Frame counter since spawn
        self.reward = 0                             # Current frame reward
        self.stepsSinceLastApple = 0                # Starvation counter
        self._last_known_state = {}                 # Cached vision data
```

**Snake Body Representation**:
```python
# Example snake state:
self.snake = [(5, 5), (4, 5), (3, 5)]  # Head at (5,5), body segments follow
self.direction = (1, 0)                 # Moving right (+x direction)
self.reward = 0.1                       # Alive reward this frame
```

### Vision System

**File**: [`services/Env/src/game/snake.py:126-167`](/app/services/Env/src/game/snake.py#L126-L167)

Each snake calculates its own **partial observation** using diamond-shaped vision:

```python
def _calc_visible_cells(self):
    head = self.snake[0]  # Snake head position
    
    # Direction-based coordinate rotation
    rotate_map = {
        (0, -1): lambda dx, dy: (dx, dy),    # North: no rotation
        (1, 0):  lambda dx, dy: (dy, -dx),   # East: 90° clockwise  
        (0, 1):  lambda dx, dy: (-dx, -dy),  # South: 180°
        (-1, 0): lambda dx, dy: (-dy, dx)    # West: 90° counter-clockwise
    }
    rotate = rotate_map.get(self.direction, rotate_map[(0, -1)])
    
    # Calculate what other snakes look like from this snake's perspective
    other_heads = {g.snake[0] for sid, g in self.snakes.items() 
                   if sid != self.snake_id and g.snake}
    other_bodies = {pos for sid, g in self.snakes.items() 
                    if sid != self.snake_id for pos in g.snake[1:]}
    
    vis = {}
    for dx in range(-self.vision_radius, self.vision_radius + 1):      # -5 to +5
        for dy in range(-self.vision_radius, self.vision_radius + 1):  # -5 to +5
            # Diamond shape constraint (Manhattan distance ≤ 5)
            if abs(dx) + abs(dy) > self.vision_radius:
                continue
            
            # Rotate coordinates based on snake's heading direction
            rx, ry = rotate(dx, dy)
            
            # Map to 11×11 vision display coordinates (snake-centered)
            cx = self.vision_display_cols // 2 + rx  # 5 + rx
            cy = self.vision_display_rows // 2 + ry  # 5 + ry
            
            # Skip if outside vision display bounds
            if not (0 <= cx < self.vision_display_cols and 0 <= cy < self.vision_display_rows):
                continue
            
            # Calculate world position (with wrapping boundaries)
            px = (head[0] + dx) % self.grid_width   # Wrap around world edges
            py = (head[1] + dy) % self.grid_height
            pos = (px, py)
            
            # Classify cell content based on what's at this position
            if pos == head:
                obj = 'HEAD'           # Own snake head
            elif pos in self.snake[1:]:
                obj = 'BODY'           # Own snake body
            elif pos in other_heads:
                obj = 'OTHER_HEAD'     # Other snake head
            elif pos in other_bodies:
                obj = 'OTHER_BODY'     # Other snake body
            elif pos in self.foods:
                obj = 'FOOD'           # Food item
            else:
                obj = 'EMPTY'          # Empty space
            
            # Store in vision dictionary with display coordinates as key
            vis[f"{cx},{cy}"] = obj
    
    return vis
```

**Vision Diamond Pattern** (radius=5):
```
       F
     F F F  
   F F F F F
 F F F F F F F
F F F F H F F F F    H = Head (center)
 F F F F F F F      F = Visible cell
   F F F F F        Up to 61 cells total
     F F F
       F
```

**Vision Dictionary Structure**:
```python
# Example vision output:
{
    "5,5": "HEAD",      # Snake's own head at center
    "4,5": "BODY",      # Own body segment  
    "6,5": "EMPTY",     # Empty space
    "7,4": "FOOD",      # Food item
    "3,6": "OTHER_BODY" # Other snake body
    # ... up to 61 cells total
}
```

### State Caching & Thread Safety

**File**: [`services/Env/src/game/snake.py:57-85`](/app/services/Env/src/game/snake.py#L57-L85)

Vision state is cached for performance and thread safety:

```python
def _update_visible_state(self):
    """Update the last known visible state"""
    try:
        new_state = self._calc_visible_cells()  # Calculate fresh vision
        with self._state_lock:
            self._last_known_state = new_state  # Thread-safe update
    except Exception as e:
        logging.error(f"Error updating visible state for snake {self.snake_id}: {e}")

def get_visible_cells(self):
    """Retrieve the last known visible state"""
    try:
        with self._state_lock:
            return self._last_known_state.copy()  # Return copy to prevent races
    except Exception as e:
        return {}
```

## HTTP State Transmission

### Stream Endpoint

**File**: [`services/Env/src/routes/snake.py:20-40`](/app/services/Env/src/routes/snake.py#L20-L40)

The environment streams state to inference via HTTP:

```python
@snake_bp.route('/snake/<sid>', methods=['GET'])
def stream_vision(sid):
    def gen():
        while True:
            # Get current game state with thread safety
            with game_manager.game_over_lock:
                is_game_over = game_manager.GAME_OVER
            with game_manager.snake_locks[sid]:
                vis = game_manager.snakes[sid].get_visible_cells()  # Cached vision
                reward = game_manager.snakes[sid].reward            # Current reward
            
            # Construct payload for this frame
            payload = {
                'snake_id': sid,
                'visible_cells': vis,         # Vision dictionary
                'reward': reward,             # Scalar reward value  
                'game_over': is_game_over,    # Boolean flag
                'episode': game_manager.episode_number,   # Episode counter
                'frame': game_manager.frame_number,       # Frame counter  
                'datetime': datetime.datetime.now().isoformat()  # Timestamp
            }
            
            yield json.dumps(payload) + '\n'  # NDJSON format
            
            if is_game_over:
                break
            time.sleep(1.0 / game_manager.FPS)  # Match game framerate (100 FPS)
```

**HTTP Stream Data Format**:
```json
{
  "snake_id": "adam",
  "visible_cells": {
    "5,5": "HEAD",
    "4,5": "BODY", 
    "6,5": "EMPTY",
    "7,4": "FOOD"
  },
  "reward": 0.1,
  "game_over": false,
  "episode": 5,
  "frame": 142,
  "datetime": "2025-01-15T14:30:15.123456"
}
```

## Inference State Processing

### Stream Reception

**File**: [`services/Inference/src/main.py:65-89`](/app/services/Inference/src/main.py#L65-L89)

The inference service receives the HTTP stream in a background thread:

```python
class StreamReader:
    def _read_stream(self):
        """Background thread that continuously reads stream"""
        while self.running:
            try:
                response = requests.get(self.base_url, stream=True, timeout=10)
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode()
                        data = json.loads(decoded)  # Parse JSON payload
                        
                        with self.lock:
                            self.latest_state = data        # Store latest state
                            self.latest_timestamp = time.time()
                        self.new_state_event.set()          # Notify main thread
```

### State Extraction

**File**: [`services/Inference/src/main.py:147-183`](/app/services/Inference/src/main.py#L147-L183)

The main inference loop processes received states:

```python
while True:
    data, tech_timestamp = stream_reader.get_latest_state()
    
    # Extract state components
    episode_count = data.get("episode", 'null')
    frame_count = data.get("frame", 'null')  
    visible_cells = data.get("visible_cells", 'null')
    reward = data.get("reward", 'null')
    game_over = data.get("game_over", 'null')
    
    # Validate state completeness
    if (visible_cells == 'null' or episode_count == 'null' or 
        frame_count == 'null' or reward == 'null' or game_over == 'null'):
        logging.error({"event": "incomplete_state", "action": "skipped"})
        continue
    
    # Process state for neural network
    action = agent.predict_action(data)  # Convert to tensor and infer
```

### Neural Network State Processing

**File**: [`services/Inference/src/state_processor.py:17-73`](/app/services/Inference/src/state_processor.py#L17-L73)

The state processor converts vision data to neural network inputs:

```python
class StateProcessor:
    def __init__(self):
        # Cell type encoding for neural network
        self.cell_encoding = {
            'FOOD': [0, 0, 1],      # Food channel
            'BODY': [0, 1, 0],      # Own body channel
            'OTHER_BODY': [1, 0, 0], # Other snake channel  
            'EMPTY': [0, 0, 0]      # Empty space
            # ⚠️ PROBLEM: Missing 'HEAD', 'OTHER_HEAD' encodings
        }
    
    def process_state(self, state):
        visible_cells = state.get('visible_cells', {})
        
        # ⚠️ CRITICAL PROBLEM: Remove HEAD information
        filtered_cells = {k: v for k, v in visible_cells.items() if v != 'HEAD'}
        
        if not filtered_cells:
            # ⚠️ PROBLEM: Inconsistent fallback tensor size
            return torch.zeros(3, 60)  # 180 elements
        
        # Sort cells by coordinates for consistent ordering
        sorted_cells = []
        for coord_str, cell_type in filtered_cells.items():
            try:
                x, y = map(int, coord_str.split(','))
                sorted_cells.append((x, y, cell_type))
            except ValueError:
                continue
        
        sorted_cells.sort(key=lambda item: (item[0], item[1]))  # Sort by (x,y)
        
        # Create tensor data
        tensor_data = []
        for x, y, cell_type in sorted_cells:
            encoding = self.cell_encoding.get(cell_type, [0, 0, 0])
            tensor_data.append(encoding)
        
        # ⚠️ CRITICAL PROBLEM: Variable tensor size
        result = torch.tensor(tensor_data, dtype=torch.float32).flatten()
        # Shape: [N*3] where N = number of visible cells (VARIES!)
        
        return result
```

### Experience Storage

**File**: [`services/Inference/src/snake_agent.py:112-121`](/app/services/Inference/src/snake_agent.py#L112-L121)

Each state-action-reward tuple is stored as an experience:

```python
def add_experience(self, state, action, reward, next_state=None, done=False):
    experience = {
        'state': state,              # Complete HTTP payload (dict)
        'action': action,            # Action taken ("left"/"right"/"forward")
        'reward': reward,            # Scalar reward (0.1, 1.0, -1.0)
        'next_state': next_state,    # Next state (usually None)
        'done': done,                # Episode termination flag
        'step': self.current_episode_steps  # Step counter within episode
    }
    self.current_episode_experiences.append(experience)
```

### Data Persistence

**File**: [`services/Inference/src/data_manager.py:17-26`](/app/services/Inference/src/data_manager.py#L17-L26)

Experiences are stored in memory and can be saved to disk:

```python
class DataManager:
    def add_experience(self, state, reward, action):
        experience = {
            'state': state,                        # Raw HTTP state dict
            'reward': reward,                      # Scalar reward
            'action': action,                      # Action string
            'timestamp': datetime.now().isoformat()  # When experience occurred
        }
        self.history.append(experience)  # Add to memory list
        
    def save_history(self, snake_id: str, filename_suffix: str = None):
        # Save as pickle file: "history_adam_20250115_143015.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)
```

## Critical State Processing Issues

### 1. **Variable Tensor Dimensions** 🚨
```python
# Problem: Neural network input size changes between frames
# Frame 1: 15 visible cells → tensor shape [45]  (15*3)
# Frame 2: 23 visible cells → tensor shape [69]  (23*3)  
# Frame 3: 8 visible cells  → tensor shape [24]  (8*3)
# → Causes PyTorch dimension mismatch errors
```

### 2. **Loss of Critical Information** 🚨
```python
# Problem: HEAD position is filtered out
filtered_cells = {k: v for k, v in visible_cells.items() if v != 'HEAD'}
# → Agent cannot see its own head position (most important reference point)
```

### 3. **Missing Object Type Encodings** ⚠️
```python
# Problem: State processor doesn't handle all environment object types
# Environment sends: 'HEAD', 'BODY', 'OTHER_HEAD', 'OTHER_BODY', 'FOOD', 'EMPTY'  
# Processor handles: 'BODY', 'OTHER_BODY', 'FOOD', 'EMPTY'
# Missing: 'HEAD', 'OTHER_HEAD' → Treated as [0,0,0] (same as EMPTY)
```

### 4. **Spatial Structure Loss** ⚠️
```python
# Problem: 2D spatial vision is flattened to 1D list
# Original: 11×11 vision grid with spatial relationships
# Processed: Flat list sorted by (x,y) coordinates  
# → Neural network cannot understand spatial geometry
```

### 5. **State Synchronization Issues** ⚠️
```python
# Problem: Vision state is cached separately from episode/frame counters
# Vision: Updated in snake.update() during game loop
# Counters: Updated in manager.game_loop() 
# → Temporal inconsistency between vision and metadata
```

## Recommended Fixes

### 1. **Fixed Tensor Size**
```python
def process_state(self, state):
    # Create fixed 11×11×3 tensor (preserves spatial structure)
    vision_tensor = torch.zeros(11, 11, 3)
    
    for coord_str, cell_type in visible_cells.items():
        x, y = map(int, coord_str.split(','))
        if 0 <= x < 11 and 0 <= y < 11:
            encoding = self.get_full_encoding(cell_type)  # Include ALL types
            vision_tensor[y, x] = torch.tensor(encoding)
    
    return vision_tensor.flatten()  # Always 363 elements (11*11*3)
```

### 2. **Complete Object Encodings**
```python
self.cell_encoding = {
    'HEAD': [1, 0, 0, 0, 0, 0],        # Own head (6-channel encoding)
    'BODY': [0, 1, 0, 0, 0, 0],        # Own body
    'OTHER_HEAD': [0, 0, 1, 0, 0, 0],  # Other snake head  
    'OTHER_BODY': [0, 0, 0, 1, 0, 0],  # Other snake body
    'FOOD': [0, 0, 0, 0, 1, 0],        # Food item
    'EMPTY': [0, 0, 0, 0, 0, 1]        # Empty space
}
```

### 3. **Atomic State Updates**
```python
# Synchronize all state components in single operation
def get_synchronized_state(self):
    with self.comprehensive_lock:
        return {
            'vision': self.get_visible_cells(),
            'reward': self.reward,
            'episode': self.game_manager.episode_number,
            'frame': self.game_manager.frame_number,
            'game_over': self.game_manager.GAME_OVER
        }
```

The current state management system has fundamental flaws that prevent effective RL training. The variable tensor dimensions and loss of spatial structure make it impossible for the neural network to learn consistent patterns.

## State Visualization Tool

For debugging and analysis, use the included state visualization script that connects to the environment HTTP stream and displays real-time snake behavior:

### Jupyter Notebook Usage

```python
# Load the visualizer
%run docs/state_visualizer.py

# From list of states (your log storage)
states = [{'visible_cells': {...}, 'reward': 0.1, 'episode': 1, 'frame': 42}, ...]
visualizer = SnakeStateVisualizer(states)
anim = visualizer.animate()

# From JSON file
visualizer = SnakeStateVisualizer.from_file('experiment_states.json')
anim = visualizer.animate()

# Or use convenience functions
anim = visualize_states(states, interval=200)
anim = visualize_from_file('states.json', interval=200)
```

### Features

**Visualization Features:**
- **World View**: Shows snake position in environment (11×11 grid)
- **Vision View**: Displays what the snake can see (diamond pattern vision)  
- **Info Panel**: Episode, frame, reward, cell counts, and frame progress
- **Animation Controls**: Configurable speed, repeat, frame-by-frame navigation

**State Analysis:**
```python
# Analyze collected states
visualizer.analyze_states()

# Save states for later analysis
visualizer.save_states('experiment_states.json')

# Replay saved states
replay_anim = visualizer.replay_animation()
```

**Command Line Usage:**
```bash
python docs/state_visualizer.py http://localhost:5000/snake/adam
```

### Visualization Output

The tool creates a 3-panel display:

1. **Left Panel - World View**: 11×11 grid showing snake's actual position
2. **Middle Panel - Vision View**: Snake's partial observation (diamond pattern)
3. **Right Panel - State Info**: Real-time metrics and cell counts

**Color Legend:**
- 🔴 **Red**: Snake head (HEAD)  
- 🟢 **Green**: Snake body (BODY)
- 🔵 **Blue**: Other snake head (OTHER_HEAD)
- 🟦 **Cyan**: Other snake body (OTHER_BODY)
- 🟡 **Yellow**: Food (FOOD)
- ⬛ **Black**: Empty space (EMPTY)

### Debugging State Issues

Use this tool to identify the critical problems described above:

```python
# Monitor input tensor size variations from your logged states
states = load_your_states()  # Your log storage
visualizer = SnakeStateVisualizer(states)
analysis = visualizer.analyze_states()

# Example output showing the critical problems:
# 📊 State Analysis:
# Total states: 1250
# 🔍 Vision Analysis:
# Vision sizes: min=45, max=69, avg=58.3  ← PROBLEM: Variable sizes
# ⚠️ CRITICAL: Variable vision sizes detected! This will cause neural network crashes.
# 🚨 Issues Detected:
#   ❌ Variable vision sizes: 8 different sizes detected
#   ❌ No HEAD cells found: Agent cannot see its own position
```

This visualization tool helps debug state processing issues and verify that fixes maintain consistent tensor dimensions while preserving spatial information.

## ClickHouse Integration

### Log Format Analysis

The system logs state data to ClickHouse in two main formats:

**Environment Frame Events** (`services/Env/src/game/manager.py:150-156`):
```json
{
  "event": "frame",
  "grid": {"0,0": [{"type": "EMPTY"}], "5,5": [{"type": "HEAD", "snake_id": "adam"}]},
  "visions": {"adam": {"5,5": "HEAD", "4,5": "BODY", "6,5": "FOOD"}},
  "statuses": {"adam": false},
  "game_over": false,
  "episode": 5,
  "frame": 142
}
```

**Inference State Events** (`services/Inference/src/main.py:176-182`):
```json
{
  "event": "state_received",
  "visible_cells": {"5,5": "HEAD", "4,5": "BODY", "6,5": "FOOD"},
  "episode": 5,
  "frame": 142,
  "reward": 0.1,
  "game_over": false,
  "delay_s": "0.012"
}
```

### ClickHouse Queries for Visualization

**Extract from Environment logs** (most complete data):
```sql
SELECT 
    'adam' as snake_id,
    episode,
    frame, 
    JSONExtract(visions, 'adam') as visible_cells,
    0.1 as reward,  -- Default alive reward
    game_over,
    formatDateTime(timestamp, '%Y-%m-%dT%H:%M:%S.%f') as datetime
FROM logs_table 
WHERE event = 'frame' 
  AND JSONHas(visions, 'adam')
ORDER BY episode, frame;
```

**Extract from Inference logs** (includes actual rewards):
```sql
SELECT
    'adam' as snake_id,
    toInt32(JSONExtract(message, 'episode')) as episode,
    toInt32(JSONExtract(message, 'frame')) as frame,
    JSONExtract(message, 'visible_cells') as visible_cells,
    toFloat64(JSONExtract(message, 'reward')) as reward,
    toBool(JSONExtract(message, 'game_over')) as game_over,
    formatDateTime(timestamp, '%Y-%m-%dT%H:%M:%S.%f') as datetime
FROM logs_table
WHERE event = 'state_received'
ORDER BY episode, frame;
```

### Python Integration Example

```python
import clickhouse_connect

# Connect to ClickHouse
client = clickhouse_connect.get_client(host='your-host', port=8123)

# Query states
result = client.query("""
    SELECT snake_id, episode, frame, visible_cells, reward, game_over, datetime
    FROM (
        -- Use one of the queries above
    )
    ORDER BY episode, frame
    LIMIT 1000
""")

# Convert to visualizer format
states = []
for row in result.result_rows:
    snake_id, episode, frame, visible_cells, reward, game_over, datetime_str = row
    
    state = {
        'snake_id': snake_id,
        'episode': episode,
        'frame': frame,
        'visible_cells': visible_cells,  # Dict: {"5,5": "HEAD", ...}
        'reward': reward,
        'game_over': game_over,
        'datetime': datetime_str
    }
    states.append(state)

# Visualize and analyze
%run docs/state_visualizer.py
visualizer = SnakeStateVisualizer(states)
analysis = visualizer.analyze_states()  # Detects tensor size problems
visualizer.animate()
```

### Required Input Format

The visualizer expects a list of state dictionaries:

```python
states = [
    {
        "snake_id": "adam",
        "episode": 1,
        "frame": 1,
        "visible_cells": {
            "5,5": "HEAD",    # Coordinates as strings "x,y"
            "4,5": "BODY",    # Cell types as strings
            "6,5": "FOOD",
            "7,5": "EMPTY"
        },
        "reward": 0.1,
        "game_over": false,
        "datetime": "2025-01-15T14:30:15.123456"
    }
    # ... more frames in chronological order
]
```

**Key Requirements:**
- `visible_cells` must be dictionary with string keys: `"x,y"`
- Cell types: `"HEAD"`, `"BODY"`, `"FOOD"`, `"EMPTY"`, `"OTHER_HEAD"`, `"OTHER_BODY"`
- States ordered by episode and frame for smooth animation
- Each state represents one frame of agent observation

### Complete State Visualizer Script

**File**: [`docs/state_visualizer.py`](/app/docs/state_visualizer.py)

```python
#!/usr/bin/env python3
"""
Snake RL State Visualizer
Receives JSON state stream and creates matplotlib animation

Usage in Jupyter:
    %run state_visualizer.py
    visualizer = SnakeStateVisualizer('http://localhost:5000/snake/adam')
    visualizer.animate()
"""

import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import threading
from collections import deque
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')

class SnakeStateVisualizer:
    def __init__(self, stream_url, max_frames=1000):
        self.stream_url = stream_url
        self.max_frames = max_frames
        
        # State storage
        self.states = deque(maxlen=max_frames)
        self.running = False
        self.stream_thread = None
        self.current_state = None
        
        # Visualization setup
        self.grid_size = 11
        self.setup_colors()
        
        # Animation components
        self.fig = None
        self.ax_world = None
        self.ax_vision = None
        self.ax_info = None
        self.im_world = None
        self.im_vision = None
        
    def setup_colors(self):
        """Setup color mapping for different cell types"""
        self.color_map = {
            'EMPTY': 0,      # Black
            'HEAD': 1,       # Red
            'BODY': 2,       # Green
            'OTHER_HEAD': 3, # Blue  
            'OTHER_BODY': 4, # Cyan
            'FOOD': 5        # Yellow
        }
        
        # Create colormap
        colors = ['black', 'red', 'green', 'blue', 'cyan', 'yellow']
        self.cmap = ListedColormap(colors)
        
    def start_stream(self):
        """Start receiving state stream in background thread"""
        self.running = True
        self.stream_thread = threading.Thread(target=self._read_stream, daemon=True)
        self.stream_thread.start()
        print(f"Started stream from {self.stream_url}")
        
    def stop_stream(self):
        """Stop receiving state stream"""
        self.running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1)
        print("Stopped stream")
        
    def _read_stream(self):
        """Background thread to read HTTP state stream"""
        try:
            response = requests.get(self.stream_url, stream=True, timeout=10)
            for line in response.iter_lines():
                if not self.running:
                    break
                if line:
                    try:
                        data = json.loads(line.decode())
                        self.current_state = data
                        self.states.append(data)
                        
                        # Print state info
                        episode = data.get('episode', 'N/A')
                        frame = data.get('frame', 'N/A')
                        reward = data.get('reward', 'N/A')
                        game_over = data.get('game_over', False)
                        
                        print(f"\rEpisode: {episode:3}, Frame: {frame:3}, Reward: {reward:6.2f}, Game Over: {game_over}", end='')
                        
                        if game_over:
                            print(f"\n🔄 Episode {episode} ended!")
                            
                    except json.JSONDecodeError as e:
                        print(f"\nJSON decode error: {e}")
                        continue
                        
        except Exception as e:
            print(f"\nStream error: {e}")
            
    def parse_vision_to_grid(self, visible_cells):
        """Convert vision dictionary to 11x11 grid"""
        vision_grid = np.zeros((self.grid_size, self.grid_size))
        
        for coord_str, cell_type in visible_cells.items():
            try:
                x, y = map(int, coord_str.split(','))
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    vision_grid[y, x] = self.color_map.get(cell_type, 0)
            except (ValueError, AttributeError):
                continue
                
        return vision_grid
        
    def create_world_grid(self, state):
        """Create full world visualization from state"""
        # For now, just show vision data as proxy for world state
        visible_cells = state.get('visible_cells', {})
        return self.parse_vision_to_grid(visible_cells)
        
    def setup_plots(self):
        """Setup matplotlib figure and subplots"""
        self.fig = plt.figure(figsize=(15, 5))
        
        # World view (left)
        self.ax_world = self.fig.add_subplot(131)
        self.ax_world.set_title('Snake World View')
        self.ax_world.set_xlabel('X Position')
        self.ax_world.set_ylabel('Y Position')
        
        # Vision view (middle)  
        self.ax_vision = self.fig.add_subplot(132)
        self.ax_vision.set_title('Snake Vision (Processed)')
        self.ax_vision.set_xlabel('Vision X')
        self.ax_vision.set_ylabel('Vision Y')
        
        # Info panel (right)
        self.ax_info = self.fig.add_subplot(133)
        self.ax_info.set_title('State Information')
        self.ax_info.axis('off')
        
        # Initialize empty grids
        empty_grid = np.zeros((self.grid_size, self.grid_size))
        
        self.im_world = self.ax_world.imshow(empty_grid, cmap=self.cmap, 
                                           vmin=0, vmax=5, interpolation='nearest')
        self.im_vision = self.ax_vision.imshow(empty_grid, cmap=self.cmap,
                                             vmin=0, vmax=5, interpolation='nearest')
        
        # Add grid lines
        for ax in [self.ax_world, self.ax_vision]:
            ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_xlim(-0.5, self.grid_size-0.5)
            ax.set_ylim(-0.5, self.grid_size-0.5)
            
        # Add colorbar legend
        cbar = plt.colorbar(self.im_world, ax=[self.ax_world, self.ax_vision], 
                           ticks=list(range(6)), shrink=0.6)
        cbar.set_ticklabels(['Empty', 'Head', 'Body', 'Other Head', 'Other Body', 'Food'])
        
        plt.tight_layout()
        
    def update_plots(self, state):
        """Update plots with new state data"""
        if state is None:
            return
            
        visible_cells = state.get('visible_cells', {})
        
        # Update world and vision grids
        world_grid = self.create_world_grid(state)
        vision_grid = self.parse_vision_to_grid(visible_cells)
        
        self.im_world.set_array(world_grid)
        self.im_vision.set_array(vision_grid)
        
        # Update info panel
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        episode = state.get('episode', 'N/A')
        frame = state.get('frame', 'N/A') 
        reward = state.get('reward', 0)
        game_over = state.get('game_over', False)
        snake_id = state.get('snake_id', 'unknown')
        datetime_str = state.get('datetime', '')
        
        # Count cell types in vision
        cell_counts = {}
        for cell_type in visible_cells.values():
            cell_counts[cell_type] = cell_counts.get(cell_type, 0) + 1
            
        info_text = f"""Snake ID: {snake_id}
Episode: {episode}
Frame: {frame}
Reward: {reward:.3f}
Game Over: {game_over}

Vision Cells:"""
        
        for cell_type, count in sorted(cell_counts.items()):
            info_text += f"\n  {cell_type}: {count}"
            
        info_text += f"\n  Total: {len(visible_cells)}"
        
        if datetime_str:
            info_text += f"\n\nTimestamp:\n{datetime_str[:19]}"
            
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace')
        
    def animate_frame(self, frame_num):
        """Animation function for matplotlib"""
        if self.current_state:
            self.update_plots(self.current_state)
        return [self.im_world, self.im_vision]
        
    def animate(self, interval=100):
        """Start real-time animation"""
        print("Setting up visualization...")
        
        # Start stream
        self.start_stream()
        
        # Wait for first state
        wait_time = 0
        while self.current_state is None and wait_time < 10:
            time.sleep(0.1)
            wait_time += 0.1
            
        if self.current_state is None:
            print("❌ No state received. Check if environment is running.")
            return
            
        print("✅ Receiving states, starting animation...")
        
        # Setup plots
        self.setup_plots()
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.animate_frame, interval=interval, 
            blit=False, cache_frame_data=False
        )
        
        plt.show()
        
        return anim
        
    def save_states(self, filename='snake_states.json'):
        """Save collected states to file"""
        states_list = list(self.states)
        with open(filename, 'w') as f:
            json.dump(states_list, f, indent=2)
        print(f"Saved {len(states_list)} states to {filename}")
        
    def load_states(self, filename='snake_states.json'):
        """Load states from file"""
        try:
            with open(filename, 'r') as f:
                states_list = json.load(f)
            self.states.extend(states_list)
            print(f"Loaded {len(states_list)} states from {filename}")
            return states_list
        except FileNotFoundError:
            print(f"File {filename} not found")
            return []
            
    def replay_animation(self, states=None, interval=200):
        """Replay animation from saved states"""
        if states is None:
            states = list(self.states)
            
        if not states:
            print("No states to replay")
            return
            
        print(f"Replaying {len(states)} states...")
        
        self.setup_plots()
        
        def replay_frame(frame_num):
            if frame_num < len(states):
                self.update_plots(states[frame_num])
            return [self.im_world, self.im_vision]
            
        anim = animation.FuncAnimation(
            self.fig, replay_frame, frames=len(states),
            interval=interval, blit=False, repeat=True
        )
        
        plt.show()
        return anim
        
    def analyze_states(self):
        """Analyze collected states and show statistics"""
        if not self.states:
            print("No states to analyze")
            return
            
        states_list = list(self.states)
        
        # Basic statistics
        episodes = [s.get('episode') for s in states_list if s.get('episode') is not None]
        frames = [s.get('frame') for s in states_list if s.get('frame') is not None]
        rewards = [s.get('reward') for s in states_list if s.get('reward') is not None]
        
        print(f"📊 State Analysis:")
        print(f"Total states: {len(states_list)}")
        
        if episodes:
            print(f"Episodes: {min(episodes)} - {max(episodes)}")
        if frames:
            print(f"Frames: {min(frames)} - {max(frames)}")
        if rewards:
            print(f"Rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, avg={np.mean(rewards):.3f}")
            
        # Vision statistics
        vision_sizes = []
        cell_type_counts = {}
        
        for state in states_list:
            visible_cells = state.get('visible_cells', {})
            vision_sizes.append(len(visible_cells))
            
            for cell_type in visible_cells.values():
                cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
                
        if vision_sizes:
            print(f"\nVision sizes: min={min(vision_sizes)}, max={max(vision_sizes)}, avg={np.mean(vision_sizes):.1f}")
            
        if cell_type_counts:
            print("\nCell type frequencies:")
            for cell_type, count in sorted(cell_type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cell_type}: {count}")
                
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_stream()


# Convenience functions for Jupyter
def quick_visualize(stream_url='http://localhost:5000/snake/adam'):
    """Quick setup and start visualization"""
    visualizer = SnakeStateVisualizer(stream_url)
    return visualizer.animate()

def replay_from_file(filename='snake_states.json'):
    """Load and replay states from file"""
    visualizer = SnakeStateVisualizer('dummy')  # No stream needed
    states = visualizer.load_states(filename)
    if states:
        return visualizer.replay_animation(states)
    return None

if __name__ == "__main__":
    # Command line usage
    import sys
    
    stream_url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:5000/snake/adam'
    
    print(f"Starting Snake RL State Visualizer")
    print(f"Stream URL: {stream_url}")
    print("Press Ctrl+C to stop")
    
    visualizer = SnakeStateVisualizer(stream_url)
    
    try:
        anim = visualizer.animate()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping visualizer...")
        visualizer.stop_stream()
        visualizer.analyze_states()
        visualizer.save_states()
        print("✅ Done!")
```

This tool is essential for debugging state processing issues and validating that any fixes maintain proper state representation while preserving spatial information for effective RL training.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Analyze state storage and processing in both Environment and Inference", "status": "completed", "activeForm": "Analyzing state storage and processing in both Environment and Inference"}, {"content": "Write comprehensive state.md documentation", "status": "completed", "activeForm": "Writing comprehensive state.md documentation"}]