# Frame-by-Frame Execution Flow: Environment ↔ Inference

This document provides a detailed step-by-step analysis of what happens during each frame in the Snake RL system, showing the interaction between the Environment service and Inference service.

## System Architecture Overview

The system consists of two main services:
- **Environment Service**: Flask-based game server (`services/Env/`)
- **Inference Service**: Neural network agent (`services/Inference/`)

Communication happens via HTTP streaming and REST APIs.

## Frame Execution Cycle

### 1. Environment Service Initialization

**File**: [`services/Env/src/app.py:24-35`](/app/services/Env/src/app.py#L24-L35)

```python
game_manager = GameManager(
    grid_width=args.grid_width,        # 11x11 grid
    grid_height=args.grid_height,
    vision_radius=args.vision_radius,  # 5 cell radius
    fps=args.fps,                      # 100 FPS
    seed=args.seed,
    reward_config=reward_config,       # {"alive": 0.1, "eat_food": 1.0, "game_over": -1.0}
    maxStepsWithoutApple=args.max_steps_without_food,  # Starvation limit
    max_snakes=args.max_snakes
)
```

**Background Game Loop Starts**: [`services/Env/src/game/manager.py:44`](/app/services/Env/src/game/manager.py#L44)

```python
threading.Thread(target=self.game_loop, daemon=True).start()
```

### 2. Inference Service Connection

**File**: [`services/Inference/src/main.py:109-143`](/app/services/Inference/src/main.py#L109-L143)

The inference service connects to the environment via HTTP streaming:

```python
base_url = f"http://{env_host}/snake/{snake_id}"  # GET stream endpoint
move_url = f"{base_url}/move"                     # POST action endpoint
```

**Stream Reader Setup**: [`services/Inference/src/main.py:143-144`](/app/services/Inference/src/main.py#L143-L144)

```python
stream_reader = StreamReader(base_url)
stream_reader.start()  # Background thread reads HTTP stream
```

## Frame-by-Frame Execution

### Environment Side (Every 1/100 second at 100 FPS)

#### Frame N: Game Loop Tick

**File**: [`services/Env/src/game/manager.py:133-173`](/app/services/Env/src/game/manager.py#L133-L173)

**Step 1**: Sleep for frame timing
```python
time.sleep(1.0 / self.FPS)  # 0.01 seconds (100 FPS)
self.frame_number += 1
```

**Step 2**: Update all snakes
```python
for sid, game in list(self.snakes.items()):
    with self.snake_locks[sid]:
        status = game.update(self.GAME_OVER)  # Move snake, check collisions
```

#### Snake Update Logic

**File**: [`services/Env/src/game/snake.py:87-124`](/app/services/Env/src/game/snake.py#L87-L124)

**Step 1**: Check game over conditions
```python
if game_over:
    self.reward += self.reward_config['game_over']  # -1.0 penalty
    return ''

if self.stepsSinceLastApple >= self.maxStepsWithoutApple:
    return "starvation"  # Death by starvation
```

**Step 2**: Calculate new snake head position
```python
head = self.snake[0]
new_head = ((head[0] + self.direction[0]) % self.grid_width,
            (head[1] + self.direction[1]) % self.grid_height)
```

**Step 3**: Check collision with any snake body
```python
occupied = {pos for game in self.snakes.values() for pos in game.snake}
if new_head in occupied:
    return 'collision'  # Death by collision
```

**Step 4**: Move snake and handle food
```python
self.snake.insert(0, new_head)  # Add new head
if new_head in self.foods:
    self.foods.remove(new_head)
    self.reward += self.reward_config['eat_food']  # +1.0 reward
    self.stepsSinceLastApple = 0
else:
    self.snake.pop()  # Remove tail (snake doesn't grow)
    self.stepsSinceLastApple += 1

self.reward += self.reward_config['alive']  # +0.1 per step alive
```

**Step 5**: Update visible state for streaming
```python
self._update_visible_state()  # Calculate vision grid
```

#### Vision Calculation

**File**: [`services/Env/src/game/snake.py:126-167`](/app/services/Env/src/game/snake.py#L126-L167)

```python
def _calc_visible_cells(self):
    head = self.snake[0]
    # Create diamond-shaped vision (Manhattan distance ≤ 5)
    for dx in range(-self.vision_radius, self.vision_radius + 1):
        for dy in range(-self.vision_radius, self.vision_radius + 1):
            if abs(dx) + abs(dy) > self.vision_radius:
                continue  # Skip cells outside vision diamond
            
            # Rotate coordinates based on snake direction
            rx, ry = rotate(dx, dy)  # Relative to snake's heading
            # Map to 11x11 vision display grid
            cx = self.vision_display_cols // 2 + rx  
            cy = self.vision_display_rows // 2 + ry
            
            # Get world position (with wrapping)
            px = (head[0] + dx) % self.grid_width
            py = (head[1] + dy) % self.grid_height
            
            # Classify cell content
            if pos == head:
                obj = 'HEAD'
            elif pos in self.snake[1:]:
                obj = 'BODY' 
            elif pos in other_heads:
                obj = 'OTHER_HEAD'
            elif pos in other_bodies:
                obj = 'OTHER_BODY'
            elif pos in self.foods:
                obj = 'FOOD'
            else:
                obj = 'EMPTY'
            
            vis[f"{cx},{cy}"] = obj
```

#### Frame Logging

**File**: [`services/Env/src/game/manager.py:148-156`](/app/services/Env/src/game/manager.py#L148-L156)

```python
if self.GAME_OVER != True:
    grid, visions, statuses, game_over = self.state()
    logging.info({
        "event": "frame",
        "grid": grid,           # Full 11x11 world state
        "visions": visions,     # Each snake's visible cells
        "statuses": statuses,   # Game over status per snake
        "game_over": game_over,
        "episode": self.episode_number,
        "frame": self.frame_number
    })
```

### HTTP Stream Response

**File**: [`services/Env/src/routes/snake.py:20-40`](/app/services/Env/src/routes/snake.py#L20-L40)

When inference service connects to `/snake/{sid}`, it gets a streaming response:

```python
def stream_vision(sid):
    def gen():
        while True:
            with game_manager.snake_locks[sid]:
                vis = game_manager.snakes[sid].get_visible_cells()
                reward = game_manager.snakes[sid].reward
            
            payload = {
                'snake_id': sid,
                'visible_cells': vis,    # Vision grid (up to 61 cells)
                'reward': reward,        # Current frame reward
                'game_over': is_game_over,
                'episode': game_manager.episode_number,
                'frame': game_manager.frame_number,
                'datetime': datetime.now().isoformat()
            }
            yield json.dumps(payload) + '\n'  # NDJSON streaming
            
            time.sleep(1.0 / game_manager.FPS)  # Match game FPS
```

### Inference Side (Reactive to Stream)

#### Frame N: Stream Data Reception

**File**: [`services/Inference/src/main.py:65-89`](/app/services/Inference/src/main.py#L65-L89)

**Step 1**: Background thread receives HTTP stream
```python
def _read_stream(self):
    response = requests.get(self.base_url, stream=True, timeout=10)
    for line in response.iter_lines():
        data = json.loads(line.decode())
        with self.lock:
            self.latest_state = data
        self.new_state_event.set()  # Signal main thread
```

**Step 2**: Main thread processes new state
```python
data, tech_timestamp = stream_reader.get_latest_state()
episode_count = data.get("episode")
frame_count = data.get("frame") 
visible_cells = data.get("visible_cells")
reward = data.get("reward")
game_over = data.get("game_over")
```

#### State Processing

**File**: [`services/Inference/src/state_processor.py:17-73`](/app/services/Inference/src/state_processor.py#L17-L73)

**Step 1**: Filter out snake HEAD from vision
```python
filtered_cells = {k: v for k, v in visible_cells.items() if v != 'HEAD'}
```

**Step 2**: Convert cell types to neural network encoding
```python
self.cell_encoding = {
    'FOOD': [0, 0, 1],      # Food channel
    'BODY': [0, 1, 0],      # Own body channel  
    'OTHER_BODY': [1, 0, 0], # Other snake channel
    'EMPTY': [0, 0, 0]      # Empty space
}
```

**Step 3**: Create sorted coordinate tensor
```python
sorted_cells = []
for coord_str, cell_type in filtered_cells.items():
    x, y = map(int, coord_str.split(','))
    sorted_cells.append((x, y, cell_type))

sorted_cells.sort(key=lambda item: (item[0], item[1]))  # Sort by x,y

tensor_data = []
for x, y, cell_type in sorted_cells:
    encoding = self.cell_encoding.get(cell_type, [0, 0, 0])
    tensor_data.append(encoding)

result = torch.tensor(tensor_data, dtype=torch.float32).flatten()
```

#### Neural Network Inference

**File**: [`services/Inference/src/snake_agent.py:92-111`](/app/services/Inference/src/snake_agent.py#L92-L111)

**Step 1**: Process state to tensor
```python
state_tensor = self.state_processor.process_state(state)
flat_tensor = state_tensor.flatten().unsqueeze(0)  # Add batch dimension
input_size = flat_tensor.shape[1]  # Dynamic input size
```

**Step 2**: Forward pass through neural network
```python
with torch.no_grad():
    self.model.eval()
    action_probs = self.model(flat_tensor)  # [1, 3] output
```

**Neural Network Architecture**: [`services/Inference/src/snake_model.py:16-30`](/app/services/Inference/src/snake_model.py#L16-L30)

```python
self.network = nn.Sequential(
    nn.Linear(input_size, hidden_units_1),  # Default: 15 neurons
    nn.Tanh(),
    nn.Dropout(dropout_rate),               # Default: 0.6
    nn.Linear(hidden_units_1, hidden_units_2), # Default: 15 neurons  
    nn.Tanh(),
    nn.Dropout(dropout_rate),
    nn.Linear(hidden_units_2, 3),           # 3 actions: left, right, forward
    nn.Softmax(dim=1)                       # Probability distribution
)
```

**Step 3**: Sample action from probability distribution
```python
action_dist = torch.distributions.Categorical(action_probs)
action_idx = action_dist.sample().item()  # Sample with entropy
predicted_action = self.actions[action_idx]  # ["left", "right", "forward"]
```

#### Action Execution

**File**: [`services/Inference/src/main.py:219-222`](/app/services/Inference/src/main.py#L219-L222)

```python
action = agent.predict_action(data)
if action != "forward":  # Only send non-default moves
    send_move(move_url, action)
previous_action = action
```

**HTTP POST to Environment**: [`services/Inference/src/main.py:90-99`](/app/services/Inference/src/main.py#L90-L99)

```python
def send_move(move_url, move: str):
    payload = {"move": move}  # "left" or "right"
    response = requests.post(move_url, json=payload)
```

### Environment Action Processing

**File**: [`services/Env/src/routes/snake.py:42-56`](/app/services/Env/src/routes/snake.py#L42-L56)

```python
@snake_bp.route('/snake/<sid>/move', methods=['POST'])
def move_snake(sid):
    data = request.get_json(force=True)
    cmd = data.get('move')  # "left" or "right"
    
    with game_manager.snake_locks[sid]:
        game_manager.snakes[sid].turn(cmd)  # Update snake direction
```

**Direction Update**: [`services/Env/src/game/snake.py:47-55`](/app/services/Env/src/game/snake.py#L47-L55)

```python
def relative_turn(self, cmd):
    if cmd == 'left':
        return (self.direction[1], -self.direction[0])  # 90° left
    if cmd == 'right':
        return (-self.direction[1], self.direction[0])  # 90° right
    return self.direction  # forward (no change)
```

### Episode End Handling

#### Environment Side

**Game Over Detection**: [`services/Env/src/game/manager.py:143-147`](/app/services/Env/src/game/manager.py#L143-L147)

```python
status = game.update(self.GAME_OVER)
if status == 'collision' or status == 'starvation':
    self.GAME_OVER = True
    logging.info({"event": "game_over", "reason": status, "snake_id": sid})
```

**Final Results Logging**: [`services/Env/src/game/manager.py:157-169`](/app/services/Env/src/game/manager.py#L157-L169)

```python
elif self.game_over_raised == False:
    snake_lens = {}
    for sid, game in list(self.snakes.items()):
        snake_len = len(game.snake)
        snake_lens[sid] = snake_len
    
    logging.info({
        "event": "game_over_results", 
        "snakes_lengths": snake_lens,
        "episode": self.episode_number,
        "frames": self.frame_number
    })
    self.game_over_raised = True
```

#### Inference Side

**Experience Collection**: [`services/Inference/src/main.py:186-191`](/app/services/Inference/src/main.py#L186-L191)

```python
should_send_batch = agent.add_experience(
    state=data,
    action=previous_action,
    reward=reward,
    done=game_over
)
```

**Episode Completion**: [`services/Inference/src/main.py:192-209`](/app/services/Inference/src/main.py#L192-L209)

```python
if game_over == True:
    episode_counter += 1
    if should_send_batch:  # Batch size reached (default: 5 episodes)
        success = agent.send_training_batch_and_wait()
        if success:
            logging.info({"event": "received_improved_model"})
```

**Environment Reset**: [`services/Inference/src/main.py:210-217`](/app/services/Inference/src/main.py#L210-L217)

```python
reset_response = requests.post(reset_url, timeout=5)
```

### Reset Cycle

**File**: [`services/Env/src/game/manager.py:90-111`](/app/services/Env/src/game/manager.py#L90-L111)

```python
def reset_game(self):
    with self.game_over_lock:
        self.GAME_OVER = True   # Stop current game
    time.sleep(0.1)             # Brief pause
    with self.game_over_lock:
        self.GAME_OVER = False  # Re-enable game
    
    self.snakes.clear()         # Remove all snakes
    self.FOODS.clear()          # Clear food
    self.spawn_food()           # Spawn new food
    self.episode_number += 1    # Increment episode
    self.frame_number = 0       # Reset frame counter
    
    # Check episode limit
    if self.episode_number >= self.max_episodes:
        logging.info({"event": "env_max_episodes_completed"})
        sys.exit(0)  # Terminate environment
```

## Key Performance Characteristics

### Timing
- **Environment FPS**: 100 Hz (10ms per frame)
- **HTTP Stream**: Real-time (10ms latency typical)
- **Neural Inference**: <1ms per prediction
- **Action Response**: <5ms total loop

### Data Flow
- **Vision Input**: Up to 61 cells (5-radius diamond)
- **State Encoding**: 3-channel (food/body/other) × N cells
- **Neural Network**: Dynamic input → 15→15→3 architecture  
- **Actions**: 3 discrete outputs (left/right/forward)
- **Rewards**: Sparse (+1.0 food, +0.1 alive, -1.0 death)

### Memory Management
- **Episode Batching**: 5 episodes before training
- **Thread Safety**: Locks protect shared game state
- **Stream Buffering**: Single latest state (no queuing)

This frame-by-frame flow repeats continuously until episode limits are reached or manual termination occurs.