# Shared Memory Architecture

> **Current Implementation**: This documents the active shared memory architecture used in production.

## System Specification

The system consists of three processes: **Clock**, **ENV**, and **INF**.

The **Clock** orchestrates execution by controlling when ENV and INF are allowed to run.

### Execution Order (Strictly Sequential)

1. **Clock triggers ENV** → ENV is allowed to execute one step
2. **ENV produces a state** and signals readiness to Clock
3. **Clock waits for ENV**, then **triggers INF** → INF is allowed to execute one step  
4. **INF consumes ENV's state**, produces an action, and signals readiness to Clock
5. **Clock waits for INF**, then returns to step 1

### Key Constraints
- **Only one of ENV or INF is active at a time** - no concurrent execution
- **Shared memory** is used for exchanging state and action data
- **Synchronization** is done with events/flags; Clock ensures no race conditions occur
- **Sequential control** eliminates timing issues and ensures deterministic execution

## Three-Service Architecture

### Service Breakdown
1. **Clock Service** (renamed from Sync) - Orchestrates sequential execution, manages synchronization
2. **Environment Service** - Pure game logic, waits for Clock trigger
3. **Inference Service** - Pure neural network logic, waits for Clock trigger

## Clock Service (Sequential Orchestrator)

### Primary Responsibilities
- **Shared Memory Creation:** Creates and manages "snake_state" and "snake_action" blocks
- **Sequential Execution Control:** Ensures only ENV or INF runs at any given time
- **Event-Based Synchronization:** Uses events/flags to trigger and wait for services
- **Race Condition Prevention:** Guarantees deterministic execution order
- **Cleanup Management:** Handles shared memory cleanup on shutdown

### Clock Service Logic (Sequential Control)
```
Initialization:
├── Create shared memory blocks (snake_state, snake_action)
├── Create synchronization events (env_trigger, inf_trigger, env_ready, inf_ready)
├── Wait for ENV and INF services to connect and signal ready
└── Begin sequential orchestration loop

Main Sequential Loop:
├── 1. Set env_trigger event → Signal ENV to execute one step
├── 2. Wait for env_ready event → ENV signals completion
├── 3. Clear env_trigger, clear env_ready events
├── 4. Set inf_trigger event → Signal INF to execute one step  
├── 5. Wait for inf_ready event → INF signals completion
├── 6. Clear inf_trigger, clear inf_ready events
└── 7. Return to step 1 (next cycle)

Shutdown Handling:
├── Set shutdown events for both ENV and INF
├── Wait for graceful service termination
└── Clean up shared memory and events
```

## Environment Service (Clock-Controlled)

### Responsibilities
- **Pure Game Logic:** Snake movement, collision detection, food spawning
- **State Publishing:** Write game state to shared memory
- **Action Consumption:** Read actions from shared memory
- **Clock Synchronization:** Wait for Clock trigger before each step

### Environment Logic (Clock-Controlled)
```
Initialization:
├── Connect to existing shared memory blocks
├── Initialize GameManager with snake
├── Signal readiness to Clock service
└── Enter wait state for first trigger

Main Game Loop (Clock-Triggered):
├── Wait for env_trigger event from Clock
├── When triggered:
│   ├── Read action from shared memory (from previous INF step)
│   ├── Process action → set turn command in GameManager  
│   ├── Execute single game step → advance game state
│   ├── Calculate vision vector → rhomb-to-vector conversion
│   ├── Write complete state to shared memory
│   └── Set env_ready event → signal Clock that step is complete
└── Return to wait state until next trigger

Shutdown Handling:
├── Wait for shutdown event from Clock
├── Perform cleanup operations
└── Exit gracefully
```

## Inference Service (Clock-Controlled)

### Responsibilities  
- **Pure ML Logic:** Neural network inference and training
- **State Consumption:** Read game states from shared memory
- **Action Publishing:** Write actions to shared memory
- **Clock Synchronization:** Wait for Clock trigger before each step

### Inference Logic (Clock-Controlled)
```
Initialization:
├── Connect to existing shared memory blocks  
├── Initialize SnakeAgent with neural network
├── Signal readiness to Clock service
└── Enter wait state for first trigger

Episode Loop:
For each episode:
  └── Game Step Loop (Clock-Triggered):
      ├── Wait for inf_trigger event from Clock
      ├── When triggered:
      │   ├── Read state from shared memory (from latest ENV step)
      │   ├── Parse binary state → extract vision, reward, game_over, etc.
      │   ├── Process state through neural network
      │   ├── Generate action prediction (0=forward, 1=left, 2=right)
      │   ├── Write action to shared memory
      │   ├── Add experience to training batch
      │   └── Set inf_ready event → signal Clock that step is complete
      ├── Return to wait state until next trigger
      ├── Handle training batch logic (when batch full)
      └── If game_over: handle episode end and continue

Shutdown Handling:
├── Wait for shutdown event from Clock
├── Save any remaining training data
├── Perform cleanup operations
└── Exit gracefully
```

## Event-Based Synchronization Protocol

### Sequential Event Flow
1. **Clock → ENV:** `env_trigger` event signals ENV to execute one step
2. **ENV → Clock:** `env_ready` event signals ENV step completion  
3. **Clock → INF:** `inf_trigger` event signals INF to execute one step
4. **INF → Clock:** `inf_ready` event signals INF step completion
5. **Return to step 1** for next cycle

### Event Lifecycle Management
- **Only Clock sets trigger events** (`env_trigger`, `inf_trigger`)
- **Only services set ready events** (`env_ready`, `inf_ready`)
- **Clock clears all events** after each step to prevent race conditions
- **Blocking waits:** Clock blocks until ready events are received
- **No concurrent execution:** ENV and INF never run simultaneously

## Docker Compose Configuration

```yaml
services:
  clock:
    image: localhost:5000/snake-rl/clock:latest
    volumes:
      - /dev/shm:/dev/shm
      - ./shared-volume:/output
    command: ["python", "clock_service.py"]
    
  env:
    image: localhost:5000/snake-rl/env:latest
    volumes:
      - /dev/shm:/dev/shm
      - ./shared-volume:/output
    depends_on:
      - clock
    command: ["python", "main.py", "--grid_width", "15", ...]
    
  inference:
    image: localhost:5000/snake-rl/inference:latest
    volumes:
      - /dev/shm:/dev/shm
      - ./shared-volume:/output
    depends_on:
      - clock
    command: ["python", "main.py", "--snake-id", "adam", ...]
```

## Service Communication Flow

### Startup Sequence
1. **Clock Service:** Creates shared memory blocks and synchronization events, waits for connections
2. **Environment Service:** Connects to shared memory, signals readiness to Clock
3. **Inference Service:** Connects to shared memory, signals readiness to Clock  
4. **Clock Service:** Begins sequential orchestration loop

### Runtime Coordination (Sequential)
1. **Clock** sets `env_trigger` → **ENV** executes one step
2. **ENV** writes state to shared memory → sets `env_ready` → waits
3. **Clock** receives `env_ready` → sets `inf_trigger` → **INF** executes one step
4. **INF** reads state, writes action → sets `inf_ready` → waits
5. **Clock** receives `inf_ready` → returns to step 1

### Failure Handling
- **Service disconnection:** Clock detects missing ready events and waits for reconnection
- **Timeout handling:** Clock can timeout on missing ready events and handle gracefully
- **Shared memory cleanup:** Clock handles cleanup on any service failure
- **Deterministic restart:** Sequential control ensures consistent state on restart

## Clock Service Implementation Structure

### Core Components
- **SharedMemoryManager:** Create and manage memory blocks
- **EventManager:** Create and manage synchronization events (`env_trigger`, `inf_trigger`, `env_ready`, `inf_ready`)
- **SequentialOrchestrator:** Implement strict sequential execution control
- **ServiceMonitor:** Track ENV and INF service connection states
- **TimeoutHandler:** Handle missing ready events and service failures
- **CleanupManager:** Handle graceful shutdown and resource cleanup

### Sequential Control Benefits
- **Deterministic execution:** No race conditions or timing issues
- **Simplified debugging:** Predictable execution order
- **Performance isolation:** Each service gets full CPU when active
- **State consistency:** No concurrent access to shared data
- **Easy testing:** Sequential execution is easier to test and validate

This architecture provides strict sequential control while eliminating race conditions and ensuring deterministic execution order.

## SharedMemoryManager Design

### Core Functionality
- **Named shared memory blocks:** "snake_state" and "snake_action"
- **NumPy array views:** Direct access to shared memory as numpy arrays
- **Binary serialization:** Pack/unpack GameState and ActionCommand structures
- **File-based signaling:** sync_inf and sync_env files for coordination
- **Exception handling:** Robust error handling for memory operations
- **Cleanup management:** Proper resource cleanup on exit

### Data Structures
- **GameState:** Variable size (vision_vector + 33 bytes metadata)
- **ActionCommand:** 13 bytes fixed (move + timestamp + sequence)
- **Binary format:** struct.pack/unpack for serialization
- **NumPy integration:** Direct array views for vision processing

## Universal Contract with Vector-Based Visible State

### Rhomb-to-Vector Conversion Strategy
- **Spiral Ordering Pattern:** Center → Ring 1 → Ring 2 → ... → Ring 5
- **Manhattan Distance:** `abs(dx) + abs(dy) <= vision_radius` for rhomb boundary
- **Fixed Size:** ~61 cells for radius=5
- **Consistent Ordering:** Sorted by angle within each ring

### Byte Encoding Scheme
```
CELL_ENCODING = {
    'EMPTY': 0,      # Most common - compressible
    'FOOD': 1,       # High priority 
    'BODY': 2,       # Own body collision
    'HEAD': 3,       # Center reference (always index 0)
    'OTHER_BODY': 4, # Multi-agent collision
    'OTHER_HEAD': 5  # Multi-agent strategy
}
```

### Contract Structure
- **GameState:** size_visible_area + vision_vector + reward + game_over + episode + frame + env_control
- **ActionCommand:** move + timestamp + sequence
- **Variable Size:** 95 bytes (radius=5) vs ~500 bytes JSON (5.3x reduction)

## Binary State Format Specification

### Sequential Byte Array Structure
```
[size_visible_area][visible_area_data][reward][game_over][episode][frame][env_control]
     8 bytes         variable length    8 bytes   1 byte   8 bytes 8 bytes   1 byte
```

### Field Definitions

#### 1. size_visible_area (8 bytes, int64)
- **Range:** 0 to 9,223,372,036,854,775,807 cells
- **Current usage:** ~61 cells for radius=5
- **Purpose:** Tells parser how many vision cells follow
- **Flexibility:** Supports massive vision areas for future extensions

#### 2. visible_area_data (variable length)
- **Length:** Determined by size_visible_area
- **Cell encoding:** Each byte represents one cell (0-5 values)
- **Order:** Spiral pattern from center outward
- **Encoding:** 0=empty, 1=food, 2=body, 3=head, 4=other_body, 5=other_head

#### 3. reward (8 bytes, float64)
- **Format:** IEEE 754 double precision
- **Purpose:** High-precision reward values for advanced RL algorithms

#### 4. game_over (1 byte, boolean)
- **Values:** 0=continue, 1=game_over
- **Purpose:** Episode termination signal

#### 5. episode (8 bytes, int64)
- **Purpose:** Episode counter for training tracking

#### 6. frame (8 bytes, int64)
- **Purpose:** Frame counter within episode

#### 7. env_control (1 byte, uint8)
- **Range:** 0-255 action indices
- **Current mapping:**
  - `0` = forward (no turn)
  - `1` = left turn
  - `2` = right turn
  - `3-255` = reserved for future actions
- **Purpose:** Environment expects this action index to be executed

### State Size Examples
```
Radius=5: [8][61][8][1][8][8][1] = 95 bytes total
Radius=3: [8][37][8][1][8][8][1] = 71 bytes total
Radius=7: [8][113][8][1][8][8][1] = 147 bytes total
```

### Action Index Mapping
```python
ACTION_MAPPING = {
    0: "forward",    # Continue in current direction
    1: "left",       # Turn left relative to current direction  
    2: "right",      # Turn right relative to current direction
    # 3-255: Reserved for future actions
}
```

### Parsing Implementation
```python
def parse_state(data: bytes) -> GameState:
    offset = 0
    size_visible = struct.unpack('q', data[offset:offset + 8])[0]; offset += 8
    vision_data = data[offset:offset + size_visible]; offset += size_visible
    reward = struct.unpack('d', data[offset:offset + 8])[0]; offset += 8
    game_over = bool(data[offset]); offset += 1
    episode = struct.unpack('q', data[offset:offset + 8])[0]; offset += 8
    frame = struct.unpack('q', data[offset:offset + 8])[0]; offset += 8
    env_control = data[offset]
    return GameState(vision_data, reward, game_over, episode, frame, env_control)

def serialize_state(vision_vector, reward, game_over, episode, frame, env_control) -> bytes:
    return (struct.pack('q', len(vision_vector)) + vision_vector.tobytes() + 
            struct.pack('d', reward) + bytes([int(game_over)]) +
            struct.pack('q', episode) + struct.pack('q', frame) + bytes([env_control]))
```

---

## Current Service Parameters

### Environment Service Parameters
From `/app/services/Env/src/config.py` and `/app/k8s/snake-rl/values.yaml`:

```python
# Game Grid Configuration
--grid_width: int = 15           # Game grid width
--grid_height: int = 15          # Game grid height

# Vision System Configuration  
--vision_radius: int = 5         # Vision radius (Manhattan distance)
--vision_display_cols: int = 11  # Vision display columns
--vision_display_rows: int = 11  # Vision display rows

# Game Mechanics Configuration
--fps: int = 100000              # Frames per second (game speed)
--seed: int = 1                  # Random seed for reproducibility
--max_steps_without_food: int = 1000  # Max steps before game over
--max_snakes: int = 1            # Maximum number of snakes

# Reward Configuration
--reward_config: str = '{"alive":1, "eat_food":1, "game_over":0}'  # JSON reward structure

# Environment Variables
LOG_LEVEL: str = "DEBUG"         # Logging level
ENABLE_CONSOLE_LOGS: bool = true # Enable console logging
RABBITMQ_ENABLED: bool = false   # Disable RabbitMQ for shared memory setup
EXPERIMENT_NAME: str = "shared_memory_test"  # Experiment identifier
```

### Inference Service Parameters
From `/app/services/Inference/src/main.py` and `/app/k8s/snake-rl/values.yaml`:

```python
# Agent Configuration
--snake_id: str = "adam"         # Snake identifier
--env_host: str = "env:5000"     # Environment host (not used in shared memory)
--model_dir: str = "/output/models"  # Model save directory
--log_file: str = "/output/agent1.log"  # Log file path

# Training Parameters
--batch_size: int = 1            # Episodes per training batch
--max_episodes: int = 1000       # Maximum episodes before exit
--learning_rate: float = 0.009   # Neural network learning rate
--gamma: float = 0.6             # Discount factor for reinforcement learning
--beta: float = 0.16             # Entropy bonus coefficient

# Early Stopping Configuration
STEPS_WITHOUT_IMPROVEMENT_LIMIT: int = 10000  # Steps before early stopping

# Environment Variables
LOG_LEVEL: str = "DEBUG"         # Logging level
ENABLE_CONSOLE_LOGS: bool = true # Enable console logging
RABBITMQ_ENABLED: bool = false   # Disable RabbitMQ for shared memory setup
EXPERIMENT_NAME: str = "shared_memory_test"  # Experiment identifier
N_EPISODES: int = None           # Alternative max episodes via env var
```

### Sync Service Parameters (New)

```python
# Shared Memory Configuration
--state_memory_size: int = 128   # State shared memory block size (bytes)
--action_memory_size: int = 32   # Action shared memory block size (bytes)
--sync_dir: str = "/dev/shm/snake_rl"  # Sync file directory

# Coordination Configuration  
--heartbeat_interval: float = 0.001  # Service heartbeat check interval (seconds)
--service_timeout: float = 10.0      # Service connection timeout (seconds)
--cleanup_on_exit: bool = true       # Clean up shared memory on exit

# Monitoring Configuration
--log_coordination: bool = true      # Log coordination events
--log_level: str = "DEBUG"          # Logging level
```

This architecture separates concerns cleanly - each service focuses on its core responsibility while the sync service handles all coordination complexity.