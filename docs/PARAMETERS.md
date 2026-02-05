# RL Training and Environment Parameters

This document describes all parameters that affect reinforcement learning training and environment setup in the Snake RL project.

## 🏗️ Environment Parameters

| Parameter | Type | Description | Impact on Training |
|-----------|------|-------------|-------------------|
| **gridWidth** | int | Game grid width in cells | Larger = more complex state space |
| **gridHeight** | int | Game grid height in cells | Larger = more exploration needed |
| **visionRadius** | int | Snake's vision distance | Larger = more observable info |
| **maxStepsWithoutFood** | int | Steps before starvation death | Higher = longer episodes, less pressure |
| **fps** | int | Game frames per second | Affects real-time vs training speed |
| **seed** | int | Random seed for reproducibility | Controls food placement randomness |
| **maxSnakes** | int | Max concurrent snakes | Multi-agent vs single-agent |
| **rewardConfig** | json | Reward structure | Core learning signal |

### Reward Configuration Details

The `rewardConfig` JSON object defines the learning incentives:

- **`alive`**: Reward per step the snake stays alive
- **`eat_food`**: Reward for successfully eating food
- **`game_over`**: Penalty for dying (collision or starvation)

Example:
```json
{
  "alive": 0.1,
  "eat_food": 1.0, 
  "game_over": -1.0
}
```

## 🧠 Inference/Agent Parameters

| Parameter | Type | Description | Impact on Learning |
|-----------|------|-------------|-------------------|
| **maxEpisodes** | int | Total episodes to run | Training duration |
| **learningRate** | float | Neural network learning rate | Learning speed/stability |
| **beta** | float | Entropy bonus coefficient | Exploration vs exploitation |
| **gamma** | float | Discount factor | Short-term vs long-term focus |
| **batchSize** | int | Episodes before model update | Learning stability (1=online) |
| **snakeId** | string | Agent identifier | For multi-agent environments |

### Parameter Details

- **learningRate**: Typical values 0.0001-0.01. Higher = faster learning but less stable
- **beta**: 0 = pure exploitation, >0 = encourages exploration
- **gamma**: 0-1 range. Higher = more long-term planning
- **batchSize**: 1 = online learning, >1 = batch learning with better stability

## ⚙️ System Parameters

| Parameter | Type | Description | Impact |
|-----------|------|-------------|--------|
| **JOB_TIMEOUT** | int | Max seconds per experiment | Prevents infinite runs |
| **TARGET_EPISODES** | int | Episodes per experiment | Statistical significance |

## 🔄 Current Experiment Configuration

### Game Over Conditions Study

**Variable parameters being tested:**
- **maxStepsWithoutFood**: [30, 45, 67, 100, 150, 225, 337, 505, 757, 1000]
- **beta**: [0, 0.0001]

**Fixed optimized parameters:**
- **gridWidth/Height**: 11 (medium complexity)
- **batchSize**: 1 (online learning)
- **gamma**: 0.95 (balanced planning)
- **maxEpisodes**: 1000 (sufficient training)
- **learningRate**: 0.001 (stable learning)
- **rewardConfig**: `{"alive": 0.1, "eat_food": 1.0, "game_over": -1.0}`

### Experiment Focus

This setup tests the relationship between:
- **Starvation pressure** (maxStepsWithoutFood): How long can snakes survive without food?
- **Exploration behavior** (beta): Should agents explore randomly or focus on known strategies?

While keeping other environmental and training factors constant for fair comparison.

## 📊 Parameter Evolution

### Historical Changes

- **Grid size**: Reduced from 22×22 to 11×11 (4x smaller state space)
- **Reward structure**: Enhanced from weak signals to stronger learning incentives
- **Training duration**: Increased from 10 to 1000 episodes for statistical significance
- **Learning approach**: Moved from batch (32) to online (1) learning

### Current Focus Areas

1. **Survival optimization**: Finding optimal starvation thresholds
2. **Exploration strategies**: Balancing exploration vs exploitation
3. **Environment complexity**: Right-sized grid for efficient learning

This parameter configuration enables systematic study of core RL trade-offs while maintaining experimental rigor.