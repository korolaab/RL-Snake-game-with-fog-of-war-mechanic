# Legacy vs Current System Comparison

This document compares the monolithic legacy system with the current Kubernetes-based distributed system, focusing on RL training, neural networks, and environment differences.

## 🏗️ Architecture Overview

| Aspect | Legacy Monolithic | Current Kubernetes |
|--------|------------------|-------------------|
| **Deployment** | Single Python process | 3 separate services (env, inference, training) |
| **Communication** | Direct function calls | HTTP API + gRPC |
| **Scaling** | Single machine only | Distributed, scalable |
| **Components** | All-in-one (`main.py`) | Service-oriented architecture |

## 🧠 Neural Network Architecture

| Aspect | Legacy (`network.py`) | Current K8s (`snake_model.py`) |
|--------|----------------------|--------------------------------|
| **Architecture** | `62×3 → 8 → 16 → 3` | `input_size → 16 → 8 → 3` |
| **Hidden Units** | 8, 16 (configurable) | 16, 8 (fixed, reversed) |
| **Activations** | Tanh + LayerNorm | Tanh only |
| **Dropout** | 0.5 | 0.5 (same) |
| **Output** | Softmax | Softmax(dim=1) |
| **Normalization** | LayerNorm after each layer | None |
| **Flexibility** | Configurable architecture | Fixed architecture |

### Network Code Comparison

**Legacy Network:**
```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_shape[0] * input_shape[1], hidden_units_1),  # default: 8
    nn.LayerNorm(hidden_units_1),
    nn.Tanh(),
    nn.Dropout(dropout_rate),  # default: 0.5
    nn.Linear(hidden_units_1, hidden_units_2),  # default: 16
    nn.LayerNorm(hidden_units_2),
    nn.Tanh(),
    nn.Dropout(dropout_rate),
    nn.Linear(hidden_units_2, num_actions),  # 3 actions
    nn.LayerNorm(num_actions),
    nn.Softmax(dim=-1)
)
```

**Current Network:**
```python
nn.Sequential(
    nn.Linear(input_size, 16),     # Reversed: bigger first layer
    nn.Tanh(),
    nn.Dropout(0.5),
    nn.Linear(16, 8),              # Smaller second layer
    nn.Tanh(),
    nn.Dropout(0.5),
    nn.Linear(8, 3),               # 3 actions
    nn.Softmax(dim=1)
)
```

## 🎯 RL Training Parameters

| Parameter | Legacy (`agent.py`) | Current K8s (`snake_agent.py`) | Change Impact |
|-----------|--------------------|---------------------------------|---------------|
| **Learning Rate** | `1e-3` (0.001) | `0.001` (same) | No change |
| **Beta (Entropy)** | `0.1` (default) | `0.1` (default) | No change |
| **Gamma (Discount)** | `0.99` | `0.99` (default) | No change |
| **Epsilon** | `0.2` (ε-greedy) | Not used | Lost dual exploration |
| **Batch Size** | Configurable episodes | `5` episodes (default) | Less flexible |
| **Optimizer** | Adam | Adam | Same |
| **Grad Clipping** | None | `max_norm=0.5` | Added stability |
| **Update Frequency** | `update_interval=1` | After each episode | Same |

## 🏆 Reward Structure

| Event | Legacy (`game.py`) | Current K8s | Learning Impact |
|-------|-------------------|-------------|-----------------|
| **Every Step** | `+1` (survival-based) | `+0.1` per step | 10x weaker survival signal |
| **Eat Food** | Implicit (via survival) | `+1.0` | Strong food-seeking incentive |
| **Game Over** | Implicit penalty | `-1.0` | Clear death penalty |
| **Starvation** | Not implemented | Death after N steps | New failure mode |

### Reward Philosophy Change
- **Legacy**: Pure survival optimization (`+1` per step alive)
- **Current**: Event-based rewards (food=+1.0, death=-1.0, alive=+0.1)

## 🎮 Environment Differences

| Parameter | Legacy (`config.py`) | Current K8s | Impact |
|-----------|---------------------|-------------|--------|
| **Grid Size** | 15×15 (225 cells) | 11×11 (121 cells) | 46% smaller state space |
| **Training Speed** | FPS=100,000 | FPS=100 | 1000x slower training |
| **Boundaries** | Wrapping (teleport) | Wall collision | Different death mechanics |
| **Multi-agent** | Up to N snakes | Single agent only | No multi-agent learning |
| **Vision** | 5 radius, 11×11 display | Same | No change |
| **Starvation** | None | `maxStepsWithoutFood` | Added time pressure |

## 📊 State Representation

### Legacy State Matrix (62×3)
```python
# Vision cells: 61 cells × 3 channels
[1,0,0]  # Own snake body
[0,1,0]  # Other snake  
[0,0,1]  # Food
[0,0,0]  # Empty

# + Snake length info (1×3):
[is_alive, 1-is_alive, 0]  # where is_alive = exp(-abs(length))

# + Last action info (1×3):
[1,0,0] if last_action != 1 else [0,1,0]
```

### Current State Processing
- **Dynamic input size** based on `state_processor.py`
- **Vision-based** (exact format varies by game state)
- **Simpler structure** (no metadata encoding)

## 🔄 Training Algorithm

Both systems use **REINFORCE with entropy bonus**, but with key differences:

### Legacy Training:
```python
# Fast, direct training with dual exploration
for episode in episodes:
    states, actions, rewards = play_episode()  # In-memory, FPS=100k
    if random.random() < epsilon:  # ε-greedy exploration
        action = random_action()
    else:
        action = policy_action() + entropy_bonus  # β=0.1
    
    loss = policy_gradient_loss + beta * entropy_loss
    optimizer.step()  # Immediate update
```

### Current Training:
```python
# Slower, distributed training with single exploration
for episode in episodes:
    state = http_get("/snake/adam")          # Network overhead
    action = model.predict(state)            # Entropy exploration only
    http_post("/snake/adam/move", action)    # FPS=100
    
    # Training happens in separate service via gRPC
    if batch_complete:
        success = agent.send_training_batch_and_wait()
```

## ⚡ Performance Comparison

### Training Speed
- **Legacy**: ~1000x faster (FPS=100,000 vs 100)
- **Current**: Realistic speed but much slower learning

### Learning Stability
- **Legacy**: Batch updates + LayerNorm (more stable)
- **Current**: Online learning + gradient clipping (less stable)

### Exploration
- **Legacy**: Dual strategy (ε-greedy + entropy)
- **Current**: Single strategy (entropy only)

### Resource Usage
- **Legacy**: Single process, direct memory
- **Current**: Multiple containers, network overhead

## 🔍 Key Trade-offs

### What was Gained (Current)
- ✅ **Distributed scaling** (multiple environments)
- ✅ **Service isolation** (failures don't crash everything)  
- ✅ **Production readiness** (Kubernetes deployment)
- ✅ **API-based communication** (language-agnostic)
- ✅ **Gradient clipping** (training stability)
- ✅ **Starvation mechanics** (more realistic)

### What was Lost (Legacy→Current)
- ❌ **Training speed** (1000x slower)
- ❌ **Multi-agent support** (N snakes → 1 snake)
- ❌ **Visual interface** (PyGame GUI)
- ❌ **Flexible architecture** (configurable → fixed network)
- ❌ **Dual exploration** (ε-greedy removed)
- ❌ **LayerNorm** (training stability)
- ❌ **Larger environment** (15×15 → 11×11)

## 📈 Learning Implications

### Legacy Advantages
- **Faster convergence** due to high FPS and stable training
- **Better exploration** with ε-greedy + entropy
- **Larger state space** for complex strategies
- **Multi-agent emergent behaviors**

### Current Advantages  
- **Realistic environment** with starvation pressure
- **Dense reward signals** encouraging food-seeking
- **Better production deployment**
- **Isolated failure modes**

## 🎯 Recommendations

For **Research**: Consider hybrid approach
- Keep distributed architecture for scalability
- Add multi-agent support back
- Increase FPS for faster training
- Restore ε-greedy exploration

For **Production**: Current system is well-suited
- Reliable service architecture
- Good monitoring and logging
- Scalable experiment execution

## 📝 Migration Notes

When moving between systems:
1. **Model compatibility**: Different input sizes and architectures
2. **Reward scaling**: Legacy rewards 10x higher than current
3. **Episode length**: Current has starvation cutoff
4. **Exploration**: Need to tune beta higher to compensate for missing ε-greedy
5. **Training time**: Expect much longer convergence in current system

This comparison shows the classic trade-off between **research velocity** (legacy) and **production scalability** (current).