# Dual Snake — No Talk (comm_dropout=1.0, ablation baseline)

## Setup

Same as `dual-snake-talk-vector` except:
- **Comm dropout**: 1.0 (talk vector completely zeroed out)

All other parameters identical:
- **Grid**: 11x11, vision_radius=5
- **Snakes**: 2, shared brain (SnakeNet), coupled life/death
- **Apple**: running, speed=0.1
- **Talk vector**: 12-dim (exists in network but input always dropped)
- **Hunger limit**: 500 steps
- **Episodes**: 3000
- **LR**: 0.001, gamma=0.9, beta=0.1

## Results

See [comparison.png](../dual-snake-talk-vector/comparison.png) for side-by-side with talk variant.

### Stats by block

| Episodes | Avg Apples | Avg Frames | Max Apples |
|----------|-----------|-----------|-----------|
| 0-500    | 0.30      | 27        | 3         |
| 500-1000 | 0.36      | 30        | 3         |
| 1000-1500| 0.50      | 53        | 3         |
| 1500-2000| 0.68      | 201       | 7         |
| 2000-2500| 0.52      | 456       | 5         |
| 2500-3000| 0.46      | 490       | 3         |

## Conclusion

Talk vector (comm_dropout=0.0) vs no talk (comm_dropout=1.0) show nearly identical performance at 3000 episodes on 11x11 grid. The communication channel did not provide a measurable advantage. Possible reasons:
- Grid too small — snakes already see each other in vision field
- 3000 episodes insufficient to learn a useful communication protocol
- Reward signal too sparse for communication to emerge

## Exact docker-compose

See [docker-compose.yaml](docker-compose.yaml) in this folder.
