import pygame
import random
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Game settings ---
CELL_SIZE = 20
GRID_WIDTH = 15
GRID_HEIGHT = 15
GAME_WIDTH = GRID_WIDTH * CELL_SIZE   # 300 pixels
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE   # 300 pixels
FPS = 100000

# --- Snake vision settings ---
VISION_RADIUS = 5
VISION_DISPLAY_COLS = 11  
VISION_DISPLAY_ROWS = 11   
VISION_CELL_SIZE = 20     
VISION_WIDTH = VISION_DISPLAY_COLS * VISION_CELL_SIZE   # 220 pixels
VISION_HEIGHT = VISION_DISPLAY_ROWS * VISION_CELL_SIZE    # 220 pixels

# --- Window settings ---
WINDOW_WIDTH = GAME_WIDTH + VISION_WIDTH + 200  # 300 + 220 = 520 pixels
WINDOW_HEIGHT = GAME_HEIGHT                     # 300 pixels

# --- Colors (RGB) ---
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GREEN     = (0, 255, 0)    # Snake head
DARKGREEN = (0, 155, 0)    # Snake body
RED       = (255, 0, 0)
BLUE      = (0, 0, 255)
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 51, 50)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake + Vision (Policy Gradient with Batch Update)")
clock = pygame.time.Clock()
score_font = pygame.font.SysFont("Arial", 24)

def random_food_position(snake):
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake:
            return pos

def turn_snake(current_direction, command):
    mapping = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }
    new_direction = mapping.get(command, current_direction)
    if (current_direction[0] + new_direction[0] == 0 and 
        current_direction[1] + new_direction[1] == 0) and current_direction != (0, 0):
        return current_direction
    return new_direction

def relative_turn(direction, turn_command):
    if turn_command == "left":
        return (direction[1], -direction[0])
    elif turn_command == "right":
        return (-direction[1], direction[0])
    else:
        return direction

def draw_game_area(snake, food):
    game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
    pygame.draw.rect(screen, BLACK, game_rect)
    
    food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, food_rect)
    
    for i, cell in enumerate(snake):
        x, y = cell
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = GREEN if i == 0 else DARKGREEN
        pygame.draw.rect(screen, color, rect)
    
    for x in range(0, GAME_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, GAME_HEIGHT))
    for y in range(0, GAME_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (GAME_WIDTH, y))

def compute_visible_state(snake, food, direction):
    head_x, head_y = snake[0]
    visible_cells = {}
    
    # Rotate so that "forward" is always up.
    if direction == (0, -1):
        def rotate(dx, dy): return (dx, dy)
    elif direction == (1, 0):
        def rotate(dx, dy): return (dy, -dx)
    elif direction == (0, 1):
        def rotate(dx, dy): return (-dx, -dy)
    elif direction == (-1, 0):
        def rotate(dx, dy): return (-dy, dx)
    else:
        def rotate(dx, dy): return (dx, dy)
    
    # For debugging: compute some features of the state (not used for training)
    apple_visible = 0
    apple_vertical = "none"
    apple_horizontal = "none"
    apple_colinear = 0
    for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            if abs(dx) + abs(dy) > VISION_RADIUS:
                continue
            global_x = (head_x + dx) % GRID_WIDTH
            global_y = (head_y + dy) % GRID_HEIGHT
            if (global_x, global_y) == food:
                apple_visible = 1
                r_x, r_y = rotate(dx, dy)
                if r_y < 0: apple_vertical = "above"
                elif r_y > 0: apple_vertical = "behind"
                else: apple_vertical = "center"
                if r_x > 0: apple_horizontal = "right"
                elif r_x < 0: apple_horizontal = "left"
                else: apple_horizontal = "center"
                if r_x == 0 or r_y == 0:
                    apple_colinear = 1
                break
        if apple_visible:
            break

    tail_above = tail_left = tail_right = 0
    for segment in snake[1:]:
        seg_dx = segment[0] - head_x
        seg_dy = segment[1] - head_y
        r_seg = rotate(seg_dx, seg_dy)
        if r_seg[1] < 0: tail_above = 1
        if r_seg[0] < 0: tail_left = 1
        if r_seg[0] > 0: tail_right = 1

    state_str = f"{apple_visible}, {apple_vertical}, {apple_horizontal}, {apple_colinear}, {tail_above}, {tail_left}, {tail_right}"
    
    # Build visible_cells for rendering the field of view.
    for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            if abs(dx) + abs(dy) > VISION_RADIUS:
                continue
            r_x, r_y = rotate(dx, dy)
            disp_col = (VISION_DISPLAY_COLS // 2) + r_x
            disp_row = (VISION_DISPLAY_ROWS // 2) + r_y
            if not (0 <= disp_col < VISION_DISPLAY_COLS and 0 <= disp_row < VISION_DISPLAY_ROWS):
                continue
            global_x = (head_x + dx) % GRID_WIDTH
            global_y = (head_y + dy) % GRID_HEIGHT
            cell = (global_x, global_y)
            if (dx, dy) == (0, 0):
                color = GREEN
            elif cell in snake:
                color = DARKGREEN
            elif cell == food:
                color = RED
            else:
                color = WHITE
            visible_cells[(disp_col, disp_row)] = color

    return state_str, visible_cells

def draw_vision_area(visible_cells):
    vision_x_offset = GAME_WIDTH + 100
    vision_y_offset = (WINDOW_HEIGHT - (VISION_DISPLAY_ROWS * VISION_CELL_SIZE)) // 2
    vision_rect = pygame.Rect(vision_x_offset, vision_y_offset, VISION_DISPLAY_COLS * VISION_CELL_SIZE, VISION_DISPLAY_ROWS * VISION_CELL_SIZE)
    pygame.draw.rect(screen, DARKGRAY, vision_rect)
    
    for col in range(VISION_DISPLAY_COLS):
        for row in range(VISION_DISPLAY_ROWS):
            cell_rect = pygame.Rect(vision_x_offset + col * VISION_CELL_SIZE,
                                    vision_y_offset + row * VISION_CELL_SIZE,
                                    VISION_CELL_SIZE, VISION_CELL_SIZE)
            if (col, row) in visible_cells:
                pygame.draw.rect(screen, visible_cells[(col, row)], cell_rect)
            pygame.draw.rect(screen, GRAY, cell_rect, 1)

def get_visible_matrix(visible_cells):
    # Create a numerical matrix (11x11) from the visible_cells:
    # -1: cell outside FOV, 0: empty, 1: snake, 2: food.
    mat = []
    i = 0
    for (col, row), color in visible_cells.items():
        if color == DARKGREEN:
            mat.append([1,0])
        elif color == RED:
            mat.append([0,1])
        elif color == WHITE:
            mat.append([0,0])
    return mat

from torchsummary import summary
# --- Policy Network and Agent using REINFORCE with Batch Updates ---
class PolicyNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNet, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 1024),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        probs = self.net(x)
        return probs

class PolicyAgent:
    def __init__(self, input_shape, num_actions, device, epsilon = 0.2, lr=1e-3, gamma=0.99, update_interval=1):
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = num_actions
        self.policy_net = PolicyNet(input_shape, num_actions).to(device)
        summary(self.policy_net, input_size=(1,input_shape[0],input_shape[1]))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # Data storage for one episode
        self.current_log_probs = []
        self.current_rewards = []
        # Buffer for accumulating data over several episodes
        self.entropy_history = []
        self.episode_buffer = []
        self.update_interval = update_interval

    def _egreedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions) 
        else:
            return self._network(state)

    def _network(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        #print(probs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.current_log_probs.append(m.log_prob(action))
        # Сохраняем энтропию для текущего распределения действий
        entropy = m.entropy().unsqueeze(0)
        self.entropy_history.append(entropy)
        return int(action.item())

    def select_action(self, state):
        #return self._egreedy_policy(state) 
        return self._network(state) 

    def store_reward(self, reward):
        self.current_rewards.append(reward)
    
    def finish_episode(self):
        # Store current episode data and reset for next episode.
        self.episode_buffer.append((self.current_log_probs, self.current_rewards))
        self.current_log_probs = []
        self.current_rewards = []
        
        # If we've accumulated enough episodes, perform a batch update.
        if len(self.episode_buffer) >= self.update_interval:
            all_log_probs = []
            all_returns = []
            for ep_log_probs, ep_rewards in self.episode_buffer:
                R = 0
                ep_returns = []
                for r in reversed(ep_rewards):
                    R = r + self.gamma * R
                    ep_returns.insert(0, R)
                ep_returns = torch.tensor(ep_returns, dtype=torch.float32).to(self.device)
               # print(ep_returns)
                ep_returns = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-5)
                #print(ep_returns.std())
                #print(ep_returns.mean())
                #print(ep_returns)
                #input()
                all_log_probs.extend(ep_log_probs)
                all_returns.append(ep_returns)
            all_returns = torch.cat(all_returns)
            policy_loss = []
            for log_prob, R in zip(all_log_probs, all_returns):
                policy_loss.append(-log_prob * R)

            if self.entropy_history:
                entropy_term = torch.cat(self.entropy_history).sum()
            else:
                entropy_term = 0
            beta = 0.1
            loss = torch.stack(policy_loss).sum() - beta * entropy_term
            print(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.episode_buffer = []  # Clear buffer after update
            self.entropy_history = []

# --- Main Game Loop ---
def main():
    global FPS
    avg_score = 0  
    episode = 0
    snake = [
        (GRID_WIDTH // 2, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
    ]
    direction = (1, 0)
    food = random_food_position(snake)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PolicyAgent(input_shape=(62,2),
                        num_actions=3,
                        device=device,
                        lr=0.005,
                        gamma=0.5,
                        update_interval=1)
    
    with open("score_history.csv", "w") as f:
        f.write("episode,score\n")
    
    ticks = 0
    score = 0
    actions_map = {0: "straight", 
                   1: "left", 
                   2: "right"}
    last_action =  0
    
    while True:
        screen.fill(BLACK)
        state_str, visible_cells = compute_visible_state(snake, food, direction)
        state_matrix = get_visible_matrix(visible_cells)
        is_alive = np.exp(-np.abs(len(snake)))
        state_matrix.append([is_alive, 1 - is_alive])
        
        last_action_vector = [0,0]
        if( last_action == 1):
            
            last_action_vector = [0,1]
        else:
            last_action_vector = [1,0]

        state_matrix.append(last_action_vector)
        state_matrix = np.array(state_matrix) 
        # Agent selects an action based on the current state
        action = agent.select_action(state_matrix)
        last_action = action
        move = actions_map[action]
        direction = relative_turn(direction, move)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # Optional manual override
                if event.key == pygame.K_LEFT:
                    direction = relative_turn(direction, "left")
                elif event.key == pygame.K_RIGHT:
                    direction = relative_turn(direction, "right")
                if event.key == pygame.K_UP:
                    FPS = 10 if FPS > 10 else 100000
         
        head_x, head_y = snake[0]
        dx, dy = direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
        done = False
        reward = 1
        if new_head in snake or len(snake) == 1:
            done = True
        else:
            snake.insert(0, new_head)
            if new_head == food:
                reward += 1 
                food = random_food_position(snake)
            else:
                if ticks == 50:
                    snake.pop()
                    ticks = 0
                snake.pop()
        
        ticks += 1
        agent.store_reward(reward)
        
        if done:
            agent.finish_episode()  # Update agent using batch update if interval met
            with open("score_history.csv", "a") as f:
                f.write(f"{episode},{score}\n")
            episode += 1
            avg_score = (avg_score * (episode - 1) + (len(snake) - 3)) / episode
            print(f"{episode} {avg_score=}")
            snake = [
                (GRID_WIDTH // 2, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
            ]
            direction = (1, 0)
            food = random_food_position(snake)
            done = False
            score = 0
        
        state_str, visible_cells = compute_visible_state(snake, food, direction)
        draw_game_area(snake, food)
        draw_vision_area(visible_cells)
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        score = max(score, len(snake) - 3)
        
        score_text = score_font.render(f"episode: {episode} avg: {avg_score:.2f} score: {score}", True, WHITE)
        score_rect = score_text.get_rect(topright=(WINDOW_WIDTH - 10, 10))
        screen.blit(score_text, score_rect)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

