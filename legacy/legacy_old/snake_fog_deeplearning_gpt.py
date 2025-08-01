import pygame
import random
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Настройки игры ---
CELL_SIZE = 20
GRID_WIDTH = 15
GRID_HEIGHT = 15
GAME_WIDTH = GRID_WIDTH * CELL_SIZE
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10  # для отладки можно увеличить скорость

# --- Настройки зрения змейки ---
VISION_RADIUS = 5  # поле зрения (манхэттенское расстояние)
# Для прорисовки поля зрения будем использовать фиксированную сетку (11x11)
VISION_GRID_SIZE = (2 * VISION_RADIUS + 1, 2 * VISION_RADIUS + 1)

# --- Настройки окна ---
WINDOW_WIDTH = GAME_WIDTH + 220
WINDOW_HEIGHT = GAME_HEIGHT

# --- Цвета (RGB) ---
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GREEN     = (0, 255, 0)    # Голова змейки
DARKGREEN = (0, 155, 0)    # Тело змейки
RED       = (255, 0, 0)
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 50, 50)

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
    
    # Рисуем еду
    food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, food_rect)
    
    # Рисуем змейку
    for i, cell in enumerate(snake):
        x, y = cell
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = GREEN if i == 0 else DARKGREEN
        pygame.draw.rect(screen, color, rect)
    
    # Рисуем сетку
    for x in range(0, GAME_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, GAME_HEIGHT))
    for y in range(0, GAME_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (GAME_WIDTH, y))

def compute_visible_state(snake, food, direction):
    head_x, head_y = snake[0]
    visible_cells = {}
    
    # Функция поворота так, чтобы направление "вперёд" всегда смотрело вверх
    if direction == (0, -1):
        rotate = lambda dx, dy: (dx, dy)
    elif direction == (1, 0):
        rotate = lambda dx, dy: (dy, -dx)
    elif direction == (0, 1):
        rotate = lambda dx, dy: (-dx, -dy)
    elif direction == (-1, 0):
        rotate = lambda dx, dy: (-dy, dx)
    else:
        rotate = lambda dx, dy: (dx, dy)
    
    # Проходим по фиксированной сетке зрения (11x11), но включаем только клетки с manhattan расстоянием <= VISION_RADIUS
    for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            if abs(dx) + abs(dy) > VISION_RADIUS:
                continue
            # Будем использовать ключ в виде (dx + VISION_RADIUS, dy + VISION_RADIUS) для упорядочивания
            key = (dx + VISION_RADIUS, dy + VISION_RADIUS)
            global_x = (head_x + dx) % GRID_WIDTH
            global_y = (head_y + dy) % GRID_HEIGHT
            cell = (global_x, global_y)
            if (dx, dy) == (0, 0):
                cell_type = 'head'
            elif cell in snake:
                cell_type = 'body'
            elif cell == food:
                cell_type = 'food'
            else:
                cell_type = 'empty'
            visible_cells[key] = cell_type
    return visible_cells

def get_visible_matrix(visible_cells):
    """
    Преобразует поле зрения (словарь клеток) в фиксированный массив (N,2),
    где каждая клетка кодируется:
      - [1, 0] для змейки (головы или тела),
      - [0, 1] для еды,
      - [0, 0] для пустой клетки.
    Порядок клеток определяется сортировкой ключей по (row, col).
    """
    keys = sorted(visible_cells.keys(), key=lambda x: (x[1], x[0]))
    mat = []
    for key in keys:
        cell_type = visible_cells[key]
        if cell_type in ['head', 'body']:
            mat.append([1, 0])
        elif cell_type == 'food':
            mat.append([0, 1])
        else:
            mat.append([0, 0])
    return np.array(mat, dtype=np.float32)

def draw_vision_area(visible_cells):
    # Рисуем область зрения справа от игрового поля
    cell_size = 20
    grid_cols, grid_rows = VISION_GRID_SIZE
    vision_x_offset = GAME_WIDTH + 10
    vision_y_offset = (WINDOW_HEIGHT - grid_rows * cell_size) // 2
    for i in range(grid_rows):
        for j in range(grid_cols):
            cell_rect = pygame.Rect(vision_x_offset + j * cell_size,
                                    vision_y_offset + i * cell_size,
                                    cell_size, cell_size)
            key = (j, i)
            cell_type = visible_cells.get(key, 'empty')
            if cell_type == 'head':
                color = GREEN
            elif cell_type == 'body':
                color = DARKGREEN
            elif cell_type == 'food':
                color = RED
            else:
                color = WHITE
            pygame.draw.rect(screen, color, cell_rect)
            pygame.draw.rect(screen, GRAY, cell_rect, 1)

# --- Сеть политики и агент (REINFORCE с пакетным обновлением) ---
class PolicyNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNet, self).__init__()
        self.input_shape = input_shape  # например, (N,2), где N — число клеток в поле зрения (например, 61)
        self.num_actions = num_actions
        self.flatten_size = input_shape[0] * input_shape[1]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, num_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class PolicyAgent:
    def __init__(self, input_shape, num_actions, device, lr=1e-2, gamma=0.99, update_interval=1):
        self.device = device
        self.gamma = gamma
        self.n_actions = num_actions
        self.policy_net = PolicyNet(input_shape, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.current_log_probs = []
        self.current_rewards = []
        self.episode_buffer = []
        self.update_interval = update_interval

    def select_action(self, state):
        # Преобразуем состояние в тензор и добавляем батч-измерение
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.current_log_probs.append(m.log_prob(action))
        return int(action.item())

    def store_reward(self, reward):
        self.current_rewards.append(reward)

    def finish_episode(self):
        # Сохраняем данные текущего эпизода и выполняем обновление, если накоплено достаточно эпизодов.
        self.episode_buffer.append((self.current_log_probs, self.current_rewards))
        self.current_log_probs = []
        self.current_rewards = []
        
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
                all_log_probs.extend(ep_log_probs)
                all_returns.append(ep_returns)
            all_returns = torch.cat(all_returns)
            policy_loss = []
            for log_prob, R in zip(all_log_probs, all_returns):
                policy_loss.append(-log_prob * R)  # отрицательный знак для градиентного спуска
            loss = torch.stack(policy_loss).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.episode_buffer = []

# --- Основной игровой цикл ---
def main():
    episode = 0
    # Начальное положение змейки
    snake = [
        (GRID_WIDTH // 2, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
    ]
    direction = (1, 0)
    food = random_food_position(snake)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Определяем форму входного состояния (число клеток в поле зрения)
    dummy_visible = compute_visible_state(snake, food, direction)
    dummy_state = get_visible_matrix(dummy_visible)
    input_shape = dummy_state.shape  # например, (61, 2)
    
    agent = PolicyAgent(input_shape=input_shape, num_actions=3, device=device,
                        lr=1e-2, gamma=0.99, update_interval=1)
    
    with open("score_history.csv", "w") as f:
        f.write("episode,score\n")
    
    score = 0
    actions_map = {0: "straight", 1: "left", 2: "right"}
    running = True
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Получаем состояние поля зрения и преобразуем его в числовой массив
        visible_cells = compute_visible_state(snake, food, direction)
        state_matrix = get_visible_matrix(visible_cells)
        
        # Агент выбирает действие
        action = agent.select_action(state_matrix)
        move = actions_map[action]
        direction = relative_turn(direction, move)
        
        head_x, head_y = snake[0]
        dx, dy = direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
        
        # Проверяем столкновение с собой
        if new_head in snake:
            reward = -1
            agent.store_reward(reward)
            agent.finish_episode()
            with open("score_history.csv", "a") as f:
                f.write(f"{episode},{score}\n")
            episode += 1
            snake = [
                (GRID_WIDTH // 2, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
            ]
            direction = (1, 0)
            food = random_food_position(snake)
            score = 0
            continue
        
        snake.insert(0, new_head)
        if new_head == food:
            reward = 1
            agent.store_reward(reward)
            food = random_food_position(snake)
            score = len(snake) - 3
        else:
            reward = 0
            agent.store_reward(reward)
            snake.pop()
        
        # Отрисовка игрового поля и области зрения
        screen.fill(BLACK)
        draw_game_area(snake, food)
        draw_vision_area(visible_cells)
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        score_text = score_font.render(f"Episode: {episode} Score: {score}", True, WHITE)
        screen.blit(score_text, (GAME_WIDTH + 30, 10))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
