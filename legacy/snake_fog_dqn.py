import pygame
import random
import sys
import csv
from collections import deque, defaultdict
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim

# --- Game settings ---
CELL_SIZE = 20
GRID_WIDTH = 15
GRID_HEIGHT = 15
GAME_WIDTH = GRID_WIDTH * CELL_SIZE   # 300 pixels
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE   # 300 pixels
FPS = 10

# --- Snake vision settings ---
VISION_RADIUS = 5
VISION_DISPLAY_COLS = 11  
VISION_DISPLAY_ROWS = 11   
VISION_CELL_SIZE = 20     
VISION_WIDTH = VISION_DISPLAY_COLS * VISION_CELL_SIZE   # 220 pixels
VISION_HEIGHT = VISION_DISPLAY_ROWS * VISION_CELL_SIZE    # 220 pixels

# --- Window settings ---
WINDOW_WIDTH = GAME_WIDTH + VISION_WIDTH + 200  # 300 + 220 = 520 pixels
WINDOW_HEIGHT = GAME_HEIGHT               # 300 pixels

# --- Colors (RGB) ---
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GREEN     = (0, 255, 0)    # Snake head
DARKGREEN = (0, 155, 0)    # Snake body
RED       = (255, 0, 0)
BLUE      = (0, 0, 255)    # Field boundary
GRAY      = (100, 100, 100)
DARKGRAY  = (50, 51, 50)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake + Vision (DQN)")
clock = pygame.time.Clock()
score_font = pygame.font.SysFont("Arial", 24)

# --------------------------
# Игровая логика
# --------------------------
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
    # Не разрешаем разворот на 180 градусов (если snake уже движется)
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
    """
    Вычисляет строковое представление состояния (для отладки) и visible_cells – словарь ячеек,
    которые видит змейка в области с радиусом VISION_RADIUS (по L1) относительно головы.
    """
    head_x, head_y = snake[0]
    visible_cells = {}
    
    # Определяем функцию поворота так, чтобы "вперёд" было всегда вверх.
    if direction == (0, -1):  # вверх: без поворота
        def rotate(dx, dy):
            return (dx, dy)
    elif direction == (1, 0):  # вправо: поворот на 90° по часовой стрелке
        def rotate(dx, dy):
            return (dy, -dx)
    elif direction == (0, 1):  # вниз: поворот на 180°
        def rotate(dx, dy):
            return (-dx, -dy)
    elif direction == (-1, 0):  # влево: поворот на 90° против часовой стрелки
        def rotate(dx, dy):
            return (-dy, dx)
    else:
        def rotate(dx, dy):
            return (dx, dy)
    
    # Для отладки – строковое представление состояния
    apple_visible = 0
    apple_vertical = "none"
    apple_horizontal = "none"
    apple_colinear = 0
    
    for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            if abs(dx) + abs(dy) > VISION_RADIUS:
                continue
            # Исправлено: GRID_WIDTH вместо некорректного значения
            global_x = (head_x + dx) % GRID_WIDTH
            global_y = (head_y + dy) % GRID_HEIGHT
            if (global_x, global_y) == food:
                apple_visible = 1
                r_x, r_y = rotate(dx, dy)
                if r_y < 0:
                    apple_vertical = "above"
                elif r_y > 0:
                    apple_vertical = "behind"
                else:
                    apple_vertical = "center"
                if r_x > 0:
                    apple_horizontal = "right"
                elif r_x < 0:
                    apple_horizontal = "left"
                else:
                    apple_horizontal = "center"
                if r_x == 0 or r_y == 0:
                    apple_colinear = 1
                break
        if apple_visible:
            break

    tail_above = 0
    tail_left = 0
    tail_right = 0
    for segment in snake[1:]:
        seg_dx = segment[0] - head_x
        seg_dy = segment[1] - head_y
        r_seg = rotate(seg_dx, seg_dy)
        if r_seg[1] < 0:
            tail_above = 1
        if r_seg[0] < 0:
            tail_left = 1
        if r_seg[0] > 0:
            tail_right = 1

    state_str = f"{apple_visible}, {apple_vertical}, {apple_horizontal}, {apple_colinear}, {tail_above}, {tail_left}, {tail_right}"
    
    # Формируем visible_cells для рендеринга области зрения.
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
    """
    Преобразует visible_cells в числовую матрицу размером 11x11.
    Значения:
      -1 : ячейка вне области видимости,
       0 : пустая ячейка,
       1 : змейка (голова или тело),
       2 : еда.
    TODO: 
        - Убрать вне области видимости
        - Cделать one_hot 
        - Понизить нейронную капасити сети. Relu 3 нейрона. categorial crossentropy.
        - Один скрытый слой.
        - Вхондной 8 тангенс -1 до 1
        - Скрытый слой 16 тангес -1 до 1
        - Выходной 3 softmax
    baby steps - с палочкой.
    """
    mat = []
    i = 0
    for (col, row), color in visible_cells.items():
        if color == DARKGREEN:
            mat.append([1,0])
        elif color == RED:
            mat.append([0,1])
        elif color == WHITE:
            mat.append([0,0])
    return np.array(mat,dtype=np.float64)

# --------------------------
# DQN агент на PyTorch
# --------------------------
from torchsummary import summary
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape  # Например, (11, 11)
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Flatten(),  # Преобразуем матрицу в вектор
            nn.Linear(input_shape[0] * input_shape[1], 8),
            nn.Tanh(),
            nn.Linear(8,16),
            nn.Tanh(),
            nn.Linear(16, num_actions),
            nn.Softmax()
        )
        
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, input_shape, num_actions, device, epsilon=0.5, lr=1e-3, gamma=0.99, buffer_capacity=10000, batch_size=32):
        self.device = device
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.policy_net = DQN(input_shape, num_actions).to(device)
        summary(self.policy_net, input_size=(1,60,2))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
    
    def select_action(self, state):
        # state имеет размер (11,11); добавляем batch dimension
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                predict = self.policy_net(state_tensor)
            return int(predict.argmax().item())
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)  # shape: (B, 11, 11)
        max_reward_idx = torch.argmax(torch.tensor(rewards))  # Берем индекс лучшего действия
        target_actions = torch.tensor([actions[max_reward_idx]])  # Целевое действие
        
        outputs = self.policy_net(states) 
        
        print(f"outputn\n{outputs}")
        print(f"target_actions\n{target_actions}")
        input("wait") 
        loss = nn.CrossEntropyLoss()(target_actions.label.long(), outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --------------------------
# Основной игровой цикл
# --------------------------
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
    game_over = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_shape=(60,2),
                         num_actions=3,
                         device=device,
                         epsilon=0.3,
                         lr=1e-3,
                         gamma=0.99,
                         buffer_capacity=10000,
                         batch_size=32)
    
    # Событие для сохранения параметров (опционально)
    TIMER_QSTATE_SAVE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(TIMER_QSTATE_SAVE_EVENT, 5000)

    last_score = 0
    with open("score_history.csv","w") as f:
        f.write("episode,score\n")
    ticks = 0
    score = 0
    
    # Маппинг действий: 0 -> "straight", 1 -> "left", 2 -> "right"
    actions_map = {0: "straight", 1: "left", 2: "right"}
    
    while True:
        screen.fill(BLACK)
        
        # Вычисляем текущее visible_state для рендеринга и для агента.
        state_str, visible_cells = compute_visible_state(snake, food, direction)
        state_matrix = get_visible_matrix(visible_cells)
        #print(state_matrix.shape) 
        # Агент выбирает действие
        action = agent.select_action(state_matrix)
        action = 0
        move = actions_map[action]
        direction = relative_turn(direction, move)
        
        reward = 0
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == TIMER_QSTATE_SAVE_EVENT:
                # Здесь можно сохранить параметры агента, если требуется
                print("Сохранение параметров агента...")
            elif event.type == pygame.KEYDOWN:
                # Позволяем управлять вручную (опционально)
                if event.key == pygame.K_LEFT:
                    direction = relative_turn(direction, "left")
                elif event.key == pygame.K_RIGHT:
                    direction = relative_turn(direction, "right")
                if event.key == pygame.K_UP:
                    FPS = 10 if FPS > 10 else 100000 
        
        # Обновление состояния игры: перемещение змейки
        head_x, head_y = snake[0]
        dx, dy = direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
        done = False
        if new_head in snake or len(snake) == 1:
            done = True
            reward = -1
        else:
            snake.insert(0, new_head)
            if new_head == food:
                reward = 1
                food = random_food_position(snake)
            else:
                reward = -0.01  # штраф за время
                if ticks == 500:
                    reward = -0.02
                    snake.pop()
                    ticks = 0
                ticks += 1
                snake.pop()
        
        # После обновления, вычисляем следующее состояние
        next_state_str, next_visible_cells = compute_visible_state(snake, food, direction)
        next_state_matrix = get_visible_matrix(next_visible_cells)
        
        # Запоминаем переход и обучаем агента
        agent.remember(state_matrix, action, reward, next_state_matrix, done)
        agent.train_step()
        
        if done:
            with open("score_history.csv","a") as f:
                f.write(f"{episode},{score}\n")
            episode += 1
            avg_score = (avg_score * (episode - 1) + (len(snake) - 3)) / episode
            snake = [
                (GRID_WIDTH // 2, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
                (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
            ]
            direction = (1, 0)
            food = random_food_position(snake)
            done = False
        
        draw_game_area(snake, food)
        draw_vision_area(visible_cells)
        pygame.draw.line(screen, GRAY, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        score = len(snake) - 3
        score_text = score_font.render(f"episode: {episode} avg: {avg_score:.2f} score: {score} eps: {agent.epsilon:.2f}", True, WHITE)
        score_rect = score_text.get_rect(topright=(WINDOW_WIDTH - 10, 10))
        screen.blit(score_text, score_rect)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

