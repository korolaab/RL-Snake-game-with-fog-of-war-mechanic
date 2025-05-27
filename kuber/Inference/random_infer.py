import requests
import random
import time
import json

SNAKE_ID = "1"
BASE_URL = f"http://localhost:5000/snake/{SNAKE_ID}"
MOVE_URL = f"{BASE_URL}/move"

def get_state_stream():
    """Подключение к стриму состояния змейки (чистый JSON на каждой строке)."""
    response = requests.get(BASE_URL, stream=True)
    for line in response.iter_lines():
        if line:
            decoded = line.decode()
            try:
                data = json.loads(decoded)
                yield data
            except json.JSONDecodeError:
                print("⚠ Не удалось распарсить JSON:", decoded)

def send_move(move: str):
    """Отправка действия змейки."""
    payload = {"move": move}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(MOVE_URL, json=payload, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Ошибка при отправке движения: {e}")

def random_agent():
    print("🤖 Агент запущен. Подключение к стриму состояния...")
    for data in get_state_stream():
        if data["game_over"] == True:
            print("💀 Игра окончена.")
            break
        move = random.choice(["left", "right"])
        print(f"➡️ Делаем случайный ход: {move}")
        send_move(move)
        time.sleep(0.2)  # замедление, чтобы не спамить

if __name__ == "__main__":
    random_agent()

