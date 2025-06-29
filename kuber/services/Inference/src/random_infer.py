import requests
import random
import time
import json
import argparse
import logging
import sys

def setup_logger(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_state_stream(base_url):
    """Чтение состояния змейки как json по строкам."""
    response = requests.get(base_url, stream=True)
    for line in response.iter_lines():
        if line:
            decoded = line.decode()
            try:
                data = json.loads(decoded)
                yield data
            except json.JSONDecodeError:
                logging.warning({"event": "failed_to_parse_json", "data": decoded})

def send_move(move_url, move: str):
    """Отправка управляющего действия."""
    payload = {"move": move}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(move_url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info({"event": "sent_move", "move": move})
    except requests.RequestException as e:
        logging.error({"event": "error_sending_move", "exception": e})

def random_agent(snake_id: str, log_file: str, env_host:str):
    setup_logger(log_file)
    base_url = f"http://{env_host}/snake/{snake_id}"
    move_url = f"{base_url}/move"
    logging.info({"event": "starting_agent", "snake_id": snake_id})

    for data in get_state_stream(base_url):
        logging.info({"event": "current_state", "data": data})
        if data.get("game_over"):
            logging.info({"event": "game_over"})
            break
        move = random.choice(["left", "right", "forward"])
        if move != "forward":
            send_move(move_url, move)
        time.sleep(0.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Snake Agent")
    parser.add_argument("--snake_id", required=True, help="ID of the snake")
    parser.add_argument("--env_host", type=str, required=True, help="Host of the snake game")
    parser.add_argument("--log_file", default="agent.log", help="Path to the log file")
    args = parser.parse_args()

    random_agent(args.snake_id, args.log_file, args.env_host)

