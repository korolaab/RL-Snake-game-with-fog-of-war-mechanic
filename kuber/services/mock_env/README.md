# mock_env Service

The mock_env service provides a lightweight, local REST API that emulates the real RL environment for development and debugging. It lets you test the Inference service in isolation—without spinning up the actual game environment.

## Purpose

- Mocks REST endpoints used by Inference (state stream, move, reset)
- Simulates agent-environment loop, with deterministic, repeatable data
- Useful for rapid development, debugging interface or workflow bugs, and CI/testing

## Key Features

- `/snake/<sid>` — Streams a fixed (mocked) sequence of game states (10 moves, then sets `game_over=true`)
- `/snake/<sid>/move` — Accepts movement commands via POST (`{"move": "left"|"right"}`) and always returns success if valid
- `/snake/<sid>/reset` — Resets the internal counter for the test snake (for repeatable testing)
- Data and endpoints conform with expected environment API contract
- Thread-safe using locks for repeated calls

## Directory Structure

```
src/
  app.py              # Flask app with all endpoints
  requirements.txt    # Python dependencies for development/testing
Dockerfile
```

## Usage

### Build and Run (Docker)
```Dockerfile
FROM korolaab/snake_rl_base:latest
COPY ./src /app
ENTRYPOINT []
```
Build:
```sh
docker build -t mock_env_service .
```
Run:
```sh
docker run -p 5000:5000 mock_env_service
```

### Run directly (Python)
```sh
pip install flask
python src/app.py
```

## API Endpoints

| Endpoint                      | Description                                    | Method | Example cURL                           |
|-------------------------------|------------------------------------------------|--------|----------------------------------------|
| /snake/<sid>                  | Stream (NDJSON) of state observations (10+1)   | GET    | curl http://localhost:5000/snake/test  |
| /snake/<sid>/move             | Accepts moves ('left', 'right')                | POST   | curl -X POST .../move -H ... -d ...    |
| /snake/<sid>/reset            | Resets mock state for snake ID                 | POST   | curl -X POST .../reset                 |
| /                             | Health check                                   | GET    | curl http://localhost:5000/            |

## Development Notes

- Used strictly for debug/testing, not intended for production or training!
- No persistent state; for full episode, call `/reset` between runs if needed.
- Data can be extended for advanced scenarios (different maps, rewards, etc).

---
