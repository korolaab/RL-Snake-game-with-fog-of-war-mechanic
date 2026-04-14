FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch mlflow numpy optuna

COPY train/train.py .
COPY train/hyperopt.py .
COPY services/Env/src/game.py .

ENTRYPOINT ["python", "train.py"]
