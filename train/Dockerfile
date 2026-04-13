FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch mlflow numpy

COPY train/train.py .
COPY services/Env/src/game.py .

ENTRYPOINT ["python", "train.py"]
