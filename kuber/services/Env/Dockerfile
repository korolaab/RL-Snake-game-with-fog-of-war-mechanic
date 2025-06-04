FROM python:3.11-slim

WORKDIR /app


COPY ./src /app

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "app.py", "--fps", "1", "--seed","1"]
