FROM python:3.11-slim

WORKDIR /app


COPY ./src /app
RUN apt update && apt install -y curl
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT []
