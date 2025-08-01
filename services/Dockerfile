FROM python:3.11-slim

WORKDIR /app

# Install PyTorch first (large and stable dependencies)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy protobuf files from utils folder
COPY /utils/training.proto /app/
COPY /utils/generate_grpc.py /app/

COPY requirements.txt /app
RUN pip install -r requirements.txt

RUN apt update && apt install -y curl
RUN python generate_grpc.py

COPY /utils/logger.py /app/
