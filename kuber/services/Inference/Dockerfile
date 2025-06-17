FROM python:3.11-slim

WORKDIR /app



RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0   -f https://download.pytorch.org/whl/cpu/torch_stable.html
COPY ./src /app
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT []
