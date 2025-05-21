FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# Удаляем недоступные пакеты
RUN sed -i '/oneccl-bind-pt/d' requirements.txt && \
    sed -i '/intel_extension_for_pytorch/d' requirements.txt

RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0   -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html


COPY . .

ENTRYPOINT ["python", "main.py", "--mlflow_experiment_name", "test", "--no_render","--episodes","1000"]
