FROM korolaab/snake_rl_base:latest

WORKDIR /app

# Default port - can be overridden in k8s deployment
ENV INFERENCE_HOST=localhost
ENV INFERENCE_PORT=5555
ENV ENV_HOST=localhost
ENV ENV_PORT=5555
ENV FREQUENCY=10.0

# Copy application code last
COPY ./src /app

ENTRYPOINT []

CMD ["python", "main.py", \
     "--inference-host", "${INFERENCE_HOST}", \
     "--inference-port", "${INFERENCE_PORT}", \
     "--env-host", "${ENV_HOST}", \
     "--env-port", "${ENV_PORT}", \
     "--frequency", "${FREQUENCY}"]