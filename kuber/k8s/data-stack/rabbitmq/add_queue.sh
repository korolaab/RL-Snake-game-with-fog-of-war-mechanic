k -n data-stack exec -it rabbitmq-0 -- curl -u admin:[password] \
  -X PUT \
  -H "Content-Type: application/json" \
  -d '{
    "durable": true,
    "auto_delete": false,
    "arguments": {
      "x-max-length": 1000000,
      "x-overflow": "drop-head",
      "x-queue-mode": "lazy"
    }
  }' \
  "http://localhost:15672/api/queues/%2F/rl_snake_logs"

