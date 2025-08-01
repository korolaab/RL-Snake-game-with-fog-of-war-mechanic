# Using ClickHouse and RabbitMQ

## Accessing Services

### Local Development (Port Forward)

```bash
make port-forward
```

- **ClickHouse**: http://localhost:8123/play
- **RabbitMQ**: http://localhost:15672

### From Applications (In-Cluster)

**ClickHouse**:
- HTTP: `clickhouse.data-stack.svc.cluster.local:8123`
- TCP: `clickhouse.data-stack.svc.cluster.local:9000`

**RabbitMQ**:
- AMQP: `rabbitmq.data-stack.svc.cluster.local:5672`
- Management: `rabbitmq.data-stack.svc.cluster.local:15672`

## Python Examples

### ClickHouse

```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host='clickhouse.data-stack.svc.cluster.local',
    port=8123,
    username='default',
    password='your-password'
)

# Create table
client.execute("""
    CREATE TABLE IF NOT EXISTS rl_experiments (
        id UUID DEFAULT generateUUIDv4(),
        experiment_name String,
        episode Int32,
        reward Float64,
        timestamp DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY (experiment_name, episode)
""")

# Insert data
client.insert('rl_experiments', [
    ['dqn_cartpole', 1, 200.0],
    ['dqn_cartpole', 2, 195.0]
], column_names=['experiment_name', 'episode', 'reward'])

# Query data
result = client.query('SELECT * FROM rl_experiments LIMIT 10')
print(result.result_rows)
```

### RabbitMQ

```python
import pika
import json

# Connection
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='rabbitmq.data-stack.svc.cluster.local',
        port=5672,
        credentials=pika.PlainCredentials('admin', 'your-password')
    )
)
channel = connection.channel()

# Declare queue
channel.queue_declare(queue='rl_jobs', durable=True)

# Publish job
job = {
    'algorithm': 'dqn',
    'environment': 'CartPole-v1',
    'episodes': 1000
}

channel.basic_publish(
    exchange='',
    routing_key='rl_jobs',
    body=json.dumps(job),
    properties=pika.BasicProperties(delivery_mode=2)  # Persistent
)

# Consume jobs
def process_job(ch, method, properties, body):
    job = json.loads(body)
    print(f"Processing job: {job}")
    # Your RL training logic here
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='rl_jobs', on_message_callback=process_job)
channel.start_consuming()
```

## Common Patterns

### RL Experiment Logging

Store experiment metrics in ClickHouse for analysis:

```sql
-- Create experiments table
CREATE TABLE experiments (
    experiment_id UUID,
    name String,
    algorithm String,
    environment String,
    episode Int32,
    step Int32,
    reward Float64,
    loss Float64,
    epsilon Float64,
    timestamp DateTime
) ENGINE = MergeTree()
ORDER BY (experiment_id, episode, step);

-- Analyze performance
SELECT 
    name,
    algorithm,
    AVG(reward) as avg_reward,
    MAX(reward) as max_reward
FROM experiments 
WHERE episode > 100  -- Skip initial episodes
GROUP BY name, algorithm
ORDER BY avg_reward DESC;
```

### Job Queue Management

Use RabbitMQ for distributing training jobs:

```python
# Job producer
def submit_training_job(algorithm, env, params):
    job = {
        'id': str(uuid.uuid4()),
        'algorithm': algorithm,
        'environment': env,
        'parameters': params,
        'created_at': datetime.now().isoformat()
    }
    
    channel.basic_publish(
        exchange='training',
        routing_key=f'jobs.{algorithm}',
        body=json.dumps(job)
    )

# Job consumer
def train_model(job_data):
    # Load environment
    env = gym.make(job_data['environment'])
    
    # Initialize algorithm
    if job_data['algorithm'] == 'dqn':
        agent = DQNAgent(env, **job_data['parameters'])
    
    # Train and log to ClickHouse
    for episode in range(job_data.get('episodes', 1000)):
        reward = agent.train_episode()
        log_to_clickhouse(job_data['id'], episode, reward)
```
