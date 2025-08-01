"""
Example: Using ClickHouse for RL experiment logging
"""
import clickhouse_connect
import uuid
from datetime import datetime

class ExperimentLogger:
    def __init__(self, host='localhost', port=8123, username='default', password=''):
        self.client = clickhouse_connect.get_client(
            host=host, port=port, username=username, password=password
        )
        self.setup_tables()
    
    def setup_tables(self):
        """Create necessary tables for RL experiments"""
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id UUID,
                name String,
                algorithm String,
                environment String,
                hyperparameters String,  -- JSON string
                created_at DateTime
            ) ENGINE = MergeTree()
            ORDER BY (experiment_id, created_at)
        """)
        
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS episode_metrics (
                experiment_id UUID,
                episode Int32,
                total_reward Float64,
                episode_length Int32,
                loss Float64,
                epsilon Float64,
                timestamp DateTime
            ) ENGINE = MergeTree()
            ORDER BY (experiment_id, episode)
        """)
    
    def start_experiment(self, name, algorithm, environment, hyperparameters):
        """Start a new experiment"""
        experiment_id = str(uuid.uuid4())
        
        self.client.insert('experiments', [[
            experiment_id, name, algorithm, environment,
            str(hyperparameters), datetime.now()
        ]], column_names=[
            'experiment_id', 'name', 'algorithm', 'environment',
            'hyperparameters', 'created_at'
        ])
        
        return experiment_id
    
    def log_episode(self, experiment_id, episode, reward, length, loss=None, epsilon=None):
        """Log metrics for a single episode"""
        self.client.insert('episode_metrics', [[
            experiment_id, episode, reward, length,
            loss or 0.0, epsilon or 0.0, datetime.now()
        ]], column_names=[
            'experiment_id', 'episode', 'total_reward', 'episode_length',
            'loss', 'epsilon', 'timestamp'
        ])
    
    def get_experiment_stats(self, experiment_id):
        """Get statistics for an experiment"""
        return self.client.query(f"""
            SELECT 
                COUNT(*) as episodes,
                AVG(total_reward) as avg_reward,
                MAX(total_reward) as max_reward,
                AVG(episode_length) as avg_length
            FROM episode_metrics 
            WHERE experiment_id = '{experiment_id}'
        """).result_rows[0]

# Usage example
if __name__ == "__main__":
    logger = ExperimentLogger()
    
    # Start experiment
    exp_id = logger.start_experiment(
        name="DQN CartPole Test",
        algorithm="DQN",
        environment="CartPole-v1",
        hyperparameters={"lr": 0.001, "batch_size": 32}
    )
    
    # Simulate logging episodes
    for episode in range(100):
        reward = 200 - episode * 0.5  # Simulated improving performance
        logger.log_episode(exp_id, episode, reward, 200)
    
    # Get stats
    stats = logger.get_experiment_stats(exp_id)
    print(f"Experiment stats: {stats}")
