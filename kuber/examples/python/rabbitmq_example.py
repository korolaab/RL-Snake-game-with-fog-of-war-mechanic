"""
Example: Using RabbitMQ for RL job distribution
"""
import pika
import json
import uuid
from datetime import datetime

class RLJobQueue:
    def __init__(self, host='localhost', port=5672, username='admin', password=''):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port, credentials=credentials)
        )
        self.channel = self.connection.channel()
        self.setup_queues()
    
    def setup_queues(self):
        """Setup queues and exchanges for RL jobs"""
        # Training jobs queue
        self.channel.queue_declare(queue='training_jobs', durable=True)
        
        # Results queue
        self.channel.queue_declare(queue='training_results', durable=True)
        
        # Priority queue for urgent jobs
        self.channel.queue_declare(
            queue='priority_jobs', 
            durable=True,
            arguments={'x-max-priority': 10}
        )
    
    def submit_training_job(self, algorithm, environment, hyperparameters, priority=0):
        """Submit a training job to the queue"""
        job = {
            'job_id': str(uuid.uuid4()),
            'algorithm': algorithm,
            'environment': environment,
            'hyperparameters': hyperparameters,
            'submitted_at': datetime.now().isoformat(),
            'status': 'queued'
        }
        
        queue = 'priority_jobs' if priority > 0 else 'training_jobs'
        
        self.channel.basic_publish(
            exchange='',
            routing_key=queue,
            body=json.dumps(job),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                priority=priority
            )
        )
        
        return job['job_id']
    
    def submit_result(self, job_id, metrics):
        """Submit training results"""
        result = {
            'job_id': job_id,
            'metrics': metrics,
            'completed_at': datetime.now().isoformat()
        }
        
        self.channel.basic_publish(
            exchange='',
            routing_key='training_results',
            body=json.dumps(result),
            properties=pika.BasicProperties(delivery_mode=2)
        )
    
    def process_jobs(self, callback):
        """Process jobs from the queue"""
        def wrapper(ch, method, properties, body):
            job = json.loads(body)
            try:
                result = callback(job)
                if result:
                    self.submit_result(job['job_id'], result)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                print(f"Job failed: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='training_jobs', on_message_callback=wrapper)
        self.channel.basic_consume(queue='priority_jobs', on_message_callback=wrapper)
        
        print("Waiting for jobs. To exit press CTRL+C")
        self.channel.start_consuming()

# Usage example
def train_agent(job):
    """Example training function"""
    print(f"Training {job['algorithm']} on {job['environment']}")
    
    # Simulate training
    import time
    time.sleep(5)  # Simulate training time
    
    # Return metrics
    return {
        'avg_reward': 195.5,
        'episodes': 1000,
        'training_time': 300
    }

if __name__ == "__main__":
    queue = RLJobQueue()
    
    # Submit some jobs
    job1 = queue.submit_training_job("DQN", "CartPole-v1", {"lr": 0.001})
    job2 = queue.submit_training_job("PPO", "CartPole-v1", {"lr": 0.0003}, priority=5)
    
    print(f"Submitted jobs: {job1}, {job2}")
    
    # Process jobs (in a real scenario, this would be in a separate worker)
    # queue.process_jobs(train_agent)
