CREATE DATABASE IF NOT EXISTS rabbitmq;
CREATE DATABASE IF NOT EXISTS raw;

-- Create RabbitMQ source table
CREATE TABLE IF NOT EXISTS rabbitmq.rl_snake_logs
(
    `row` String
)
ENGINE = RabbitMQ
SETTINGS 
 rabbitmq_host_port = 'rabbitmq.data-stack.svc.cluster.local:5672',
 rabbitmq_exchange_name = 'rl_snake_logs',
 rabbitmq_username = 'tech',
 rabbitmq_password = 'tech', 
 rabbitmq_format = 'TSV';

-- Create raw storage table
CREATE TABLE IF NOT EXISTS raw.rl_snake_logs
(
    `dt` DateTime64(9, 'UTC'),
    `experiment_name` String,
    `deploy_dt` DateTime64(3, 'UTC'),
    `container` String,
    `event` String,
    `level` String,
    `data` JSON(
        SKIP container,
        SKIP run_id,
        SKIP experiment_name,
        SKIP level,
        SKIP timestamp,
        SKIP event
    ),
    `_ingested_at` DateTime64(9, 'UTC') DEFAULT now64(9)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(dt)
ORDER BY dt
SETTINGS index_granularity = 8192;

-- Create materialized view for data processing
CREATE MATERIALIZED VIEW IF NOT EXISTS default.logs_processing_mv TO raw.rl_snake_logs
(
    `dt` DateTime64(9,'UTC'),
    `experiment_name` String,
    `deploy_dt` DateTime('UTC'),
    `container` String,
    `event` String,
    `level` String,
    `data` JSON(
        SKIP container,
        SKIP run_id,
        SKIP experiment_name,
        SKIP level,
        SKIP timestamp,
        SKIP event
    )
)
AS SELECT
    parseDateTime64BestEffort(row.timestamp, 9,'UTC') AS dt,
    row.container AS container,
    parseDateTimeBestEffort(row.run_id, 'UTC') AS deploy_dt,
    row.experiment_name AS experiment_name,
    row.event AS event,
    row.level AS level,
	row as data
FROM
(
    SELECT CAST(row, 'JSON') AS row
    FROM rabbitmq.rl_snake_logs
);

