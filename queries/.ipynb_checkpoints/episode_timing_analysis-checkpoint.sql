-- Episode Timing Analysis Query
-- Calculates average, min, max, median time between episodes for game over experiments
-- Helps identify which max_steps_without_food values affect episode duration

WITH episode_timing AS (
    SELECT
        experiment_name,
        max_steps_without_food,
        beta,
        episode,
        dt,
        -- Calculate time difference between consecutive episodes
        dt - lagInFrame(dt, 1) OVER (
            PARTITION BY experiment_name 
            ORDER BY episode
        ) AS time_between_episodes_sec
    FROM (
        SELECT
            toTimezone(dt, 'Europe/Moscow') AS dt,
            toInt64(data.snakes_lengths.adam) AS adam_length,
            toInt64(data.episode) AS episode,
            experiment_name,
            deploy_dt,
            toInt32(extractGroups(experiment_name, 'game_over_maxsteps(\\d+)_')[1]) AS max_steps_without_food,
            CASE 
                WHEN experiment_name LIKE '%beta0_lr%' THEN 0
                WHEN experiment_name LIKE '%beta0_0001_lr%' THEN 0.0001
                ELSE NULL
            END AS beta
        FROM raw.rl_snake_logs
        PREWHERE (deploy_dt == (SELECT max(deploy_dt) FROM raw.rl_snake_logs)) 
            AND event REGEXP 'game_over_res'
            AND experiment_name REGEXP '^game_over_maxsteps\\d+_beta.*'
    )
    ORDER BY experiment_name, episode
)
SELECT
    experiment_name,
    max_steps_without_food,
    beta,
    COUNT(*) as total_episodes,
    -- Time statistics in seconds
    round(avg(time_between_episodes_sec), 2) AS avg_time_between_episodes_sec,
    min(time_between_episodes_sec) AS min_time_between_episodes_sec,
    max(time_between_episodes_sec) AS max_time_between_episodes_sec,
    round(median(time_between_episodes_sec), 2) AS median_time_between_episodes_sec,
    -- Convert to minutes for readability
    round(avg(time_between_episodes_sec) / 60, 2) AS avg_time_between_episodes_min,
    round(median(time_between_episodes_sec) / 60, 2) AS median_time_between_episodes_min,
    -- Percentiles for more insights
    round(quantile(0.25)(time_between_episodes_sec), 2) AS p25_time_sec,
    round(quantile(0.75)(time_between_episodes_sec), 2) AS p75_time_sec
FROM episode_timing
WHERE time_between_episodes_sec IS NOT NULL  -- Exclude first episode (no previous episode)
GROUP BY experiment_name, max_steps_without_food, beta
ORDER BY max_steps_without_food, beta;

-- Alternative query: Overall timing statistics across all experiments
/*
SELECT
    'All Game Over Experiments' AS summary,
    COUNT(DISTINCT experiment_name) AS total_experiments,
    COUNT(*) as total_episodes,
    round(avg(time_between_episodes_sec), 2) AS overall_avg_time_sec,
    round(median(time_between_episodes_sec), 2) AS overall_median_time_sec,
    round(avg(time_between_episodes_sec) / 60, 2) AS overall_avg_time_min,
    round(median(time_between_episodes_sec) / 60, 2) AS overall_median_time_min
FROM episode_timing
WHERE time_between_episodes_sec IS NOT NULL;
*/