import torch
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import os
import json
import argparse
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt

# Define the search space for hyperparameters
def get_hyperparameter_space():
    return {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
        'gamma': hp.uniform('gamma', 0.9, 0.9999),
        'beta': hp.uniform('beta', 0.01, 0.3),
        'update_interval': hp.quniform('update_interval', 1, 10, 1),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.7),
        'hidden_units_1': hp.quniform('hidden_units_1', 8, 128, 8),
        'hidden_units_2': hp.quniform('hidden_units_2', 16, 256, 8),
        # Add activation function choices
        'activation_1': hp.choice('activation_1', ['ReLU', 'Tanh', 'LeakyReLU', 'GELU', 'ELU']),
        'activation_2': hp.choice('activation_2', ['ReLU', 'Tanh', 'LeakyReLU', 'GELU', 'ELU'])
    }

def objective(params):
    """
    Objective function for hyperopt to minimize.
    Runs the main.py script with the specified parameters and returns the negative reward.
    
    Args:
        params: Dictionary of hyperparameters
    
    Returns:
        Dictionary with 'loss' (negative reward), 'status', and other info
    """
    # Convert parameters to command line arguments
    args = []
    for key, value in params.items():
        # Handle special cases for types
        if key in ['update_interval', 'hidden_units_1', 'hidden_units_2']:
            value = int(value)
        
        args.append(f"--{key}={value}")
    
    # Set a specific seed for reproducibility
    args.append("--seed=42")
    
    # Add any other fixed parameters you need
    args.append("--episodes=50")  # Run for 50 episodes per trial
    
    # Create a unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.append(f"--run_id={run_id}")
    # No render mode
    args.append(f"--no_render")
    
    # Run the training script as a subprocess
    cmd = ["python3", "main.py"] + args
    print(f"\nRunning command: {' '.join(cmd)}")
    
    try:
        # Run the training script and capture output
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Parse the results from output - assuming main.py prints "Average Score: X.XX" at the end
        for line in output.split('\n'):
            if "Average Score:" in line:
                score = float(line.split("Average Score:")[1].strip())
                print(f"Trial completed with score: {score}")
                
                # Store parameters and results
                trial_result = {
                    'params': params,
                    'avg_score': score,
                    'run_id': run_id
                }
                
                # Save trial results to a JSON file
                os.makedirs('hyperopt_results', exist_ok=True)
                with open(f'hyperopt_results/trial_{run_id}.json', 'w') as f:
                    json.dump(trial_result, f, indent=2)
                
                # Return negative reward since hyperopt minimizes
                return {'loss': -score, 'status': STATUS_OK, 'run_id': run_id}
        
        # If we didn't find the reward
        print("Warning: Couldn't find reward in output")
        return {'loss': 0, 'status': STATUS_OK, 'run_id': run_id}
        
    except subprocess.CalledProcessError as e:
        print(f"Error running training script: {e}")
        print(f"Output: {e.output}")
        return {'loss': 0, 'status': STATUS_OK, 'run_id': run_id}

def plot_results(trials):
    """Plot the results of hyperparameter optimization"""
    # Extract information from trials
    losses = [t['result']['loss'] for t in trials.trials]
    iterations = list(range(len(losses)))
    
    # Plot the convergence over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, [-loss for loss in losses], marker='o')
    plt.plot(iterations, [-min(losses[:i+1]) for i in range(len(losses))], 'r--', label='Best so far')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Hyperparameter Optimization Progress')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs('hyperopt_results', exist_ok=True)
    plt.savefig('hyperopt_results/optimization_progress.png')
    plt.close()
    
    # Create additional plots for analyzing parameter importance
    if len(losses) > 5:  # Only if we have enough data points
        try:
            plt.figure(figsize=(12, 8))
            # Extract parameters for visualization
            params_to_plot = ['learning_rate', 'gamma', 'beta', 'dropout_rate']
            rewards = [-l for l in losses]
            
            for i, param in enumerate(params_to_plot):
                plt.subplot(2, 2, i+1)
                param_values = [t['misc']['vals'][param][0] for t in trials.trials]
                
                # For learning_rate, convert from log space
                if param == 'learning_rate':
                    param_values = [np.exp(val) for val in param_values]
                    
                plt.scatter(param_values, rewards, alpha=0.7)
                plt.xlabel(param)
                plt.ylabel('Reward')
                plt.title(f'{param} vs Reward')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('hyperopt_results/parameter_analysis.png')
            plt.close()
            
            # Create plot for activation functions
            plt.figure(figsize=(10, 6))
            act_params = ['activation_1', 'activation_2']
            activations = ['ReLU', 'Tanh', 'LeakyReLU', 'GELU', 'ELU']
            
            for i, act_param in enumerate(act_params):
                plt.subplot(1, 2, i+1)
                # Group by activation type
                act_rewards = {}
                for t, reward in zip(trials.trials, rewards):
                    act_idx = t['misc']['vals'][act_param][0]
                    act_type = activations[act_idx]
                    if act_type not in act_rewards:
                        act_rewards[act_type] = []
                    act_rewards[act_type].append(reward)
                
                # Calculate mean reward per activation
                act_types = []
                mean_rewards = []
                std_rewards = []
                
                for act_type, reward_list in act_rewards.items():
                    act_types.append(act_type)
                    mean_rewards.append(np.mean(reward_list))
                    std_rewards.append(np.std(reward_list))
                
                # Sort by mean reward
                sorted_indices = np.argsort(mean_rewards)[::-1]  # Descending order
                sorted_act_types = [act_types[i] for i in sorted_indices]
                sorted_mean_rewards = [mean_rewards[i] for i in sorted_indices]
                sorted_std_rewards = [std_rewards[i] for i in sorted_indices]
                
                # Plot bar chart
                bars = plt.bar(sorted_act_types, sorted_mean_rewards, yerr=sorted_std_rewards)
                plt.xlabel('Activation Function')
                plt.ylabel('Mean Reward')
                plt.title(f'Layer {i+1} Activation Function Performance')
                plt.xticks(rotation=45)
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('hyperopt_results/activation_analysis.png')
            plt.close()
            
        except Exception as e:
            print(f"Error creating detailed plots: {e}")
    
    # Print best parameters and corresponding reward
    best_idx = np.argmin(losses)
    best_reward = -losses[best_idx]
    best_run_id = trials.trials[best_idx]['result'].get('run_id', 'unknown')
    print(f"\nBest run ID: {best_run_id}")
    print(f"Best reward: {best_reward:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for RL agent')
    parser.add_argument('--max_evals', type=int, default=100,
                        help='Maximum number of parameter combinations to try')
    parser.add_argument('--output_dir', type=str, default='hyperopt_results',
                        help='Directory to save results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume optimization from saved trials')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize or load trials
    if args.resume and os.path.exists(f'{args.output_dir}/trials.pkl'):
        try:
            import pickle
            with open(f'{args.output_dir}/trials.pkl', 'rb') as f:
                trials = pickle.load(f)
            print(f"Resuming optimization from {len(trials.trials)} previous trials")
        except Exception as e:
            print(f"Error loading previous trials: {e}")
            trials = Trials()
    else:
        trials = Trials()
    
    print(f"Starting hyperparameter optimization with {args.max_evals} evaluations...")
    print(f"Results will be saved to {args.output_dir}")
    
    # Run hyperparameter optimization
    best = fmin(
        fn=objective,
        space=get_hyperparameter_space(),
        algo=tpe.suggest,
        max_evals=args.max_evals if not args.resume else len(trials.trials) + args.max_evals,
        trials=trials
    )
    
    # Save trials for potential resuming later
    try:
        import pickle
        with open(f'{args.output_dir}/trials.pkl', 'wb') as f:
            pickle.dump(trials, f)
    except Exception as e:
        print(f"Error saving trials: {e}")
    
    # Get the best parameters
    best_params = space_eval(get_hyperparameter_space(), best)
    
    # Format best parameters for printing
    print("\nBest hyperparameters found:")
    activation_list = ['ReLU', 'Tanh', 'LeakyReLU', 'GELU', 'ELU']
    for param, value in best_params.items():
        if param in ['update_interval', 'hidden_units_1', 'hidden_units_2']:
            print(f"{param}: {int(value)}")
        elif param in ['learning_rate', 'gamma', 'beta', 'dropout_rate']:
            print(f"{param}: {value:.6f}")
        elif param.startswith('activation_'):
            print(f"{param}: {value}")
        else:
            print(f"{param}: {value}")
    
    # Save the best parameters
    with open(f'{args.output_dir}/best_params.json', 'w') as f:
        # Convert values to appropriate types for JSON
        serializable_params = {}
        for param, value in best_params.items():
            if param in ['update_interval', 'hidden_units_1', 'hidden_units_2']:
                serializable_params[param] = int(value)
            elif param.startswith('activation_'):
                serializable_params[param] = value
            else:
                serializable_params[param] = float(value)
        json.dump(serializable_params, f, indent=2)
    
    # Plot and save the results
    plot_results(trials)
    
    print(f"\nOptimization complete. Best parameters saved to {args.output_dir}/best_params.json")
    print(f"To use these parameters, run: python main.py --params_file={args.output_dir}/best_params.json")

if __name__ == "__main__":
    main()
