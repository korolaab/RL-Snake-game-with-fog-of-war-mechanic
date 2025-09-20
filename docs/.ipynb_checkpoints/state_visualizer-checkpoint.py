#!/usr/bin/env python3
"""
Snake RL State Visualizer
Creates matplotlib animation from logged JSON states

Usage in Jupyter:
    %run state_visualizer.py
    
    # From list of states
    states = [{'visible_cells': {...}, 'reward': 0.1, ...}, ...]
    visualizer = SnakeStateVisualizer(states)
    visualizer.animate()
    
    # From file
    visualizer = SnakeStateVisualizer.from_file('states.json')
    visualizer.animate()
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from collections import deque
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')

class SnakeStateVisualizer:
    def __init__(self, states=None, max_frames=1000):
        self.max_frames = max_frames
        
        # State storage
        if states is None:
            states = []
        self.states = deque(states[-max_frames:], maxlen=max_frames)  # Keep most recent states
        self.current_frame = 0
        
        # Visualization setup
        self.grid_size = 11
        self.setup_colors()
        
        # Animation components
        self.fig = None
        self.ax_world = None
        self.ax_vision = None
        self.ax_info = None
        self.im_world = None
        self.im_vision = None
        
    def setup_colors(self):
        """Setup color mapping for different cell types"""
        self.color_map = {
            'EMPTY': 0,      # Black
            'HEAD': 1,       # Red
            'BODY': 2,       # Green
            'OTHER_HEAD': 3, # Blue  
            'OTHER_BODY': 4, # Cyan
            'FOOD': 5        # Yellow
        }
        
        # Create colormap
        colors = ['black', 'darkgreen', 'lightgreen', 'blue', 'cyan', 'yellow']
        self.cmap = ListedColormap(colors)
        
    @classmethod
    def from_file(cls, filename):
        """Create visualizer from JSON file containing list of states"""
        try:
            with open(filename, 'r') as f:
                states = json.load(f)
            print(f"Loaded {len(states)} states from {filename}")
            return cls(states)
        except FileNotFoundError:
            print(f"File {filename} not found")
            return cls([])
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filename}")
            return cls([])
            
    def add_states(self, new_states):
        """Add new states to the visualization"""
        self.states.extend(new_states)
        print(f"Added {len(new_states)} states, total: {len(self.states)}")
            
    def parse_vision_to_grid(self, visible_cells):
        """Convert vision dictionary to 11x11 grid"""
        vision_grid = np.zeros((self.grid_size, self.grid_size))
        
        for coord_str, cell_type in visible_cells.items():
            try:
                x, y = map(int, coord_str.split(','))
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    vision_grid[y, x] = self.color_map.get(cell_type, 0)
            except (ValueError, AttributeError):
                continue
                
        return vision_grid
        
    def create_world_grid(self, state):
        """Create full world visualization from state"""
        # For now, just show vision data as proxy for world state
        visible_cells = state.get('visible_cells', {})
        return self.parse_vision_to_grid(visible_cells)
        
    def setup_plots(self):
        """Setup matplotlib figure and subplots - Inference debugging focused"""
        self.fig = plt.figure(figsize=(12, 6))
        
        # Agent Vision (left) - What the neural network actually sees
        self.ax_vision = self.fig.add_subplot(121)
        self.ax_vision.set_title('Agent Vision Input (Neural Network Sees This)')
        self.ax_vision.set_xlabel('Vision Grid X')
        self.ax_vision.set_ylabel('Vision Grid Y')
        
        # Info panel (right) - Debugging information
        self.ax_info = self.fig.add_subplot(122)
        self.ax_info.set_title('Inference Debug Info')
        self.ax_info.axis('off')
        
        # Initialize vision grid
        empty_grid = np.zeros((self.grid_size, self.grid_size))
        
        self.im_vision = self.ax_vision.imshow(empty_grid, cmap=self.cmap,
                                             vmin=0, vmax=5, interpolation='nearest')
        
        # Add grid lines to vision
        self.ax_vision.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax_vision.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax_vision.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        self.ax_vision.set_xlim(-0.5, self.grid_size-0.5)
        self.ax_vision.set_ylim(-0.5, self.grid_size-0.5)
        
        # Add tick labels for coordinates
        self.ax_vision.set_xticks(range(self.grid_size))
        self.ax_vision.set_yticks(range(self.grid_size))
            
        # Add colorbar legend
        cbar = plt.colorbar(self.im_vision, ax=self.ax_vision, shrink=0.8)
        cbar.set_ticks(list(range(6)))
        cbar.set_ticklabels(['Empty', 'Head', 'Body', 'Other Head', 'Other Body', 'Food'])
        
        plt.tight_layout()
        
    def update_plots(self, state):
        """Update plots with inference debugging focus"""
        if state is None:
            return
            
        visible_cells = state.get('visible_cells', {})
        
        # Update only vision grid (what neural network sees)
        vision_grid = self.parse_vision_to_grid(visible_cells)
        self.im_vision.set_array(vision_grid)
        
        # Update debugging info panel
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        episode = state.get('episode', 'N/A')
        frame = state.get('frame', 'N/A') 
        reward = state.get('reward', 0)
        game_over = state.get('game_over', False)
        snake_id = state.get('snake_id', 'unknown')
        datetime_str = state.get('datetime', '')
        
        # Count cell types in vision
        cell_counts = {}
        for cell_type in visible_cells.values():
            cell_counts[cell_type] = cell_counts.get(cell_type, 0) + 1
            
        # Calculate tensor size (the CRITICAL debugging info)
        tensor_size = len(visible_cells) * 3  # Each cell = 3 channels
        
        info_text = f"""🔍 INFERENCE DEBUG

Agent State:
  Snake ID: {snake_id}
  Episode: {episode}
  Frame: {frame}
  Reward: {reward:.3f}
  Game Over: {game_over}

⚠️ Neural Network Input:
  Visible Cells: {len(visible_cells)}
  Tensor Size: {tensor_size} elements
  Expected: 363 elements (11×11×3)
  
Vision Breakdown:"""
        
        for cell_type, count in sorted(cell_counts.items()):
            info_text += f"\n  {cell_type}: {count}"
            
        # Add critical warnings
        if len(set([len(s.get('visible_cells', {})) for s in self.states])) > 1:
            info_text += "\n\n🚨 CRITICAL ISSUE:"
            info_text += "\n  Variable tensor sizes detected!"
            info_text += "\n  This will crash the neural network!"
            
        if 'HEAD' not in cell_counts:
            info_text += "\n\n❌ HEAD MISSING:"
            info_text += "\n  Agent cannot see its own position!"
            
        # Add frame progress
        info_text += f"\n\nAnimation: {self.current_frame + 1}/{len(self.states)}"
        
        if datetime_str:
            info_text += f"\nTime: {datetime_str[11:19]}"  # Just HH:MM:SS
            
        self.ax_info.text(0.02, 0.98, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace')
        
    def animate_frame(self, frame_num):
        """Animation function for matplotlib"""
        if frame_num < len(self.states):
            self.current_frame = frame_num
            state = self.states[frame_num]
            self.update_plots(state)
        return [self.im_vision]
        
    def animate(self, interval=200, repeat=True):
        """Start animation of loaded states"""
        if not self.states:
            print("❌ No states to animate. Load states first.")
            return None
            
        print(f"✅ Animating {len(self.states)} states...")
        
        # Setup plots
        self.setup_plots()
        
        # Enable interactive mode for Jupyter
        plt.ion()
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=len(self.states),
            interval=interval, blit=False, repeat=repeat, cache_frame_data=False
        )
        
        # For Jupyter notebook compatibility
        try:
            from IPython.display import HTML
            display(HTML(anim.to_jshtml()))
        except ImportError:
            # Fallback for non-Jupyter environments
            plt.show()
        
        return anim
        
    def save_states(self, filename='snake_states.json'):
        """Save collected states to file"""
        states_list = list(self.states)
        with open(filename, 'w') as f:
            json.dump(states_list, f, indent=2)
        print(f"Saved {len(states_list)} states to {filename}")
        
    def analyze_states(self):
        """Analyze loaded states and show statistics"""
        if not self.states:
            print("No states to analyze")
            return
            
        states_list = list(self.states)
        
        # Basic statistics
        episodes = [s.get('episode') for s in states_list if s.get('episode') is not None]
        frames = [s.get('frame') for s in states_list if s.get('frame') is not None]
        rewards = [s.get('reward') for s in states_list if s.get('reward') is not None]
        
        print(f"📊 State Analysis:")
        print(f"Total states: {len(states_list)}")
        
        if episodes:
            print(f"Episodes: {min(episodes)} - {max(episodes)} (span: {max(episodes) - min(episodes) + 1})")
        if frames:
            print(f"Frames: {min(frames)} - {max(frames)}")
        if rewards:
            print(f"Rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, avg={np.mean(rewards):.3f}")
            
        # Vision statistics
        vision_sizes = []
        cell_type_counts = {}
        
        for state in states_list:
            visible_cells = state.get('visible_cells', {})
            vision_sizes.append(len(visible_cells))
            
            for cell_type in visible_cells.values():
                cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
                
        if vision_sizes:
            print(f"\n🔍 Vision Analysis:")
            print(f"Vision sizes: min={min(vision_sizes)}, max={max(vision_sizes)}, avg={np.mean(vision_sizes):.1f}")
            print(f"⚠️  CRITICAL: Variable vision sizes detected! This will cause neural network crashes.")
            
        if cell_type_counts:
            print("\n📊 Cell type frequencies:")
            total_cells = sum(cell_type_counts.values())
            for cell_type, count in sorted(cell_type_counts.items(), key=lambda x: x[1], reverse=True):
                percent = (count / total_cells) * 100
                print(f"  {cell_type}: {count} ({percent:.1f}%)")
                
        # Check for problematic patterns
        print(f"\n🚨 Issues Detected:")
        issues = 0
        
        # Check for variable tensor sizes (critical problem)
        if vision_sizes and len(set(vision_sizes)) > 1:
            print(f"  ❌ Variable vision sizes: {len(set(vision_sizes))} different sizes detected")
            print(f"     This causes neural network dimension mismatches!")
            issues += 1
            
        # Check for missing HEAD information (critical problem)  
        if 'HEAD' not in cell_type_counts:
            print(f"  ❌ No HEAD cells found: Agent cannot see its own position")
            issues += 1
        elif cell_type_counts['HEAD'] < len(states_list) * 0.8:  # Should appear in most frames
            print(f"  ⚠️  HEAD appears in only {cell_type_counts['HEAD']} of {len(states_list)} states")
            issues += 1
            
        # Check for missing object types
        expected_types = ['HEAD', 'BODY', 'FOOD', 'EMPTY']
        missing_types = [t for t in expected_types if t not in cell_type_counts]
        if missing_types:
            print(f"  ⚠️  Missing object types: {missing_types}")
            issues += 1
            
        if issues == 0:
            print("  ✅ No major issues detected")
            
        return {
            'total_states': len(states_list),
            'episode_range': (min(episodes) if episodes else None, max(episodes) if episodes else None),
            'frame_range': (min(frames) if frames else None, max(frames) if frames else None),
            'reward_stats': {
                'min': min(rewards) if rewards else None,
                'max': max(rewards) if rewards else None,
                'avg': np.mean(rewards) if rewards else None
            },
            'vision_size_range': (min(vision_sizes) if vision_sizes else None, max(vision_sizes) if vision_sizes else None),
            'cell_type_counts': cell_type_counts,
            'issues_detected': issues
        }
        
    def show_frame(self, frame_idx):
        """Show a specific frame without animation"""
        if frame_idx >= len(self.states):
            print(f"Frame {frame_idx} not available. Max frame: {len(self.states) - 1}")
            return
            
        self.setup_plots()
        self.current_frame = frame_idx
        self.update_plots(self.states[frame_idx])
        plt.show()
        
    def show_episode_summary(self, episode_num=None):
        """Show summary of states for a specific episode"""
        if not self.states:
            print("No states available")
            return
            
        # Filter states by episode
        episode_states = []
        if episode_num is not None:
            episode_states = [s for s in self.states if s.get('episode') == episode_num]
            if not episode_states:
                available_episodes = sorted(set(s.get('episode') for s in self.states if s.get('episode') is not None))
                print(f"Episode {episode_num} not found. Available episodes: {available_episodes}")
                return
        else:
            episode_states = list(self.states)
            
        print(f"📊 Episode Summary (Episode {episode_num if episode_num else 'All'}):")
        print(f"Total frames: {len(episode_states)}")
        
        rewards = [s.get('reward', 0) for s in episode_states]
        if rewards:
            print(f"Total reward: {sum(rewards):.3f}")
            print(f"Average reward per frame: {np.mean(rewards):.3f}")
            
        # Find game over events
        game_over_frames = [i for i, s in enumerate(episode_states) if s.get('game_over')]
        if game_over_frames:
            print(f"Game over at frames: {game_over_frames}")


# Convenience functions for Jupyter
def visualize_states(states, interval=200):
    """Quick visualization from list of states"""
    visualizer = SnakeStateVisualizer(states)
    return visualizer.animate(interval=interval)

def visualize_from_file(filename, interval=200):
    """Load and visualize states from file"""
    visualizer = SnakeStateVisualizer.from_file(filename)
    if visualizer.states:
        return visualizer.animate(interval=interval)
    return None

def analyze_file(filename):
    """Quick analysis of states from file"""
    visualizer = SnakeStateVisualizer.from_file(filename)
    return visualizer.analyze_states()

if __name__ == "__main__":
    # Command line usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python state_visualizer.py <states_file.json> [interval_ms]")
        sys.exit(1)
        
    filename = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    
    print(f"Snake RL State Visualizer")
    print(f"Loading states from: {filename}")
    
    visualizer = SnakeStateVisualizer.from_file(filename)
    
    if not visualizer.states:
        print("No states loaded, exiting.")
        sys.exit(1)
        
    print(f"Loaded {len(visualizer.states)} states")
    visualizer.analyze_states()
    
    print(f"\nStarting animation (interval: {interval}ms)...")
    print("Close the plot window to exit.")
    
    anim = visualizer.animate(interval=interval)
    
    # Keep the plot open
    plt.show(block=True)