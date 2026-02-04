#!/usr/bin/env python3
"""
Snake RL HTML State Visualizer
Interactive HTML-based visualization for Jupyter notebooks

Usage in Jupyter:
    from html_state_visualizer import HTMLSnakeVisualizer
    
    # From list of states
    states = [{'visible_cells': {...}, 'reward': 0.1, ...}, ...]
    visualizer = HTMLSnakeVisualizer(states)
    visualizer.show()  # Displays interactive HTML widget
    
    # From file
    visualizer = HTMLSnakeVisualizer.from_file('states.json')
    visualizer.show()
    
    # Quick one-liner
    HTMLSnakeVisualizer.from_file('states.json').show()
"""

import json
import numpy as np
from IPython.display import HTML, display
import base64
from collections import deque

class HTMLSnakeVisualizer:
    def __init__(self, states=None, max_frames=1000):
        self.max_frames = max_frames
        
        # State storage
        if states is None:
            states = []
        self.states = list(states)[-max_frames:]  # Keep most recent states
        
        # Visualization setup
        self.grid_size = 11
        self.vision_radius = 5
        self.setup_colors()
        
    def setup_colors(self):
        """Setup color mapping for different cell types"""
        self.color_map = {
            'EMPTY': '#FFFFFF',      # White
            'HEAD': '#006400',       # Dark green  
            'BODY': '#90EE90',       # Light green
            'OTHER_HEAD': '#0000FF', # Blue  
            'OTHER_BODY': '#00FFFF', # Cyan
            'FOOD': '#FF0000',       # Red
            'NOT_VISIBLE': '#000000' # Black
        }
        
    @classmethod
    def from_file(cls, filename):
        """Create visualizer from JSON file containing list of states"""
        try:
            with open(filename, 'r') as f:
                states = json.load(f)
            print(f"✅ Loaded {len(states)} states from {filename}")
            return cls(states)
        except FileNotFoundError:
            print(f"❌ File {filename} not found")
            return cls([])
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {filename}")
            return cls([])
            
    def parse_vision_to_grid(self, visible_cells):
        """Convert vision dictionary to 11x11 grid with diamond-shaped vision"""
        center = self.grid_size // 2  # Center at (5, 5)
        grid = []
        
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                # Calculate Manhattan distance from center
                dx = abs(x - center)
                dy = abs(y - center)
                manhattan_distance = dx + dy
                
                # Default color based on vision range
                if manhattan_distance <= self.vision_radius:
                    cell_color = self.color_map['EMPTY']  # Default visible empty
                    cell_type = 'EMPTY'
                else:
                    cell_color = self.color_map['NOT_VISIBLE']  # Outside vision
                    cell_type = 'NOT_VISIBLE'
                
                # Override with actual cell data if available
                coord_str = f"{x},{y}"
                if coord_str in visible_cells:
                    actual_type = visible_cells[coord_str]
                    cell_color = self.color_map.get(actual_type, self.color_map['EMPTY'])
                    cell_type = actual_type
                
                row.append({
                    'color': cell_color,
                    'type': cell_type,
                    'coord': f"({x},{y})"
                })
            grid.append(row)
        return grid
        
    def analyze_state(self, state):
        """Analyze a single state and return debug info"""
        visible_cells = state.get('visible_cells', {})
        
        # Count cell types
        cell_counts = {}
        for cell_type in visible_cells.values():
            cell_counts[cell_type] = cell_counts.get(cell_type, 0) + 1
            
        # Calculate tensor info
        tensor_size = len(visible_cells) * 2  # Each cell = 2 channels in your system
        diamond_cells = sum(1 for y in range(11) for x in range(11) 
                           if abs(x-5) + abs(y-5) <= 5)  # Should be 61 cells
        expected_tensor_size = diamond_cells * 2  # 61 * 2 = 122 elements
        
        return {
            'episode': state.get('episode', 'N/A'),
            'frame': state.get('frame', 'N/A'),
            'reward': state.get('reward', 0),
            'snake_length': state.get('snake_length', 'N/A'),
            'game_over': state.get('game_over', False),
            'ticks': state.get('ticks', 'N/A'),
            'visible_cells_count': len(visible_cells),
            'tensor_size': tensor_size,
            'expected_tensor_size': expected_tensor_size,
            'cell_counts': cell_counts,
            'diamond_cells': diamond_cells,
            'datetime': state.get('datetime', '')
        }
        
    def generate_html(self):
        """Generate complete HTML visualization"""
        if not self.states:
            return "<div style='color: red; font-size: 18px;'>❌ No states to visualize</div>"
        
        # Convert states to JSON for JavaScript
        js_states = []
        for i, state in enumerate(self.states):
            grid = self.parse_vision_to_grid(state.get('visible_cells', {}))
            analysis = self.analyze_state(state)
            js_states.append({
                'id': i,
                'grid': grid,
                'analysis': analysis
            })
     
        states_json = json.dumps(js_states)
        
        html = f"""
        <div id="snake-visualizer" style="font-family: 'Courier New', monospace;">
            <style>
                .visualizer-container {{
                    display: flex;
                    gap: 20px;
                    margin: 10px 0;
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #ddd;
                }}
                .vision-panel {{
                    flex: 1;
                }}
                .info-panel {{
                    flex: 1;
                    background: #ffffff;
                    padding: 15px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                    font-size: 12px;
                }}
                .grid-container {{
                    display: inline-block;
                    border: 2px solid #333;
                    border-radius: 4px;
                    background: #000;
                }}
                .grid-row {{
                    display: flex;
                    margin: 0;
                    padding: 0;
                }}
                .grid-cell {{
                    width: 25px;
                    height: 25px;
                    margin: 1px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 10px;
                    font-weight: bold;
                    border: 1px solid #666;
                }}
                .controls {{
                    margin: 10px 0;
                    text-align: center;
                }}
                .btn {{
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    margin: 0 5px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                }}
                .btn:hover {{
                    background: #0056b3;
                }}
                .btn:disabled {{
                    background: #6c757d;
                    cursor: not-allowed;
                }}
                .slider {{
                    width: 300px;
                    margin: 10px;
                }}
                .info-section {{
                    margin: 10px 0;
                    padding: 8px;
                    background: #f1f3f4;
                    border-radius: 4px;
                }}
                .critical-issue {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .success {{
                    color: #28a745;
                }}
                .warning {{
                    color: #ffc107;
                }}
            </style>
            
            <h3>🐍 Snake RL State Visualizer (HTML)</h3>
            
            <div class="controls">
                <button class="btn" onclick="playPause()">▶️ Play/Pause</button>
                <button class="btn" onclick="previousFrame()">⏮️ Previous</button>
                <button class="btn" onclick="nextFrame()">⏭️ Next</button>
                <button class="btn" onclick="resetToStart()">⏪ Reset</button>
                <br>
                <input type="range" id="frameSlider" class="slider" min="0" max="{len(self.states)-1}" value="0" onchange="setFrame(this.value)">
                <br>
                <label>Speed: </label>
                <select id="speedSelect" onchange="setSpeed()">
                    <option value="1000">Slow (1s)</option>
                    <option value="500">Medium (0.5s)</option>
                    <option value="200" selected>Fast (0.2s)</option>
                    <option value="100">Very Fast (0.1s)</option>
                </select>
            </div>
            
            <div class="visualizer-container">
                <div class="vision-panel">
                    <h4>🔍 Agent Vision (Neural Network Input)</h4>
                    <div id="vision-grid" class="grid-container">
                        <!-- Grid will be populated by JavaScript -->
                    </div>
                    <div style="margin-top: 10px; font-size: 11px;">
                        <strong>Legend:</strong>
                        <span style="background: {self.color_map['HEAD']}; color: white; padding: 2px 4px; margin: 2px;">HEAD</span>
                        <span style="background: {self.color_map['BODY']}; color: black; padding: 2px 4px; margin: 2px;">BODY</span>
                        <span style="background: {self.color_map['FOOD']}; color: white; padding: 2px 4px; margin: 2px;">FOOD</span>
                        <span style="background: {self.color_map['EMPTY']}; color: black; padding: 2px 4px; margin: 2px; border: 1px solid #ccc;">EMPTY</span>
                    </div>
                </div>
                
                <div class="info-panel">
                    <h4>📊 Debug Information</h4>
                    <div id="debug-info">
                        <!-- Debug info will be populated by JavaScript -->
                    </div>
                </div>
            </div>
            
            <script>
                const states = {states_json};
                let currentFrame = 0;
                let isPlaying = false;
                let playInterval = null;
                let playSpeed = 200;
                
                function updateDisplay() {{
                    if (currentFrame >= states.length) return;
                    
                    const state = states[currentFrame];
                    const grid = state.grid;
                    const analysis = state.analysis;
                    
                    // Update vision grid
                    let gridHtml = '';
                    for (let y = 0; y < {self.grid_size}; y++) {{
                        gridHtml += '<div class="grid-row">';
                        for (let x = 0; x < {self.grid_size}; x++) {{
                            const cell = grid[y][x];
                            gridHtml += `<div class="grid-cell" style="background-color: ${{cell.color}};" title="${{cell.type}} ${{cell.coord}}"></div>`;
                        }}
                        gridHtml += '</div>';
                    }}
                    document.getElementById('vision-grid').innerHTML = gridHtml;
                    
                    // Update debug info
                    let issues = [];
                    let issueColor = 'success';
                    
                    if (analysis.tensor_size !== analysis.expected_tensor_size) {{
                        issues.push(`❌ Tensor size mismatch: ${{analysis.tensor_size}} != ${{analysis.expected_tensor_size}}`);
                        issueColor = 'critical-issue';
                    }}
                    
                    if (analysis.cell_counts.HEAD === undefined) {{
                        issues.push('❌ HEAD missing - agent cannot see position');
                        issueColor = 'critical-issue';
                    }}
                    
                    const cellBreakdown = Object.entries(analysis.cell_counts)
                        .map(([type, count]) => `${{type}}: ${{count}}`)
                        .join(', ');
                    
                    const debugHtml = `
                        <div class="info-section">
                            <strong>🎮 Game State:</strong><br>
                            Episode: ${{analysis.episode}} | Frame: ${{analysis.frame}}<br>
                            Reward: ${{analysis.reward}} | Length: ${{analysis.snake_length}}<br>
                            Game Over: ${{analysis.game_over}} | Ticks: ${{analysis.ticks}}
                        </div>
                        
                        <div class="info-section">
                            <strong>🧠 Neural Network Input:</strong><br>
                            Visible Cells: ${{analysis.visible_cells_count}}<br>
                            Tensor Size: ${{analysis.tensor_size}} elements<br>
                            Expected: ${{analysis.expected_tensor_size}} (diamond: ${{analysis.diamond_cells}} × 2)
                        </div>
                        
                        <div class="info-section">
                            <strong>🔍 Vision Breakdown:</strong><br>
                            ${{cellBreakdown || 'No cells visible'}}
                        </div>
                        
                        <div class="info-section">
                            <strong>🚨 Status:</strong><br>
                            <span class="${{issueColor}}">
                                ${{issues.length > 0 ? issues.join('<br>') : '✅ All checks passed'}}
                            </span>
                        </div>
                        
                        <div class="info-section" style="font-size: 10px;">
                            Frame ${{currentFrame + 1}} of ${{states.length}}<br>
                            Time: ${{analysis.datetime.substring(11, 19) || 'N/A'}}
                        </div>
                    `;
                    
                    document.getElementById('debug-info').innerHTML = debugHtml;
                    document.getElementById('frameSlider').value = currentFrame;
                }}
                
                function nextFrame() {{
                    if (currentFrame < states.length - 1) {{
                        currentFrame++;
                        updateDisplay();
                    }}
                }}
                
                function previousFrame() {{
                    if (currentFrame > 0) {{
                        currentFrame--;
                        updateDisplay();
                    }}
                }}
                
                function setFrame(frame) {{
                    currentFrame = parseInt(frame);
                    updateDisplay();
                }}
                
                function resetToStart() {{
                    currentFrame = 0;
                    updateDisplay();
                }}
                
                function playPause() {{
                    if (isPlaying) {{
                        clearInterval(playInterval);
                        isPlaying = false;
                    }} else {{
                        playInterval = setInterval(() => {{
                            if (currentFrame >= states.length - 1) {{
                                clearInterval(playInterval);
                                isPlaying = false;
                                return;
                            }}
                            nextFrame();
                        }}, playSpeed);
                        isPlaying = true;
                    }}
                }}
                
                function setSpeed() {{
                    const speed = parseInt(document.getElementById('speedSelect').value);
                    playSpeed = speed;
                    if (isPlaying) {{
                        clearInterval(playInterval);
                        playInterval = setInterval(() => {{
                            if (currentFrame >= states.length - 1) {{
                                clearInterval(playInterval);
                                isPlaying = false;
                                return;
                            }}
                            nextFrame();
                        }}, playSpeed);
                    }}
                }}
                
                // Initialize
                updateDisplay();
            </script>
        </div>
        """
        
        return html
        
    def show(self):
        """Display the HTML visualizer in Jupyter"""
        if not self.states:
            display(HTML("<div style='color: red; font-size: 18px;'>❌ No states to visualize. Load states first!</div>"))
            return
            
        print(f"🎬 Displaying {len(self.states)} states in HTML visualizer")
        display(HTML(self.generate_html()))
        
    def analyze_states(self):
        """Analyze all states and show statistics"""
        if not self.states:
            print("❌ No states to analyze")
            return
            
        print(f"📊 Analyzing {len(self.states)} states...")
        
        # Collect statistics
        episodes = [s.get('episode') for s in self.states if s.get('episode') is not None]
        frames = [s.get('frame') for s in self.states if s.get('frame') is not None]
        rewards = [s.get('reward') for s in self.states if s.get('reward') is not None]
        snake_lengths = [s.get('snake_length') for s in self.states if s.get('snake_length') is not None]
        
        vision_sizes = []
        cell_type_counts = {}
        tensor_sizes = []
        
        for state in self.states:
            visible_cells = state.get('visible_cells', {})
            vision_sizes.append(len(visible_cells))
            tensor_sizes.append(len(visible_cells) * 2)  # 2 channels per cell
            
            for cell_type in visible_cells.values():
                cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
        
        # Display analysis
        analysis_html = f"""
        <div style="font-family: 'Courier New', monospace; background: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h3>📊 State Analysis Report</h3>
            
            <div style="background: white; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>📈 Basic Statistics:</strong><br>
                Total states: {len(self.states)}<br>
                Episodes: {min(episodes) if episodes else 'N/A'} - {max(episodes) if episodes else 'N/A'}<br>
                Frames: {min(frames) if frames else 'N/A'} - {max(frames) if frames else 'N/A'}<br>
                Rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, avg={np.mean(rewards):.3f}<br>
                Snake lengths: min={min(snake_lengths) if snake_lengths else 'N/A'}, max={max(snake_lengths) if snake_lengths else 'N/A'}, avg={np.mean(snake_lengths):.1f if snake_lengths else 'N/A'}
            </div>
            
            <div style="background: white; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>🧠 Neural Network Input Analysis:</strong><br>
                Vision sizes: min={min(vision_sizes)}, max={max(vision_sizes)}, avg={np.mean(vision_sizes):.1f}<br>
                Tensor sizes: min={min(tensor_sizes)}, max={max(tensor_sizes)}, avg={np.mean(tensor_sizes):.1f}<br>
                Expected tensor size: 122 elements (61 diamond cells × 2 channels)
            </div>
            
            <div style="background: white; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>🔍 Cell Type Frequencies:</strong><br>
        """
        
        if cell_type_counts:
            total_cells = sum(cell_type_counts.values())
            for cell_type, count in sorted(cell_type_counts.items(), key=lambda x: x[1], reverse=True):
                percent = (count / total_cells) * 100
                analysis_html += f"{cell_type}: {count} ({percent:.1f}%)<br>"
        
        # Issues detection
        issues = []
        if vision_sizes and len(set(vision_sizes)) > 1:
            issues.append(f"❌ Variable vision sizes: {len(set(vision_sizes))} different sizes")
        if 'HEAD' not in cell_type_counts:
            issues.append("❌ No HEAD cells found")
        if not issues:
            issues.append("✅ No major issues detected")
            
        analysis_html += f"""
            </div>
            
            <div style="background: {'#d4edda' if not any('❌' in issue for issue in issues) else '#f8d7da'}; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>🚨 Issues Detected:</strong><br>
                {'<br>'.join(issues)}
            </div>
        </div>
        """
        
        display(HTML(analysis_html))
        
        return {
            'total_states': len(self.states),
            'vision_size_stats': {'min': min(vision_sizes), 'max': max(vision_sizes), 'avg': np.mean(vision_sizes)},
            'tensor_size_stats': {'min': min(tensor_sizes), 'max': max(tensor_sizes), 'avg': np.mean(tensor_sizes)},
            'cell_type_counts': cell_type_counts,
            'issues': issues
        }


# Convenience functions
def visualize_states(states):
    """Quick HTML visualization from list of states"""
    visualizer = HTMLSnakeVisualizer(states)
    visualizer.show()
    return visualizer

def visualize_from_file(filename):
    """Load and visualize states from file"""
    visualizer = HTMLSnakeVisualizer.from_file(filename)
    visualizer.show()
    return visualizer

def analyze_file(filename):
    """Quick analysis of states from file"""
    visualizer = HTMLSnakeVisualizer.from_file(filename)
    return visualizer.analyze_states()


# ============================================================================
# JUPYTER NOTEBOOK USAGE EXAMPLES (from notebooks/ folder in project)
# ============================================================================

"""
Example 1: Quick visualization from inference logs
---------------------------------------------------
# In notebooks/debug_snake_states.ipynb

import sys
sys.path.append('../docs')  # Add docs folder to path
from html_state_visualizer import HTMLSnakeVisualizer

# Load states from inference output logs  
viz = HTMLSnakeVisualizer.from_file('../shared/inference_states.json')
viz.show()  # Interactive HTML widget appears below cell


Example 2: Analyze training run data
------------------------------------
# In notebooks/analyze_training.ipynb

import sys
sys.path.append('../docs')
from html_state_visualizer import analyze_file, visualize_from_file

# Quick analysis report
analyze_file('../shared/episode_states_2025_01_20.json')

# Then visualize if interesting
visualize_from_file('../shared/episode_states_2025_01_20.json')


Example 3: Debug specific episode issues  
-----------------------------------------
# In notebooks/episode_debugging.ipynb

import sys, json
sys.path.append('../docs')  
from html_state_visualizer import HTMLSnakeVisualizer

# Load full training data
with open('../shared/all_states.json', 'r') as f:
    all_states = json.load(f)

# Filter to problematic episode (e.g., episode 15)
episode_15_states = [s for s in all_states if s.get('episode') == 15]

# Create visualizer for specific episode
viz = HTMLSnakeVisualizer(episode_15_states)
viz.show()

# Check for tensor size issues
viz.analyze_states()


Example 4: Real-time debugging during training
----------------------------------------------  
# In notebooks/live_training_monitor.ipynb

import sys, time, json, os
sys.path.append('../docs')
from html_state_visualizer import HTMLSnakeVisualizer

def monitor_live_states(state_file='../shared/live_states.json'):
    \"\"\"Monitor states as they're written during training\"\"\"
    
    viz = HTMLSnakeVisualizer([])
    
    while True:
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    new_states = json.load(f)
                
                # Add new states to visualizer
                viz.states = new_states[-100:]  # Keep last 100 states
                
                print(f"📊 Updated: {len(viz.states)} states")
                viz.show()  # Refresh display
                
            except (json.JSONDecodeError, FileNotFoundError):
                print("⏳ Waiting for valid state data...")
                
        time.sleep(5)  # Check every 5 seconds

# Start monitoring (run in separate cell)
# monitor_live_states()


Example 5: Compare legacy vs current behavior
---------------------------------------------
# In notebooks/legacy_comparison.ipynb

import sys, json
sys.path.append('../docs')
from html_state_visualizer import HTMLSnakeVisualizer

# Load legacy reference data
legacy_states = json.load(open('../legacy_data/good_run_states.json'))
current_states = json.load(open('../shared/current_run_states.json'))

print("🔍 LEGACY BEHAVIOR:")
legacy_viz = HTMLSnakeVisualizer(legacy_states)
legacy_viz.analyze_states()
legacy_viz.show()

print("\\n" + "="*50 + "\\n")

print("🔍 CURRENT BEHAVIOR:")  
current_viz = HTMLSnakeVisualizer(current_states)
current_viz.analyze_states()
current_viz.show()


Example 6: Debug tensor dimension mismatches
--------------------------------------------
# In notebooks/tensor_debugging.ipynb

import sys, json, numpy as np
sys.path.append('../docs')
from html_state_visualizer import HTMLSnakeVisualizer

# Load problematic states
states = json.load(open('../shared/crash_states.json'))

viz = HTMLSnakeVisualizer(states)

# First analyze for issues
analysis = viz.analyze_states()

# Show tensor size variations
tensor_sizes = []
for state in states:
    visible_cells = state.get('visible_cells', {})
    tensor_sizes.append(len(visible_cells) * 2)

print(f"\\n🧠 TENSOR SIZE ANALYSIS:")
print(f"Unique sizes: {sorted(set(tensor_sizes))}")
print(f"Size distribution: {np.bincount(tensor_sizes)}")

# Visualize the problematic frames
viz.show()


Example 7: State capture from live inference
--------------------------------------------
# In notebooks/capture_live_states.ipynb

import sys, requests, json, time
sys.path.append('../docs')
from html_state_visualizer import HTMLSnakeVisualizer

def capture_live_inference(env_url='http://localhost:5000', snake_id='adam', num_frames=50):
    \"\"\"Capture states directly from running inference\"\"\"
    
    captured_states = []
    
    print(f"📡 Capturing {num_frames} states from live inference...")
    
    for i in range(num_frames):
        try:
            response = requests.get(f"{env_url}/snake/{snake_id}", timeout=2)
            if response.status_code == 200:
                state = response.json()
                captured_states.append(state)
                print(f"Captured frame {i+1}/{num_frames} - Episode: {state.get('episode')}, Frame: {state.get('frame')}")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except requests.RequestException as e:
            print(f"❌ Connection failed: {e}")
            
        time.sleep(0.2)  # Capture every 200ms
    
    # Save captured data
    with open('../shared/live_captured_states.json', 'w') as f:
        json.dump(captured_states, f, indent=2)
    
    # Visualize captured states
    if captured_states:
        viz = HTMLSnakeVisualizer(captured_states)
        viz.analyze_states()
        viz.show()
        return viz
    else:
        print("❌ No states captured")
        return None

# Capture live data (make sure inference is running first!)
# live_viz = capture_live_inference()


Example 8: File patterns and locations in project
-------------------------------------------------
# Common file locations in the project:

# Inference output logs:
# '../shared/inference_logs/'
# '../shared/models/'

# Kubernetes persistent volume data:
# '../shared/k8s_data/'

# Environment server logs:  
# '../shared/env_logs/'

# Training outputs:
# '../shared/training_outputs/'

# Historical good runs (for comparison):
# '../legacy_data/'
# '../figures/'

# Typical state file patterns:
# - snake_states_2025_01_20_15_30_45.json  # Timestamped runs
# - episode_123_states.json                # Specific episode
# - inference_adam_states.json             # Per-agent states  
# - training_batch_5_states.json           # Per-batch states
# - crash_debug_states.json                # Error debugging
# - live_states.json                       # Real-time capture
"""