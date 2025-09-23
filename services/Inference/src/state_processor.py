import torch
import logging
import numpy as np


class StateProcessor:
    """Класс для обработки состояния игры в тензор для нейронной сети."""
    
    def __init__(self, vision_size=11):
        self.vision_size = vision_size  # 11x11 vision grid
        # Simplified 2-channel encoding like legacy
        self.cell_encoding = {
            'EMPTY': [0, 0],             # Empty space
            'BODY': [1, 0],              # Own body (like legacy snake)
            'FOOD': [0, 1],              # Food
            'NOT_VISIBLE': [0, 0]        # Outside vision (all zeros)
        }
        # Action encoding for last action feature (legacy compatibility)
        self.action_encoding = {
            'forward': [1, 0],
            'left': [0, 1], 
            'right': [0, 1]  # right and left both encoded as [0, 1] vs forward [1, 0]
        }
        self.last_action = 'forward'  # Track last action
    
    def process_state(self, state, action=None):
        """
        Convert game state to variable-length sequence like legacy.
        ONLY encode non-empty visible cells, skip empty cells entirely.
        Add snake length and last action features like legacy.
        
        Args:
            state (dict): Game state containing visible_cells
            action (str): Current action for tracking last action
            
        Returns:
            torch.Tensor: Variable length tensor [num_features, 2]
        """
        
        visible_cells = state.get('visible_cells', {})
        episode = state.get('episode', 'null')
        frame = state.get('frame', 'null')
        
        # Get actual snake length from environment (not just visible segments)
        snake_length = state.get('snake_length', 3)  # Default to 3 if not provided
        
        # Update last action if provided
        if action is not None:
            self.last_action = action
        
        logging.debug({'event': 'debug_visible_cells_in_process_state',
                       'episode': episode,
                       'frame': frame,
                       'visible_cells_count': len(visible_cells),
                       'actual_snake_length': snake_length})
        
        # Legacy-style: encode ALL visible cells including empty cells
        matrix = []
        
        for coord_str, cell_type in visible_cells.items():
            try:
                # Encode ALL cells like legacy
                if cell_type == 'HEAD':
                    # Skip HEAD encoding - ignore head position like legacy
                    continue
                elif cell_type == 'EMPTY':
                    matrix.append([0, 0])  # EMPTY: [0, 0]
                elif cell_type == 'BODY':
                    matrix.append([1, 0])  # BODY: [1, 0]
                elif cell_type == 'FOOD':
                    matrix.append([0, 1])  # FOOD: [0, 1]
                else:
                    # Unknown cell types default to empty
                    matrix.append([0, 0])
                    
            except (ValueError, IndexError) as e:
                logging.warning({"event": "invalid_coordinate", "coordinate": coord_str, "error": str(e)})
                continue
        
        # Add snake length feature like legacy (exponential decay encoding)
        is_alive = np.exp(-np.abs(snake_length))
        snake_length_feature = [is_alive, 1 - is_alive]
        matrix.append(snake_length_feature)
        
        # Add last action feature like legacy
        action_feature = self.action_encoding.get(self.last_action, [1, 0])  # Default to forward
        matrix.append(action_feature)
        
        # Convert to tensor (variable length like legacy)
        if matrix:
            result = torch.tensor(matrix, dtype=torch.float32)
        else:
            # If no visible objects, return minimal tensor with just snake length and action
            result = torch.tensor([snake_length_feature, action_feature], dtype=torch.float32)
        
        # Flatten the tensor for neural network input (convert [N, 2] to [N*2])
        result = result.flatten()
        
        logging.debug({"event": "processed_state_legacy_style", 
                       "visible_cells_count": len(visible_cells),
                       "non_empty_cells": len([x for x in matrix if x not in [snake_length_feature, action_feature]]),
                       "actual_snake_length": snake_length,
                       "last_action": self.last_action,
                       "tensor_shape": result.shape})
        
        return result
    
    def get_tensor_info(self, state):
        """Получение информации о тензоре без его создания."""
        visible_cells = state.get('visible_cells', {})
        filtered_cells = {k: v for k, v in visible_cells.items() if v != 'HEAD'}
        
        valid_coords = 0
        for coord_str in filtered_cells.keys():
            try:
                x, y = map(int, coord_str.split(','))
                valid_coords += 1
            except ValueError:
                continue
        
        return {
            'total_cells': len(visible_cells),
            'filtered_cells': len(filtered_cells),
            'valid_coords': valid_coords,
            'expected_tensor_shape': (max(1, valid_coords), 3)
        }
