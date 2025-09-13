import torch
import logging


class StateProcessor:
    """Класс для обработки состояния игры в тензор для нейронной сети."""
    
    def __init__(self, vision_size=11):
        self.vision_size = vision_size  # 11x11 vision grid
        # One-hot encoding for all cell types (4 channels)
        self.cell_encoding = {
            'EMPTY': [1, 0, 0, 0],       # Channel 0: Empty space
            'BODY': [0, 1, 0, 0],        # Channel 1: Own body
            'FOOD': [0, 0, 1, 0],        # Channel 2: Food
            'OTHER_HEAD': [0, 0, 0, 1],  # Channel 3: Enemy head
            'OTHER_BODY': [0, 0, 0, 1],  # Channel 3: Enemy body (same as head)
            'NOT_VISIBLE': [0, 0, 0, 0]  # Outside vision (all zeros)
        }
    
    def process_state(self, state):
        """
        Convert game state to fixed-size grid tensor preserving spatial relationships.
        
        Args:
            state (dict): Game state containing visible_cells
            
        Returns:
            torch.Tensor: Fixed tensor shape [vision_size, vision_size, 4] = [11, 11, 4]
        """
        
        visible_cells = state.get('visible_cells', {})
        episode = state.get('episode', 'null')
        frame = state.get('frame', 'null')
        
        logging.debug({'event': 'debug_visible_cells_in_process_state',
                       'episode': episode,
                       'frame': frame,
                       'visible_cells_count': len(visible_cells)})
        
        # Initialize grid with NOT_VISIBLE everywhere
        grid = torch.zeros(self.vision_size, self.vision_size, 4, dtype=torch.float32)
        
        # Fill grid with visible cells (including HEAD for spatial context)
        for coord_str, cell_type in visible_cells.items():
            try:
                x, y = map(int, coord_str.split(','))
                
                # Validate coordinates are within vision grid
                if 0 <= x < self.vision_size and 0 <= y < self.vision_size:
                    # Get encoding for cell type
                    if cell_type == 'HEAD':
                        # HEAD gets same encoding as BODY for neural network
                        # (agent shouldn't distinguish its own head position)
                        encoding = self.cell_encoding['BODY']
                    else:
                        encoding = self.cell_encoding.get(cell_type, self.cell_encoding['EMPTY'])
                    
                    # Place encoding in grid at (y, x) - note coordinate swap for tensor indexing
                    grid[y, x] = torch.tensor(encoding, dtype=torch.float32)
                    
            except (ValueError, IndexError) as e:
                logging.warning({"event": "invalid_coordinate", "coordinate": coord_str, "error": str(e)})
                continue
        
        # Flatten grid for neural network: [11, 11, 4] -> [484]
        result = grid.flatten()
        
        logging.debug({"event": "processed_state_grid", 
                       "visible_cells_count": len(visible_cells),
                       "tensor_shape": result.shape,
                       "expected_shape": [self.vision_size * self.vision_size * 4]})
        
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
