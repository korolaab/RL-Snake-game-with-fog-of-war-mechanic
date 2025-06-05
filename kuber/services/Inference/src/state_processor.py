import torch
import logging


class StateProcessor:
    """Класс для обработки состояния игры в тензор для нейронной сети."""
    
    def __init__(self):
        self.cell_encoding = {
            'FOOD': [0, 0, 1],
            'BODY': [0, 1, 0],
            'OTHER_BODY': [1, 0, 0],
            'EMPTY': [0, 0, 0]  # пустые ячейки кодируем нулями
        }
    
    def process_state(self, state):
        """
        Обработка состояния в тензор для нейронной сети.
        
        Args:
            state (dict): Состояние игры содержащее visible_cells
            
        Returns:
            torch.Tensor: Тензор размером [N, 3] где N - количество видимых ячеек
        """
        visible_cells = state.get('visible_cells', {})
        
        # Исключаем HEAD из обработки
        filtered_cells = {k: v for k, v in visible_cells.items() if v != 'HEAD'}
        
        if not filtered_cells:
            logging.warning("No visible cells found (excluding HEAD)")
            return torch.zeros(1, 3)  # если нет видимых ячеек
        
        # Сортируем ячейки по координатам (x, y)
        sorted_cells = []
        for coord_str, cell_type in filtered_cells.items():
            try:
                x, y = map(int, coord_str.split(','))
                sorted_cells.append((x, y, cell_type))
            except ValueError:
                logging.warning(f"Invalid coordinate format: {coord_str}")
                continue
        
        if not sorted_cells:
            logging.warning("No valid coordinates found")
            return torch.zeros(1, 3)
        
        # Сортируем по x, затем по y
        sorted_cells.sort(key=lambda item: (item[0], item[1]))
        
        # Создаем тензор
        tensor_data = []
        for x, y, cell_type in sorted_cells:
            encoding = self.cell_encoding.get(cell_type, [0, 0, 0])
            tensor_data.append(encoding)
        
        if not tensor_data:
            return torch.zeros(1, 3)
        
        result = torch.tensor(tensor_data, dtype=torch.float32)
        logging.debug(f"Processed state: {len(sorted_cells)} cells -> tensor shape {result.shape}")
        
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
