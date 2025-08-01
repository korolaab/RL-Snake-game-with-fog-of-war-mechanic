"""
food.py
Логика генерации, хранения и спавна еды для SnakeGame
"""
import random

class FoodManager:
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.food_positions = set()

    def clear(self):
        """Очистить всю еду с поля"""
        self.food_positions.clear()

    def get_positions(self):
        """Получить копию текущих позиций еды"""
        return set(self.food_positions)

    def spawn_food(self, snakes, n=1):
        """
        Сгенерировать одну или несколько новых позиций еды.
        - snakes: dict или список объектов SnakeGame, у которых есть атрибут .snake
        - n: сколько новых единиц еды добавить
        """
        occupied = set()
        for snake in snakes.values() if hasattr(snakes, 'values') else snakes:
            occupied.update(snake.snake)
        occupied |= self.food_positions

        for _ in range(n):
            for _ in range(100):  # максимум 100 попыток найти свободную клетку
                pos = (random.randint(0, self.grid_width-1), random.randint(0, self.grid_height-1))
                if pos not in occupied:
                    self.food_positions.add(pos)
                    break

    def remove_food(self, pos):
        """Удалить еду с данной позиции, если есть"""
        self.food_positions.discard(pos)

