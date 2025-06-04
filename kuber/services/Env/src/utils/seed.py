import random
import numpy as np

# Установка random seed для воспроизводимости
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

