import logging
import sys

def setup_logger(name=__name__, level=logging.INFO):
    """
    Настраивает логгер с указанным именем и уровнем.
    По умолчанию — INFO, логирует в stdout с простым форматированием.
    Возвращает объект логгера.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

# Пример использования:
# logger = setup_logger(__name__, logging.DEBUG)
# logger.info({"message": "Logger initialized"})

