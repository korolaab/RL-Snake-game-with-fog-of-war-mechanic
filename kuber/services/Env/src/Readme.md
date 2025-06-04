# Snake Vision Stream API

## Описание

API для многопользовательской змейки с поддержкой стримов «видимости» и управления змейками через HTTP-запросы. Проект реализован на Flask.

---

## Установка и запуск

### 1. Клонирование и переход в директорию

```bash
git clone <ваш-репозиторий>
cd <папка_проекта>
```

### 2. Установка зависимостей

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Запуск сервера

```bash
python app.py
```

**Параметры запуска (опционально):**

* `--grid_width` — ширина поля
* `--grid_height` — высота поля
* `--vision_radius` — радиус видимости
* `--fps` — кадров в секунду
* `--seed` — случайное зерно

Пример:

```bash
python app.py --grid_width 20 --grid_height 20 --fps 15 --seed 42
```

---

## Структура проекта

```
project_root/
│
├── app.py                  # Точка входа
├── config.py               # Настройки и парсинг аргументов
├── requirements.txt        # Зависимости
│
├── game/
│   ├── snake_game.py       # Логика одной змейки
│   ├── food.py             # Логика еды
│   └── manager.py          # Управление состояниями
│
├── routes/
│   ├── snake.py            # API управления змейками
│   ├── state.py            # API общего состояния и сброса
│   └── errors.py           # Обработка ошибок
│
├── utils/
│   ├── logger.py           # Логирование
│   └── documentation.py    # API-документация (Markdown)
│
└── README.md
```

---

## Использование API

### Получить поток видимости змейки

```bash
curl -N http://localhost:5000/snake/<snake_id>
```

### Повернуть змейку

```bash
curl -X POST http://localhost:5000/snake/<snake_id>/move \
     -H "Content-Type: application/json" \
     -d '{"move":"left"}'
```

### Получить состояние поля

```bash
curl http://localhost:5000/state
```

### Сбросить игру

```bash
curl -X POST http://localhost:5000/reset
```

---

## Примечания

* Для многопользовательского режима используйте уникальные `snake_id` для каждой сессии/игрока.
* В случае ошибок API возвращает справку по доступным эндпоинтам и примеры.
* Пример расширенной документации смотрите в файле `utils/documentation.py`.

