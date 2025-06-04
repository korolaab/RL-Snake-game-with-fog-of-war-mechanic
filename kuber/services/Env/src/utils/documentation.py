"""
API Snake Vision Stream Documentation
"""

API_DOCUMENTATION_MD = """
# API Snake Vision Stream

---

## 1. Получение видеопотока "видимости" змейки

    GET /snake/{snake_id}

**Описание**: открывает непрерывный stream (NDJSON), в котором приходят кадры видимости для змейки `snake_id`.

- Параметры пути: `snake_id` (string) — уникальный идентификатор сессии змейки.
- Ответ: 200 OK, Content-Type: application/x-ndjson

Пример curl:

```bash
curl -N http://localhost:5000/snake/123
```

---

## 2. Управление направлением змейки

    POST /snake/{snake_id}/move
    Content-Type: application/json

**Описание**: поворачивает змейку влево или вправо.

- Параметры пути: `snake_id` (string)
- Тело запроса:

```json
{ "move": "left" }      # или "right"
```

- Ответ: 200 OK

```json
{"snake_id":"123","game_over":false}
```

Пример curl:

```bash
curl -X POST http://localhost:5000/snake/123/move \
     -H "Content-Type: application/json" \
     -d '{"move":"left"}'
```

---

## 3. Текущее состояние поля

    GET /state

**Описание**: возвращает полную карту поля, позиции всех змей и еды, а также их локальные видимости и общий флаг `game_over`.

- Ответ: 200 OK

```json
{
  "grid": { "0,0":[{"type":"EMPTY"}], …, "5,5":[{"type":"HEAD","snake_id":"123"}] },
  "visions": { "123":{…}, … },
  "status": { "123":false, … },
  "global_game_over":false
}
```

---

## 4. Сброс игры

    POST /reset

**Описание**: завершает все активные стримы, очищает змей и еду, создаёт новую еду.

- Ответ: 200 OK

```json
{"message":"Game reset successfully"}
```

Пример curl:

```bash
curl -X POST http://localhost:5000/reset
```

---

## 5. Корневой эндпоинт

    GET /

**Описание**: просто проверочный ping.

- Ответ: 200 OK

```json
{"message":"Snake Vision Stream API","game_over":false}
```

---

## Пример всех эндпоинтов для подсказки при ошибке

```json
[
  {"method": "GET", "path": "/snake/{id}", "usage": "curl -N http://localhost:5000/snake/123"},
  {"method": "POST", "path": "/snake/{id}/move", "usage": "curl -X POST http://localhost:5000/snake/123/move -d '{\"move\":\"left\"}' -H 'Content-Type: application/json'"},
  {"method": "GET", "path": "/state"},
  {"method": "POST", "path": "/reset"},
  {"method": "GET", "path": "/"}
]
```
"""

ENDPOINTS_FOR_ERROR = [
    {"method": "GET", "path": "/snake/{id}", "usage": "curl -N http://localhost:5000/snake/123"},
    {"method": "POST", "path": "/snake/{id}/move", "usage": "curl -X POST http://localhost:5000/snake/123/move -d '{\"move\":\"left\"}' -H 'Content-Type: application/json'"},
    {"method": "GET", "path": "/state"},
    {"method": "POST", "path": "/reset"},
    {"method": "GET", "path": "/"}
]

