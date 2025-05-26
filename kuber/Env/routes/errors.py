"""
Глобальные обработчики ошибок для Snake Vision Stream API.
"""
from flask import Blueprint, jsonify

# Импортируйте документацию как строку из utils, либо определите прямо здесь
try:
    from utils.documentation import API_DOC
except ImportError:
    API_DOC = """# Snake Vision Stream API\n\n… (сюда можно добавить краткое описание) …"""

errors_bp = Blueprint('errors', __name__)


# Список эндпоинтов для справки
ENDPOINTS = [
    {"method": "GET", "path": "/snake/{snake_id}", "usage": "curl -N http://localhost:5000/snake/123"},
    {"method": "POST", "path": "/snake/{snake_id}/move", "usage": "curl -X POST http://localhost:5000/snake/123/move -H 'Content-Type: application/json' -d '{\"move\":\"left\"}'"},
    {"method": "GET", "path": "/state", "usage": "curl http://localhost:5000/state"},
    {"method": "POST", "path": "/reset", "usage": "curl -X POST http://localhost:5000/reset"},
    {"method": "GET", "path": "/", "usage": "curl http://localhost:5000/"}
]


def make_error_response(error_message, status_code=400):
    return jsonify({
        "error": error_message,
        "endpoints": ENDPOINTS,
        "documentation_markdown": API_DOC
    }), status_code


@errors_bp.app_errorhandler(400)
def bad_request(error):
    return make_error_response(getattr(error, 'description', 'Bad Request'), 400)

@errors_bp.app_errorhandler(404)
def not_found(error):
    return make_error_response(getattr(error, 'description', 'Not Found'), 404)

@errors_bp.app_errorhandler(503)
def service_unavailable(error):
    return make_error_response(getattr(error, 'description', 'Service Unavailable'), 503)

# Если хотите универсально — для всех неожиданных ошибок
@errors_bp.app_errorhandler(Exception)
def handle_exception(error):
    # Можно не раскрывать детали ошибки клиенту
    return make_error_response('Internal server error', 500)

