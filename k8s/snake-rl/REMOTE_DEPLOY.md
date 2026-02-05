# Remote Kubernetes Cluster Deployment

Инструкция для деплоя Snake RL на удаленный Kubernetes кластер с приватным Docker Registry.

## Требования

- Удаленный хост с Kubernetes кластером
- Docker Registry на том же хосте (порт 5000)
- Локальный kubectl настроенный для работы с удаленным кластером
- SSH доступ к удаленному хосту

## Быстрый старт

### 1. Установите переменные окружения

```bash
# IP адрес удаленного хоста с K8s кластером и registry
export REMOTE_REGISTRY_HOST=192.168.88.253

# (Опционально) Имя пользователя для SSH
export REMOTE_USER=your_username  # По умолчанию: $USER
```

### 2. Запустите деплой

```bash
./scripts/deploy-to-remote.sh
```

Скрипт автоматически:
1. Сгенерирует `values-remote.yaml` из template
2. Скопирует проект на удаленный хост
3. Соберет Docker образы на удаленном хосте
4. Запушит образы в registry
5. Задеплоит Helm chart в кластер

### 3. Мониторинг

```bash
# Следить за статусом job
kubectl get jobs -w

# Логи Clock service (основной координатор)
kubectl logs job/snake-rl -c clock -f

# Логи всех контейнеров
kubectl logs job/snake-rl -c env        # Environment
kubectl logs job/snake-rl -c inference  # Inference
```

## Ручной деплой

### Шаг 1: Сгенерировать values-remote.yaml

```bash
export REMOTE_REGISTRY_HOST=192.168.88.253
envsubst < k8s/snake-rl/values-remote.yaml.template > k8s/snake-rl/values-remote.yaml
```

### Шаг 2: Собрать образы на удаленном хосте

```bash
ssh user@192.168.88.253

cd ~/snake_rl/services

# Базовый образ
docker build -t 192.168.88.253:5000/snake-rl/base:latest -f Dockerfile .

# Сервисы
docker build -t 192.168.88.253:5000/snake-rl/clock:latest -f Clock/Dockerfile Clock/
docker build -t 192.168.88.253:5000/snake-rl/env:latest -f Env/Dockerfile Env/
docker build -t 192.168.88.253:5000/snake-rl/inference:latest -f Inference/Dockerfile Inference/

# Push в registry
docker push 192.168.88.253:5000/snake-rl/clock:latest
docker push 192.168.88.253:5000/snake-rl/env:latest
docker push 192.168.88.253:5000/snake-rl/inference:latest
```

### Шаг 3: Деплой с Helm

```bash
# Локально (с настроенным kubectl)
helm install snake-rl k8s/snake-rl/ -f k8s/snake-rl/values-remote.yaml

# Мониторинг
kubectl get pods -w
kubectl logs job/snake-rl -c clock -f
```

## Настройка параметров эксперимента

Отредактируйте `values-remote.yaml.template`:

```yaml
experiment:
  maxEpisodes: 3000  # Количество эпизодов

env:
  gridWidth: 11      # Размер поля
  gridHeight: 11

inference:
  learningRate: 0.001  # Learning rate
  gamma: 0.9           # Discount factor
  beta: 0.1            # Entropy coefficient
```

После изменений пересоздайте `values-remote.yaml` через `envsubst`.

## Получение результатов

### Логи

```bash
# Все логи Clock service
kubectl logs job/snake-rl -c clock > experiment.log

# Логи в реальном времени
kubectl logs job/snake-rl -c clock -f
```

### Model checkpoints

```bash
# Список файлов
kubectl exec -it job/snake-rl -c clock -- ls -la /logs/

# Скопировать checkpoint
kubectl cp snake-rl:/logs/model_checkpoint_episode_3000.pth ./model.pth
```

### MLflow данные

```bash
# Скопировать все логи
kubectl cp snake-rl:/logs ./experiment-results/
```

## Очистка

```bash
# Удалить job
helm uninstall snake-rl

# Удалить PVC (опционально, удалит логи)
kubectl delete pvc snake-rl-logs

# Удалить сгенерированный values файл
rm k8s/snake-rl/values-remote.yaml
```

## Troubleshooting

### Ошибка: ImagePullBackOff

```bash
# Проверьте доступность registry
curl http://192.168.88.253:5000/v2/_catalog

# Проверьте, что образы запушены
ssh user@192.168.88.253
docker images | grep snake-rl
```

### Ошибка: Pods not starting

```bash
# Проверьте события
kubectl describe job snake-rl
kubectl describe pod <pod-name>

# Проверьте ресурсы кластера
kubectl top nodes
```

### Storage issues

```bash
# Проверьте storage class
kubectl get storageclass

# Проверьте PVC
kubectl get pvc
kubectl describe pvc snake-rl-logs
```

## Безопасность

- ⚠️ **values-remote.yaml** добавлен в `.gitignore` и НЕ должен коммититься
- ✅ Используйте **values-remote.yaml.template** для изменений
- ✅ IP адрес хранится в переменной окружения `REMOTE_REGISTRY_HOST`
