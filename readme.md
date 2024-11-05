# QA Search & Answer System

## Описание


## 📁 Структура проекта

```
qa-system/
├── 📄 README.md
├── 🐳 Dockerfile
├── 📄 docker-compose.yml
├── 📄 requirements.txt
├── 📄 papers.csv
├── 📂 models/
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
├── 📂 cache/
│   ├── search_cache.pkl
│   └── embeddings.pt
├── 📂 notebooks/
│   └── search_and_ask.ipynb
└── 📂 src/
    ├── service.py              # Search Service (port 8001)
    ├── llm_service_mist_gguf.py  # LLM Service (port 8002)
    ├── router_service.py       # Router Service (port 8003)
    └── run_all_services.py     # Service Manager
```

## 🚀 Быстрый старт

### Локальный запуск

1. Создайте виртуальное окружение:
```bash
python -m venv search_env
source search_env/bin/activate  # Linux/Mac
search_env\Scripts\activate     # Windows
```


2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Загрузите модель:
```bash
mkdir -p models
wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

4. Запустите сервисы:
```bash
python run_all_services.py
``` 


### Docker запуск

1. Соберите образ:
```bash
docker-compose build
```


2. Запустите контейнеры:
```bash
docker-compose up
```


## 🔧 Конфигурация

### Порты сервисов:
Search Service: 8001
LLM Service: 8002
Router Service: 8003

### Переменные окружения:
```env
CHUNK_SIZE=512
OVERLAP=0.1
MAX_LENGTH=2048
TEMPERATURE=0.5
```

## 📚 API Documentation

Подробная документация API доступна по адресам:

- `http://localhost:8001/docs` - Search Service
- `http://localhost:8002/docs` - LLM Service
- `http://localhost:8003/docs` - Router Service


## 🔍 Использование

### Python пример:
```python
import requests

# Поиск документов
response = requests.post(
    "http://localhost:8001/search",
    json={"query": "What is DeepPiCar?", "top_k": 3}
)

# Получение ответа
answer = requests.post(
    "http://localhost:8003/ask",
    json={"question": "What is DeepPiCar?"}
)
```

## 📊 Мониторинг

Метрики Prometheus доступны по адресам:
- `http://localhost:8001/metrics`
- `http://localhost:8002/metrics`
- `http://localhost:8003/metrics`


## 🛠 Разработка
### Требования

- Python 3.10+
- Docker (опционально)
- 8GB+ RAM
- CUDA-compatible GPU (опционально)


### Тестирование
```bash
pytest tests/
```

## 📝 Логирование

Логи доступны:
В консоли при запуске
В файлах *.log
В Docker логах при использовании контейнеров

## 📄 License
MIT