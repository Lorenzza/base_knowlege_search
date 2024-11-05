# Базовый образ Python
FROM python:3.10-slim

# Установка рабочей директории
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements.txt
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов проекта
COPY service.py .
COPY llm_service_mist_gguf.py .
COPY router_service.py .
COPY run_all_services.py .
COPY papers.csv .

# Создание директории для моделей и кэша
RUN mkdir -p models cache

# Загрузка модели Mistral (если она не монтируется извне)
RUN mkdir -p models && \
    wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Открытие портов
EXPOSE 8001 8002 8003

# Запуск сервисов
CMD ["python", "run_all_services.py"]