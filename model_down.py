import requests
import os

# Создаем директорию если её нет
os.makedirs('models', exist_ok=True)

# URL модели
url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Путь для сохранения
output_path = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Скачиваем файл
print("Downloading model...")
response = requests.get(url, stream=True)
response.raise_for_status()

# Получаем размер файла
total_size = int(response.headers.get('content-length', 0))

# Сохраняем файл с отображением прогресса
block_size = 1024  # 1 Kibibyte
progress = 0

with open(output_path, 'wb') as f:
    for data in response.iter_content(block_size):
        progress += len(data)
        f.write(data)
        done = int(50 * progress / total_size)
        print(f"\rProgress: [{'=' * done}{' ' * (50-done)}] {progress}/{total_size} bytes", end='')

print("\nDownload completed!")