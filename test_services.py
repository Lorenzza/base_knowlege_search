import requests

# Тест поискового сервиса
search_url = "http://127.0.0.1:8001/search"
search_payload = {
    "query": "What is machine learning?",
    "top_k": 3
}

try:
    response = requests.post(search_url, json=search_payload, timeout=10)
    print("Search Service Response:")
    print(response.json())
except Exception as e:
    print(f"Search Service Error: {str(e)}")