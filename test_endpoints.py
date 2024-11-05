import requests
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_health_endpoints():
    """Проверяет health endpoints всех сервисов"""
    services = [
        ("Search Service", "http://127.0.0.1:8001/health"),
        ("LLM Service", "http://127.0.0.1:8002/health"),
        ("Gateway", "http://127.0.0.1:8000/health")
    ]
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            logger.info(f"{name} health check: {response.json()}")
        except Exception as e:
            logger.error(f"{name} health check failed: {str(e)}")

def test_search_endpoint():
    """Тестирует поисковый сервис"""
    url = "http://127.0.0.1:8001/search"
    payload = {
        "query": "What is machine learning?",
        "top_k": 2
    }
    
    try:
        logger.info("Testing search endpoint...")
        response = requests.post(url, json=payload, timeout=10)
        results = response.json()
        logger.info(f"Search results: {results}")
        return results
    except Exception as e:
        logger.error(f"Search endpoint test failed: {str(e)}")
        return None

def test_llm_endpoint(context):
    """Тестирует LLM сервис"""
    if not context:
        logger.error("No context available for LLM test")
        return
        
    url = "http://127.0.0.1:8002/generate"
    payload = {
        "question": "What is machine learning?",
        "context": context
    }
    
    try:
        logger.info("Testing LLM endpoint...")
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()
        logger.info(f"LLM response: {result}")
    except Exception as e:
        logger.error(f"LLM endpoint test failed: {str(e)}")

def test_gateway_endpoint():
    """Тестирует gateway"""
    url = "http://127.0.0.1:8000/qa"
    payload = {
        "query": "What is machine learning?",
        "top_k": 2
    }
    
    try:
        logger.info("Testing gateway endpoint...")
        response = requests.post(url, json=payload, timeout=60)
        result = response.json()
        logger.info(f"Gateway response: {result}")
    except Exception as e:
        logger.error(f"Gateway test failed: {str(e)}")

if __name__ == "__main__":
    # Ждем немного, чтобы сервисы успели запуститься
    time.sleep(5)
    
    # Проверяем health endpoints
    logger.info("Checking health endpoints...")
    test_health_endpoints()
    
    # Тестируем каждый сервис по отдельности
    logger.info("\nTesting individual services...")
    search_results = test_search_endpoint()
    
    if search_results:
        test_llm_endpoint(search_results)
    
    # Тестируем gateway
    logger.info("\nTesting gateway...")
    test_gateway_endpoint()