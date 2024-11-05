from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import httpx
import logging
import sys
from typing import List, Dict, Any
import time
import json

# Настройка расширенного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('router_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация сервисов
SEARCH_SERVICE_URL = "http://127.0.0.1:8001"
LLM_SERVICE_URL = "http://127.0.0.1:8002"
SEARCH_TIMEOUT = 30.0
LLM_TIMEOUT = 300.0

# Модели данных
class Question(BaseModel):
    question: str

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

class Answer(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str
    llm_name: str
    latency: float
    sources: List[str] = []

# Создание приложения
app = FastAPI(
    title="Router Service API",
    description="API для маршрутизации запросов между сервисами поиска и LLM",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def check_service_health(client: httpx.AsyncClient, service_url: str, service_name: str) -> bool:
    """Проверка здоровья сервиса"""
    try:
        response = await client.get(f"{service_url}/health", timeout=5.0)
        is_healthy = response.status_code == 200
        logger.info(f"{service_name} health check: {'healthy' if is_healthy else 'unhealthy'}")
        return is_healthy
    except Exception as e:
        logger.error(f"{service_name} health check failed: {str(e)}")
        return False

@app.post("/ask", response_model=Answer)
async def ask_question(query: Question):
    """Обработка вопроса"""
    start_time = time.time()
    logger.info("="*50)
    logger.info(f"New request received: {query.question}")

    async with httpx.AsyncClient() as client:
        try:
            # Проверка доступности сервисов
            search_healthy = await check_service_health(client, SEARCH_SERVICE_URL, "Search service")
            llm_healthy = await check_service_health(client, LLM_SERVICE_URL, "LLM service")

            if not search_healthy or not llm_healthy:
                raise HTTPException(
                    status_code=503,
                    detail="One or more required services are unavailable"
                )

            # 1. Поиск релевантных документов
            search_payload = {
                "query": query.question,
                "top_k": 3
            }
            
            logger.info(f"Sending search request: {json.dumps(search_payload)}")
            
            try:
                search_response = await client.post(
                    f"{SEARCH_SERVICE_URL}/search",
                    json=search_payload,
                    timeout=SEARCH_TIMEOUT,
                    headers={"Content-Type": "application/json"}
                )
                search_response.raise_for_status()
                
                search_results = search_response.json()
                logger.info(f"Search results received: {len(search_results)} documents")
                
            except httpx.TimeoutException:
                logger.error("Search service timeout")
                raise HTTPException(status_code=504, detail="Search service timeout")
            except Exception as e:
                logger.error(f"Search service error: {str(e)}")
                raise HTTPException(status_code=503, detail=str(e))

            # 2. Подготовка контекста для LLM
            context_list = []
            sources = []
            
            for result in search_results:
                context_list.append(result["content"])
                if "title" in result:
                    sources.append(result["title"])

            # 3. Запрос к LLM
            llm_payload = {
                "question": query.question,
                "context": context_list
            }
            
            logger.info(f"Sending request to LLM service with payload: {json.dumps(llm_payload, indent=2)}")
            
            try:
                llm_response = await client.post(
                    f"{LLM_SERVICE_URL}/generate",
                    json=llm_payload,
                    timeout=LLM_TIMEOUT,
                    headers={"Content-Type": "application/json"}
                )
                llm_response.raise_for_status()
                
                llm_result = llm_response.json()
                logger.info(f"LLM response received: {json.dumps(llm_result, indent=2)}")
                
            except httpx.TimeoutException:
                logger.error("LLM service timeout")
                raise HTTPException(status_code=504, detail="LLM service timeout")
            except Exception as e:
                logger.error(f"LLM service error: {str(e)}")
                raise HTTPException(status_code=503, detail=str(e))

            # 4. Формирование ответа
            latency = time.time() - start_time
            answer = Answer(
                answer=llm_result["answer"],
                llm_name=llm_result.get("llm_name", "unknown"),
                latency=latency,
                sources=list(set(sources))
            )
            
            logger.info(f"Request completed in {latency:.2f} seconds")
            return answer

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    async with httpx.AsyncClient() as client:
        search_healthy = await check_service_health(client, SEARCH_SERVICE_URL, "Search service")
        llm_healthy = await check_service_health(client, LLM_SERVICE_URL, "LLM service")
        
        status = "healthy" if search_healthy and llm_healthy else "unhealthy"
        return {
            "status": status,
            "search_service": search_healthy,
            "llm_service": llm_healthy,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting router service on http://127.0.0.1:8003")
    uvicorn.run(app, host="127.0.0.1", port=8003, log_level="info")