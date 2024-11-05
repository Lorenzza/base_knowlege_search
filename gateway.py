from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncio
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

# Метрики
GATEWAY_REQUESTS = Counter('gateway_requests_total', 'Total gateway requests')
GATEWAY_LATENCY = Histogram('gateway_latency_seconds', 'Gateway request latency')
SEARCH_FAILURES = Counter('search_failures_total', 'Search service failures')
LLM_FAILURES = Counter('llm_failures_total', 'LLM service failures')

# Конфигурация сервисов
SEARCH_SERVICE_URL = "http://127.0.0.1:8001"
LLM_SERVICE_URL = "http://127.0.0.1:8002"
TIMEOUT = 30.0  # секунды

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    search_latency: float
    llm_latency: float
    total_latency: float

app = FastAPI(title="QA System Gateway")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def call_search_service(query: str, top_k: int) -> Dict:
    """Вызов сервиса поиска"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SEARCH_SERVICE_URL}/search",
                json={"query": query, "top_k": top_k},
                timeout=TIMEOUT
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        SEARCH_FAILURES.inc()
        raise HTTPException(status_code=503, detail=f"Search service error: {str(e)}")

async def call_llm_service(question: str, context: List[Dict]) -> Dict:
    """Вызов LLM сервиса"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json={"question": question, "context": context},
                timeout=TIMEOUT
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        LLM_FAILURES.inc()
        raise HTTPException(status_code=503, detail=f"LLM service error: {str(e)}")

@app.post("/qa", response_model=QueryResponse)
@GATEWAY_LATENCY.time()
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Основной endpoint для обработки вопросов
    """
    GATEWAY_REQUESTS.inc()
    start_time = time.time()
    
    # Вызов сервиса поиска
    search_start = time.time()
    context = await call_search_service(request.query, request.top_k)
    search_latency = time.time() - search_start
    
    # Вызов LLM сервиса
    llm_start = time.time()
    llm_response = await call_llm_service(request.query, context)
    llm_latency = time.time() - llm_start
    
    total_latency = time.time() - start_time
    
    return QueryResponse(
        answer=llm_response["answer"],
        context=context,
        search_latency=search_latency,
        llm_latency=llm_latency,
        total_latency=total_latency
    )

@app.get("/metrics")
async def metrics():
    """
    Endpoint для метрик Prometheus
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """
    Проверка здоровья всех сервисов
    """
    try:
        async with httpx.AsyncClient() as client:
            # Проверяем search сервис
            search_health = await client.get(f"{SEARCH_SERVICE_URL}/health")
            search_health.raise_for_status()
            
            # Проверяем LLM сервис
            llm_health = await client.get(f"{LLM_SERVICE_URL}/health")
            llm_health.raise_for_status()
            
            return {
                "status": "healthy",
                "search_service": "up",
                "llm_service": "up"
            }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Some services are unavailable: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)