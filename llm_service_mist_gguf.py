from contextlib import asynccontextmanager
from typing import List, Dict, Any
import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel, ConfigDict
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from ctransformers import AutoModelForCausalLM
import time
import logging
import sys
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('llm_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Метрики Prometheus
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total LLM inference requests')
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference request latency')

# Константы
MAX_LENGTH = 2048
TEMPERATURE = 0.5
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = os.path.join("models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Модели данных
class LLMQuery(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    question: str
    context: List[Dict[str, Any]]

class LLMResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str
    llm_name: str  # Изменено с model_name
    latency: float

# Сервис LLM
class LLMService:
    def __init__(self, model_name: str = MODEL_NAME):
        try:
            logger.info("="*50)
            logger.info("Starting LLM Service initialization...")
            
            # Проверка CUDA
            logger.info("Checking CUDA availability...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Загрузка модели
            logger.info(f"Loading model from {model_name}...")
            
            # Проверяем существование файла
            if not os.path.exists(MODEL_FILE):
                raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")
                
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_FILE,  # Используем путь к файлу напрямую
                model_type="mistral",
                gpu_layers=0 if self.device == "cpu" else None,
                context_length=MAX_LENGTH
            )
            # Установка параметров генерации
            self.model.config.temperature = TEMPERATURE
            self.model.config.top_p = 0.95
            self.model.config.repetition_penalty = 1.15
                
            logger.info("Model loaded successfully")
            

            # Тестовый прогон
            logger.info("Running test inference...")
            test_response = self.model("Test prompt", max_new_tokens=20)
            logger.info(f"Test response: {test_response}")
            
            logger.info("LLM Service initialized successfully")

        except Exception as e:
            logger.error(f"LLM Service initialization failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise


    def create_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Создание промпта на основе контекста"""
        context_text = ""
        
        for i, doc in enumerate(context):
            doc_info = f"Processing document {i+1}: {doc['title']}"
            logger.info(doc_info)
            context_text += f"Document {i+1}:\nTitle: {doc['title']}\nContent: {doc['content']}\n\n"

        prompt = f"""You are a helpful AI assistant. Please answer the following question based strictly on the provided context. If you cannot find the answer in the context, say so.

        Context:
        {context_text}

        Question: {question}

        Answer:"""

        logger.info("Generated prompt:")
        logger.info("-"*50)
        logger.info(prompt)
        logger.info("-"*50)
        
        return prompt

    @INFERENCE_LATENCY.time()
    def generate_response(self, question: str, context: List[Dict[str, Any]]) -> LLMResponse:
        """Генерация ответа"""
        INFERENCE_REQUESTS.inc()
        start_time = time.time()
        
        try:
            logger.info("="*50)
            logger.info("Generating response...")
            
            prompt = self.create_prompt(question, context)
            
            logger.info("Running inference...")
            response = self.model(
                prompt,
                max_new_tokens=MAX_LENGTH,
                temperature=TEMPERATURE,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            answer = response.split("Answer:")[-1].strip()
            
            logger.info("Generated answer:")
            logger.info("-"*50)
            logger.info(answer)
            logger.info("-"*50)
            
            latency = time.time() - start_time
            logger.info(f"Generated response in {latency:.2f} seconds")
            
            return LLMResponse(
                answer=answer,
                llm_name=MODEL_NAME,
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

# FastAPI приложение
app = FastAPI(title="LLM Inference Service")

# Глобальный экземпляр сервиса
llm_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Код, выполняемый при запуске
    global llm_service
    try:
        logger.info("Initializing LLM Service...")
        llm_service = LLMService()
        logger.info("LLM Service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize LLM Service: {str(e)}")
        raise
    finally:
        # Код, выполняемый при завершении
        logger.info("Shutting down LLM Service...")
        if llm_service and hasattr(llm_service, 'model'):
            del llm_service.model
        logger.info("LLM Service shut down successfully")

# Обновляем создание приложения
app = FastAPI(
    title="LLM Inference Service",
    lifespan=lifespan
)

@app.post("/generate", response_model=LLMResponse)
async def generate(query: LLMQuery) -> LLMResponse:
    """Endpoint для генерации ответа"""
    logger.info(f"Received generation request: {query.question}")
    return llm_service.generate_response(query.question, query.context)

@app.get("/metrics")
async def metrics():
    """Endpoint для метрик Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """Endpoint для проверки здоровья сервиса"""
    try:
        components_status = {
            "model": hasattr(llm_service, 'model')
        }
        
        device_info = {
            "device": llm_service.device,
            "cuda_available": torch.cuda.is_available()
        }
        
        if device_info["device"] == "cuda":
            device_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
            })
        
        return {
            "status": "healthy" if all(components_status.values()) else "degraded",
            "service": "llm-inference",
            "components": components_status,
            "device_info": device_info,
            "model_name": MODEL_NAME
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return Response(
            status_code=503,
            content=str(e),
            media_type="text/plain"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)