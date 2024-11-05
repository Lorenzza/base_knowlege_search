from typing import List, Dict, Any
import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel, ConfigDict
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import logging
import sys
import os

# Настройка расширенного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('llm_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Метрики
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total LLM inference requests')
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference request latency')

# Константы
MAX_LENGTH = 2048
TEMPERATURE = 0.5
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Добавляем определение файла модели
MODEL_PATH = os.path.join("models", MODEL_FILE)  # Полный путь к файлу модели

# Создаем директорию для моделей, если её нет
os.makedirs("models", exist_ok=True)

# Определение моделей данных
class LLMQuery(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    question: str
    context: List[Dict[str, Any]]

class LLMResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str
    llm_name: str  # Изменено с model_name
    latency: float

# Определение сервиса
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
            try:
                # Проверяем наличие файла модели
                if not os.path.exists(MODEL_PATH):
                    logger.error(f"Model file not found: {MODEL_PATH}")
                    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    model_file=MODEL_FILE,
                    model_type="mistral",
                    gpu_layers=0 if self.device == "cpu" else None,
                    context_length=MAX_LENGTH
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

            # 3. Настройка параметров модели
            logger.info("Configuring model parameters...")
            model_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "token": True  # Использует сохраненный токен
            }
            
            if self.device == "cuda":
                model_kwargs["load_in_8bit"] = True
                logger.info("Enabled 8-bit quantization for GPU")
            
            logger.info(f"Model parameters: {model_kwargs}")

            # 4. Загрузка модели
            logger.info(f"Loading model from {model_name}...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

            # 5. Перемещение модели на устройство
            if self.device == "cpu":
                logger.info("Moving model to CPU...")
                self.model = self.model.to("cpu")
            
            # 6. Создание пайплайна
            logger.info("Creating inference pipeline...")
            try:
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=MAX_LENGTH,
                    temperature=TEMPERATURE,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Pipeline created successfully")
            except Exception as e:
                logger.error(f"Error creating pipeline: {str(e)}")
                raise

            # 7. Тестовый прогон
            logger.info("Running test inference...")
            try:
                test_output = self.pipe(
                    "Test prompt",
                    max_length=20,
                    num_return_sequences=1
                )
                logger.info("Test inference successful")
            except Exception as e:
                logger.error(f"Test inference failed: {str(e)}")
                raise

            logger.info("LLM Service initialization completed successfully")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"LLM Service initialization failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise

    def create_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Создание промпта для модели"""
        logger.info("="*50)
        logger.info("Creating prompt...")
        logger.info(f"Question: {question}")
        logger.info(f"Context documents count: {len(context)}")
        
        # Логируем детали каждого документа
        context_text = ""
        for i, doc in enumerate(context):
            # Форматируем контент с ограничением длины
            content = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            
            doc_info = (
            f"\nDocument {i+1}:"
            f"\nTitle: {doc['title']}"
            f"\nContent: {content}"
            f"\nScore: {doc.get('score', 'N/A')}"
            f"\nDocument ID: {doc.get('document_id', 'N/A')}"
            f"\nChunk ID: {doc.get('chunk_id', 'N/A')}"
            )
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
        """Генерация ответа на основе контекста"""
        INFERENCE_REQUESTS.inc()
        start_time = time.time()
        
        try:
            logger.info("="*50)
            logger.info("Generating response...")
            
            prompt = self.create_prompt(question, context)
            
            logger.info("Running inference...")
            response = self.pipe(
                prompt,
                max_length=MAX_LENGTH,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]['generated_text']
            
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

app = FastAPI(title="LLM Inference Service")

# Инициализация сервиса
llm_service = None

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global llm_service
    try:
        llm_service = LLMService()
    except Exception as e:
        logger.error(f"Failed to initialize LLM Service: {str(e)}")
        raise

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
        # Проверяем, что все компоненты загружены
        components_status = {
            "tokenizer": hasattr(llm_service, 'tokenizer'),
            "model": hasattr(llm_service, 'model'),
            "pipeline": hasattr(llm_service, 'pipe')
        }
        
        # Собираем информацию об устройстве
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