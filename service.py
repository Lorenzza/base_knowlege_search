from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Добавляем импорт CORS
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import pickle
import os
import logging
from functools import wraps

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Константы
CHUNK_SIZE = 512
OVERLAP = 0.1
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "search_cache.pkl")
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "embeddings.pt")
MODEL_NAME = 'all-MiniLM-L6-v2'
CSV_PATH = 'papers.csv'

# Метрики
SEARCH_REQUESTS = Counter('search_requests_total', 'Total search requests')
SEARCH_LATENCY = Histogram('search_latency_seconds', 'Search request latency')
SEARCH_ERRORS = Counter('search_errors_total', 'Total search errors')

# Модели данных
class SearchQuery(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    top_k: int = Field(default=3, ge=1, le=10, description="Количество результатов")

class SearchResult(BaseModel):
    document_id: int = Field(..., description="ID документа")
    chunk_id: int = Field(..., description="ID чанка")
    score: float = Field(..., description="Релевантность")
    content: str = Field(..., description="Содержимое чанка")
    title: str = Field(..., description="Заголовок документа")

def handle_exceptions(func):
    """Декоратор для обработки исключений"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            SEARCH_ERRORS.inc()
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper

class EmbeddingSearchService:
    def __init__(self, model_name: str = MODEL_NAME):
        """Инициализация сервиса"""
        try:
            logger.info(f"Initializing EmbeddingSearchService with model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.documents: Dict[int, Dict[str, str]] = {}
            self.chunks: Dict[Tuple[int, int], str] = {}
            self.embeddings: Optional[torch.Tensor] = None
            self.chunk_mapping: Dict[Tuple[int, int], Dict] = {}
            logger.info("EmbeddingSearchService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingSearchService: {str(e)}")
            raise

    def create_chunks(self, text: str) -> List[str]:
        """Разбивает текст на чанки с перехлестом"""
        overlap_size = int(CHUNK_SIZE * OVERLAP)
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + CHUNK_SIZE
            if end < len(text):
                end += overlap_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - overlap_size
            
        return chunks

    def load_documents(self, csv_path: str = CSV_PATH, force_reload: bool = False) -> None:
        """Загружает документы и создает эмбеддинги"""
        try:
            if not force_reload and self.load_cache():
                return

            logger.info(f"Loading documents from {csv_path}")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            df = pd.read_csv(csv_path)
            required_columns = ['Title', 'Text']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")

            all_chunks = []
            for idx, row in df.iterrows():
                self.documents[idx] = {
                    'title': row['Title'],
                    'text': row['Text']
                }
                
                document_chunks = self.create_chunks(row['Text'])
                for chunk_idx, chunk in enumerate(document_chunks):
                    chunk_key = (idx, chunk_idx)
                    self.chunks[chunk_key] = chunk
                    self.chunk_mapping[chunk_key] = {
                        'document_id': idx,
                        'chunk_id': chunk_idx,
                        'title': row['Title']
                    }
                    all_chunks.append(chunk)

            logger.info(f"Created {len(all_chunks)} chunks from {len(self.documents)} documents")
            
            logger.info("Creating embeddings...")
            self.embeddings = self.model.encode(
                all_chunks,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
            self.save_cache()
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def save_cache(self) -> None:
        """Сохраняет данные в кэш"""
        try:
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
                
            cache_data = {
                'documents': self.documents,
                'chunks': self.chunks,
                'chunk_mapping': self.chunk_mapping
            }
            
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)
                
            torch.save(self.embeddings, EMBEDDINGS_FILE)
            logger.info(f"Cache saved successfully to {CACHE_DIR}")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            raise

    def load_cache(self) -> bool:
        """Загружает данные из кэша"""
        try:
            if os.path.exists(CACHE_FILE) and os.path.exists(EMBEDDINGS_FILE):
                with open(CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.documents = cache_data['documents']
                self.chunks = cache_data['chunks']
                self.chunk_mapping = cache_data['chunk_mapping']
                self.embeddings = torch.load(EMBEDDINGS_FILE)
                
                logger.info(f"Loaded {len(self.documents)} documents and {len(self.chunks)} chunks from cache")
                return True
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
        return False

    @SEARCH_LATENCY.time()
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Поиск релевантных чанков"""
        try:
            if self.embeddings is None:
                raise ValueError("Search service not initialized properly")
                
            SEARCH_REQUESTS.inc()
            
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            cos_scores = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.embeddings,
                dim=1
            )
            
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.chunks)))
            
            results = []
            chunk_keys = list(self.chunks.keys())
            
            for score, idx in zip(top_results.values, top_results.indices):
                chunk_key = chunk_keys[idx]
                mapping = self.chunk_mapping[chunk_key]
                
                results.append(
                    SearchResult(
                        document_id=mapping['document_id'],
                        chunk_id=mapping['chunk_id'],
                        score=float(score),
                        content=self.chunks[chunk_key],
                        title=mapping['title']
                    )
                )
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

# Инициализация сервиса
search_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер жизненного цикла приложения"""
    global search_service
    try:
        logger.info("Initializing search service...")
        search_service = EmbeddingSearchService()
        search_service.load_documents()
        logger.info("Search service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize search service: {str(e)}")
        raise
    finally:
        logger.info("Shutting down search service...")

# Создаем приложение FastAPI
app = FastAPI(
    title="Embedding Search Service",
    description="API для поиска по эмбеддингам документов",
    version="1.0.0",
    lifespan=lifespan
)
# Добавляем middleware для CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы
    allow_headers=["*"],  # Разрешаем все заголовки
)
@app.post("/search", response_model=List[SearchResult])
@handle_exceptions
async def search(query: SearchQuery) -> List[SearchResult]:
    """Endpoint для поиска документов"""
    if not search_service or search_service.embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Search service not initialized properly"
        )
    
    return search_service.search(query.query, query.top_k)

@app.get("/metrics")
async def metrics():
    """Endpoint для метрик Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
@handle_exceptions
async def health_check():
    """Endpoint для проверки здоровья сервиса"""
    if not search_service or search_service.embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Search service not initialized properly"
        )
    return {
        "status": "healthy",
        "service": "embedding-search",
        "documents_loaded": len(search_service.documents)
    }

if __name__ == "__main__":
    try:
        logger.info("Starting search service on http://127.0.0.1:8001")
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise