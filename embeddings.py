import numpy as np
import logging
from typing import Optional
import aiohttp
from sentence_transformers import SentenceTransformer
from config import settings

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """
    Handles embedding generation using either local models, API endpoints, or Pinecone inference.
    
    Supports:
    - Local: sentence-transformers models (default: all-MiniLM-L6-v2)
    - API: OpenAI-compatible embedding endpoints
    - Pinecone: Pinecone's integrated inference API
    """
    
    def __init__(self):
        self.provider_type = settings.EMBEDDING_PROVIDER
        self._local_model = None
        self._session = None
        
        if self.provider_type == "local":
            self._init_local_model()
        elif self.provider_type == "api":
            self._validate_api_config()
        elif self.provider_type == "pinecone":
            self._validate_pinecone_config()
        else:
            raise ValueError(f"Invalid EMBEDDING_PROVIDER: {self.provider_type}. Must be 'local', 'api', or 'pinecone'")
    
    def _init_local_model(self):
        """Initialize local sentence-transformer model"""
        try:
            logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL}")
            self._local_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Local embedding model loaded successfully (dim: {self._local_model.get_sentence_embedding_dimension()})")
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            raise RuntimeError(f"Could not initialize local embedding model: {e}")
    
    def _validate_api_config(self):
        """Validate API configuration"""
        if not settings.EMBEDDING_API_URL:
            raise ValueError("EMBEDDING_API_URL required for API provider")
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY required for API provider")
        logger.info(f"Using API embedding provider: {settings.EMBEDDING_API_URL}")
    
    def _validate_pinecone_config(self):
        """Validate Pinecone inference configuration"""
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY required for Pinecone inference")
        logger.info(f"Using Pinecone inference with model: {settings.EMBEDDING_MODEL}")
    
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text using configured provider.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector as numpy array, or None on error
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            if self.provider_type == "local":
                return await self._get_local_embedding(text)
            elif self.provider_type == "api":
                return await self._get_api_embedding(text)
            elif self.provider_type == "pinecone":
                return await self._get_pinecone_embedding(text)
            else:
                logger.error(f"Invalid embedding provider: {self.provider_type}")
                return None
        except Exception as e:
            logger.error(f"Error getting embedding: {e}", exc_info=True)
            return None
    
    async def _get_local_embedding(self, text: str) -> np.ndarray:
        """Get embedding from local sentence-transformer model"""
        if not self._local_model:
            raise RuntimeError("Local model not initialized")
        
        max_len = 512
        truncated = text[:max_len * 4]
        
        if len(text) > len(truncated):
            logger.debug(f"Truncated text from {len(text)} to {len(truncated)} chars")
        
        try:
            embedding = self._local_model.encode(
                truncated,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            embedding = embedding.astype(np.float32)
            logger.debug(f"Generated local embedding (dim: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise
    
    async def _get_pinecone_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from Pinecone's inference API.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector or None on error
        """
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        url = "https://api.pinecone.io/inference/embed"
        
        headers = {
            "Api-Key": settings.PINECONE_API_KEY,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-10"
        }
        
        data = {
            "model": settings.EMBEDDING_MODEL,
            "parameters": {
                "input_type": "passage"
            },
            "inputs": [{"text": text}]
        }
        
        try:
            async with self._session.post(
                url,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=settings.EMBEDDING_TIMEOUT)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Pinecone inference API error {response.status}: {error_text}")
                    return None
                
                result = await response.json()
                
                if not result.get("data") or len(result["data"]) == 0:
                    logger.error(f"Invalid Pinecone inference response: {result}")
                    return None
                
                embedding_list = result["data"][0].get("values")
                
                if not embedding_list or not isinstance(embedding_list, list):
                    logger.error("Invalid Pinecone inference response: missing values array")
                    return None
                
                embedding = np.array(embedding_list, dtype=np.float32)
                
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    embedding = embedding / norm
                else:
                    logger.warning("Embedding has near-zero norm, cannot normalize")
                
                logger.debug(f"Generated Pinecone embedding (dim: {len(embedding)})")
                return embedding
                
        except aiohttp.ClientError as e:
            logger.error(f"Pinecone inference API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Pinecone inference: {e}", exc_info=True)
            return None
    
    async def _get_api_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from OpenAI-compatible API"""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"
        }
        
        data = {
            "input": text,
            "model": settings.EMBEDDING_MODEL
        }
        
        try:
            async with self._session.post(
                settings.EMBEDDING_API_URL,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=settings.EMBEDDING_TIMEOUT)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Embedding API error {response.status}: {error_text}")
                    return None
                
                result = await response.json()
                
                if not result.get("data") or len(result["data"]) == 0:
                    logger.error("Invalid embedding API response: missing data")
                    return None
                
                embedding_list = result["data"][0].get("embedding")
                
                if not embedding_list or not isinstance(embedding_list, list):
                    logger.error("Invalid embedding API response: missing embedding array")
                    return None
                
                embedding = np.array(embedding_list, dtype=np.float32)
                
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    embedding = embedding / norm
                else:
                    logger.warning("Embedding has near-zero norm, cannot normalize")
                
                logger.debug(f"Generated API embedding (dim: {len(embedding)})")
                return embedding
                
        except aiohttp.ClientError as e:
            logger.error(f"Embedding API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API embedding: {e}", exc_info=True)
            return None
    
    async def close(self):
        """Close any open sessions"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed embedding API session")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._session and not self._session.closed:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass
