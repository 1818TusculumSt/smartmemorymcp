from pydantic_settings import BaseSettings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Configuration settings loaded from environment variables.
    
    For MCP servers, these are set in Claude Desktop's config file.
    """
    
    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "adaptive-memory"
    
    # LLM Provider Configuration
    LLM_API_URL: str
    LLM_API_KEY: str
    LLM_MODEL: str = "gpt-4o-mini"
    
    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: str = "pinecone"  # Options: "local", "api", or "pinecone"
    EMBEDDING_MODEL: str = "llama-text-embed-v2"
    EMBEDDING_API_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    
    # Memory Management Settings
    MAX_MEMORIES: int = 200
    DEDUP_THRESHOLD: float = 0.95
    MIN_CONFIDENCE: float = 0.5
    RELEVANCE_THRESHOLD: float = 0.6
    
    # Performance Settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    LLM_TIMEOUT: int = 60
    EMBEDDING_TIMEOUT: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        case_sensitive = True
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate settings and log warnings for potential issues"""
        
        # Validate embedding provider
        if self.EMBEDDING_PROVIDER not in ["local", "api", "pinecone"]:
            logger.warning(
                f"Invalid EMBEDDING_PROVIDER '{self.EMBEDDING_PROVIDER}'. "
                f"Defaulting to 'local'. Valid options: 'local', 'api', 'pinecone'"
            )
            self.EMBEDDING_PROVIDER = "local"
        
        # Validate API embedding config
        if self.EMBEDDING_PROVIDER == "api":
            if not self.EMBEDDING_API_URL:
                raise ValueError(
                    "EMBEDDING_API_URL is required when EMBEDDING_PROVIDER='api'"
                )
            if not self.EMBEDDING_API_KEY:
                raise ValueError(
                    "EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER='api'"
                )
        
        # Validate Pinecone embedding config
        if self.EMBEDDING_PROVIDER == "pinecone":
            if not self.PINECONE_API_KEY:
                raise ValueError(
                    "PINECONE_API_KEY is required when EMBEDDING_PROVIDER='pinecone'"
                )
        
        # Validate thresholds
        if not 0.0 <= self.DEDUP_THRESHOLD <= 1.0:
            logger.warning(
                f"DEDUP_THRESHOLD {self.DEDUP_THRESHOLD} out of range [0.0, 1.0]. "
                f"Defaulting to 0.95"
            )
            self.DEDUP_THRESHOLD = 0.95
        
        if not 0.0 <= self.MIN_CONFIDENCE <= 1.0:
            logger.warning(
                f"MIN_CONFIDENCE {self.MIN_CONFIDENCE} out of range [0.0, 1.0]. "
                f"Defaulting to 0.5"
            )
            self.MIN_CONFIDENCE = 0.5
        
        if not 0.0 <= self.RELEVANCE_THRESHOLD <= 1.0:
            logger.warning(
                f"RELEVANCE_THRESHOLD {self.RELEVANCE_THRESHOLD} out of range [0.0, 1.0]. "
                f"Defaulting to 0.6"
            )
            self.RELEVANCE_THRESHOLD = 0.6
        
        # Validate MAX_MEMORIES
        if self.MAX_MEMORIES < 1:
            logger.warning(
                f"MAX_MEMORIES {self.MAX_MEMORIES} is too low. Defaulting to 200"
            )
            self.MAX_MEMORIES = 200
        
        # Log configuration
        logger.info("Configuration loaded successfully")
        logger.info(f"  Embedding Provider: {self.EMBEDDING_PROVIDER}")
        logger.info(f"  Embedding Model: {self.EMBEDDING_MODEL}")
        logger.info(f"  LLM Model: {self.LLM_MODEL}")
        logger.info(f"  Pinecone Index: {self.PINECONE_INDEX_NAME}")
        logger.info(f"  Max Memories: {self.MAX_MEMORIES}")
        logger.info(f"  Dedup Threshold: {self.DEDUP_THRESHOLD}")
        logger.info(f"  Min Confidence: {self.MIN_CONFIDENCE}")
        logger.info(f"  Relevance Threshold: {self.RELEVANCE_THRESHOLD}")


# Initialize settings singleton
try:
    settings = Settings()
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    raise
