import aiohttp
import asyncio
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for making requests to OpenAI-compatible LLM APIs.
    
    Handles:
    - Chat completions
    - Retry logic with exponential backoff
    - Rate limiting
    - Error handling
    """
    
    def __init__(self):
        self.api_url = settings.LLM_API_URL
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
        self.timeout = settings.LLM_TIMEOUT
        self._session = None
        
        logger.info(f"LLM client initialized: {self.model} at {self.api_url}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Optional[str]:
        """
        Query LLM with retry logic.
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message/query
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response text or None on failure
        """
        if not system_prompt or not user_prompt:
            logger.error("Empty prompt provided to LLM")
            return None
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        session = await self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM request attempt {attempt + 1}/{self.max_retries}")
                
                async with session.post(
                    self.api_url,
                    json=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get("choices") and len(result["choices"]) > 0:
                            message = result["choices"][0].get("message", {})
                            content = message.get("content")
                            
                            if content:
                                logger.debug(f"LLM response received ({len(content)} chars)")
                                return content
                            else:
                                logger.error("LLM response missing content field")
                                return None
                        else:
                            logger.error("LLM response missing choices array")
                            return None
                    
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"LLM rate limited (429): {error_text[:200]}")
                        
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_delay * (2 ** attempt)
                            logger.info(f"Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error("Max retries reached for rate limit")
                            return None
                    
                    elif response.status in [500, 502, 503, 504]:
                        error_text = await response.text()
                        logger.warning(f"LLM server error ({response.status}): {error_text[:200]}")
                        
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_delay * (2 ** attempt)
                            logger.info(f"Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error("Max retries reached for server error")
                            return None
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM API error ({response.status}): {error_text[:500]}")
                        return None
            
            except asyncio.TimeoutError:
                logger.warning(f"LLM request timeout (attempt {attempt + 1}/{self.max_retries})")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error("Max retries reached for timeout")
                    return None
            
            except aiohttp.ClientError as e:
                logger.error(f"LLM connection error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error("Max retries reached for connection error")
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected LLM error: {e}", exc_info=True)
                return None
        
        logger.error("LLM query failed after all retries")
        return None
    
    async def close(self):
        """Close session"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed LLM client session")
    
    def __del__(self):
        """Cleanup on deletion"""
