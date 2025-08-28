import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional

from .logger import logger

class QueryCache:
    """Система кэширования для RAG запросов"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache_file = os.path.join(cache_dir, "query_cache.json")
        self.cache = {}
        
        # Создаем директорию для кэша
        os.makedirs(cache_dir, exist_ok=True)
        
        # Загружаем существующий кэш
        self.load_cache()
    
    def _generate_cache_key(self, question: str, quality_threshold: float, max_iterations: int) -> str:
        """Генерация ключа кэша на основе параметров запроса"""
        key_string = f"{question.lower().strip()}_{quality_threshold}_{max_iterations}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, question: str, quality_threshold: float, max_iterations: int) -> Optional[Dict]:
        """Получить результат из кэша"""
        cache_key = self._generate_cache_key(question, quality_threshold, max_iterations)
        
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            # Проверяем актуальность (например, кэш действителен 24 часа)
            cache_time = datetime.fromisoformat(cached_result['cached_at'])
            current_time = datetime.now()
            
            if (current_time - cache_time).total_seconds() < 86400:  # 24 часа
                logger.info(f"Cache hit for question: {question[:50]}...")
                return cached_result['result']
            else:
                # Удаляем устаревший кэш
                del self.cache[cache_key]
        
        logger.info(f"Cache miss for question: {question[:50]}...")
        return None
    
    def set(self, question: str, quality_threshold: float, max_iterations: int, result: Dict):
        """Сохранить результат в кэш"""
        cache_key = self._generate_cache_key(question, quality_threshold, max_iterations)
        
        # Управление размером кэша
        if len(self.cache) >= self.max_cache_size:
            # Удаляем самый старый элемент
            oldest_key = min(
                self.cache.keys(), 
                key=lambda k: self.cache[k]['cached_at']
            )
            del self.cache[oldest_key]
        
        self.cache[cache_key] = {
            'result': result,
            'cached_at': datetime.now().isoformat(),
            'question': question[:100]  # Сохраняем часть вопроса для отладки
        }
        
        self.save_cache()
        logger.info(f"Cached result for question: {question[:50]}...")
    
    def save_cache(self):
        """Сохранить кэш в файл"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def load_cache(self):
        """Загрузить кэш из файла"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} items from cache")
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            self.cache = {}
    
    def clear_cache(self):
        """Очистить весь кэш"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Получить статистику кэша"""
        if not self.cache:
            return {"total_items": 0, "cache_size": 0}
        
        cache_times = [
            datetime.fromisoformat(item['cached_at']) 
            for item in self.cache.values()
        ]
        
        return {
            "total_items": len(self.cache),
            "oldest_item": min(cache_times).isoformat(),
            "newest_item": max(cache_times).isoformat(),
            "cache_file_size": os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
        }
