from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChatEmbeddings
from langchain_gigachat import GigaChat

from langchain_chroma import Chroma

from chromadb.config import Settings

import os
from dotenv import find_dotenv, load_dotenv
import getpass
import json
import re
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

load_dotenv(find_dotenv())

if "GIGACHAT_CREDENTIALS" not in os.environ:
    os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")

EMBEDDINGS = GigaChatEmbeddings(verify_ssl_certs=False)

VECTOR_STORE = Chroma(
    embedding_function=EMBEDDINGS,
    client_settings=Settings(anonymized_telemetry=False),
    persist_directory='./vector_db'
)

PROMPT = ChatPromptTemplate.from_template(
"""
Ты эксперт по вопросам криптографии с юридической квалификацией. Отвечай ТОЛЬКО на основе предоставленного контекста.
Если информации недостаточно - вежливо откажись отвечать. Сохраняй профессиональный тон.

**Роль:**
- Эксперт по криптографии
- Юрист со знаниями об криптографии

**Инструкции:**
1. Анализируй контекст из базы знаний
2. Отвечай макимально конкретно на вопрос
3. Если в контексте нет ответа: "Извините, ХЗ"
4. Для сложных вопросов разбивай ответ на пункты

**Стиль ответа:**
- Отвечай точно
- Добавляй название файла в котором найдена информация в конце ответа

**Пример ответа:**
<основной ответ из контекста>
Найдено в документе: <название документа>

Context: {context}
Question: {input}
"""
)
# "Question: {question}" for rag similarity
# "Question: {input}" for rag-chain

LLM = GigaChat(verify_ssl_certs=False, model="GigaChat-2-Max", temperature=0.1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_quality_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Класс для мониторинга производительности RAG системы"""
    
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_response_time': 0.0,
            'average_iterations': 0.0,
            'average_quality_score': 0.0,
            'quality_distribution': {'0.0-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0},
            'improvement_types': {},
            'query_history': []
        }
    
    def start_query(self) -> float:
        """Начать отслеживание времени запроса"""
        return time.time()
    
    def end_query(self, start_time: float, result: Dict) -> Dict:
        """Завершить отслеживание и обновить метрики"""
        end_time = time.time()
        response_time = end_time - start_time
        
        self.metrics['total_queries'] += 1
        if result['quality_acceptable']:
            self.metrics['successful_queries'] += 1
        
        # Обновляем средние значения
        self._update_averages(response_time, result)
        
        # Обновляем распределение качества
        self._update_quality_distribution(result['final_score'])
        
        # Анализируем типы улучшений
        self._analyze_improvements(result['verification_history'])
        
        # Сохраняем историю
        query_record = {
            'timestamp': datetime.now().isoformat(),
            'question': result['question'],
            'response_time': response_time,
            'iterations': result['iterations'],
            'final_score': result['final_score'],
            'quality_acceptable': result['quality_acceptable']
        }
        self.metrics['query_history'].append(query_record)
        
        # Логируем
        logger.info(f"Query processed: {response_time:.2f}s, {result['iterations']} iterations, score: {result['final_score']:.2f}")
        
        return {
            'response_time': response_time,
            'query_record': query_record
        }
    
    def _update_averages(self, response_time: float, result: Dict):
        """Обновить средние значения"""
        n = self.metrics['total_queries']
        
        # Средний response time
        self.metrics['average_response_time'] = (
            (self.metrics['average_response_time'] * (n - 1) + response_time) / n
        )
        
        # Среднее количество итераций
        self.metrics['average_iterations'] = (
            (self.metrics['average_iterations'] * (n - 1) + result['iterations']) / n
        )
        
        # Средняя оценка качества
        self.metrics['average_quality_score'] = (
            (self.metrics['average_quality_score'] * (n - 1) + result['final_score']) / n
        )
    
    def _update_quality_distribution(self, score: float):
        """Обновить распределение оценок качества"""
        if score < 0.5:
            self.metrics['quality_distribution']['0.0-0.5'] += 1
        elif score < 0.7:
            self.metrics['quality_distribution']['0.5-0.7'] += 1
        elif score < 0.9:
            self.metrics['quality_distribution']['0.7-0.9'] += 1
        else:
            self.metrics['quality_distribution']['0.9-1.0'] += 1
    
    def _analyze_improvements(self, verification_history: List[Dict]):
        """Анализировать типы улучшений"""
        for verification in verification_history:
            for improvement in verification.get('improvements', []):
                if improvement not in self.metrics['improvement_types']:
                    self.metrics['improvement_types'][improvement] = 0
                self.metrics['improvement_types'][improvement] += 1
    
    def get_metrics_summary(self) -> Dict:
        """Получить сводку метрик"""
        success_rate = (
            self.metrics['successful_queries'] / self.metrics['total_queries'] 
            if self.metrics['total_queries'] > 0 else 0
        )
        
        return {
            'total_queries': self.metrics['total_queries'],
            'success_rate': f"{success_rate:.2%}",
            'average_response_time': f"{self.metrics['average_response_time']:.2f}s",
            'average_iterations': f"{self.metrics['average_iterations']:.2f}",
            'average_quality_score': f"{self.metrics['average_quality_score']:.2f}",
            'quality_distribution': self.metrics['quality_distribution'],
            'top_improvements': dict(sorted(
                self.metrics['improvement_types'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])
        }
    
    def save_metrics(self, filename: str = 'rag_metrics.json'):
        """Сохранить метрики в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to {filename}")
    
    def load_metrics(self, filename: str = 'rag_metrics.json'):
        """Загрузить метрики из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)
            logger.info(f"Metrics loaded from {filename}")
        except FileNotFoundError:
            logger.warning(f"Metrics file {filename} not found, starting fresh")

class QualityAnalytics:
    """Класс для аналитики качества ответов"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def analyze_query_patterns(self) -> Dict:
        """Анализ паттернов запросов"""
        history = self.monitor.metrics['query_history']
        if not history:
            return {"message": "Недостаточно данных для анализа"}
        
        # Анализ времени суток
        hours = [datetime.fromisoformat(q['timestamp']).hour for q in history]
        hour_distribution = {}
        for hour in hours:
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
        
        # Анализ качества по времени
        quality_by_time = {}
        for query in history:
            hour = datetime.fromisoformat(query['timestamp']).hour
            if hour not in quality_by_time:
                quality_by_time[hour] = []
            quality_by_time[hour].append(query['final_score'])
        
        # Средние оценки по часам
        avg_quality_by_hour = {
            hour: sum(scores) / len(scores) 
            for hour, scores in quality_by_time.items()
        }
        
        return {
            'query_frequency_by_hour': hour_distribution,
            'average_quality_by_hour': avg_quality_by_hour,
            'peak_hours': sorted(hour_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def get_improvement_recommendations(self) -> List[str]:
        """Получить рекомендации по улучшению"""
        metrics = self.monitor.metrics
        recommendations = []
        
        # Анализ success rate
        success_rate = (
            metrics['successful_queries'] / metrics['total_queries'] 
            if metrics['total_queries'] > 0 else 0
        )
        
        if success_rate < 0.8:
            recommendations.append("Низкий success rate. Рассмотрите снижение порога качества или увеличение количества итераций.")
        
        # Анализ времени ответа
        if metrics['average_response_time'] > 10.0:
            recommendations.append("Высокое время ответа. Оптимизируйте поиск в векторной базе или используйте кэширование.")
        
        # Анализ итераций
        if metrics['average_iterations'] > 2.5:
            recommendations.append("Много итераций улучшения. Улучшите начальный промпт или настройте критерии оценки.")
        
        # Анализ распределения качества
        low_quality = metrics['quality_distribution']['0.0-0.5'] + metrics['quality_distribution']['0.5-0.7']
        total = sum(metrics['quality_distribution'].values())
        
        if total > 0 and low_quality / total > 0.3:
            recommendations.append("Много ответов низкого качества. Проверьте качество документов в базе знаний.")
        
        # Анализ частых улучшений
        top_improvements = sorted(
            metrics['improvement_types'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        if top_improvements:
            frequent_issue = top_improvements[0][0]
            recommendations.append(f"Частая проблема: '{frequent_issue}'. Обновите промпт для решения этой проблемы.")
        
        return recommendations if recommendations else ["Система работает оптимально!"]

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

class AnswerQualityVerifier:
    """Агент для проверки качества и полноты ответов"""
    
    def __init__(self, llm):
        self.llm = llm
        self.verification_prompt = ChatPromptTemplate.from_template(
            """
            Ты - эксперт по оценке качества ответов в системах криптографии и информационной безопасности.
            
            Твоя задача: оценить полноту и точность ответа на основе предоставленного контекста.
            
            **Критерии оценки:**
            1. Полнота ответа (отвечает ли на все аспекты вопроса)
            2. Точность (соответствует ли контексту документов)
            3. Релевантность (связан ли ответ с вопросом)
            4. Юридическая корректность (для нормативных актов)
            
            **Шкала оценки:**
            - 1.0: Идеальный ответ - полный, точный, юридически корректный
            - 0.9: Очень хороший ответ - незначительные недочеты
            - 0.8: Хороший ответ - есть что улучшить
            - 0.7: Удовлетворительный ответ - значительные недочеты
            - 0.6 и ниже: Неудовлетворительный ответ - требует переработки
            
            **Исходный вопрос:** {question}
            
            **Контекст из документов:**
            {context}
            
            **Полученный ответ:**
            {answer}
            
            **Инструкция:**
            Оцени ответ по шкале от 0.0 до 1.0 и дай краткое обоснование.
            Верни результат СТРОГО в JSON формате:
            {{
                "score": 0.X,
                "reasoning": "краткое обоснование оценки",
                "improvements": ["что можно улучшить", "еще одно улучшение"],
                "is_acceptable": true/false
            }}
            """
        )
    
    def verify_answer(self, question: str, context: str, answer: str) -> Dict:
        """
        Проверяет качество ответа
        
        Args:
            question: Исходный вопрос
            context: Контекст из документов
            answer: Полученный ответ
            
        Returns:
            Dict с оценкой и рекомендациями
        """
        try:
            # Формируем промпт для верификации
            verification_message = self.verification_prompt.invoke({
                "question": question,
                "context": context,
                "answer": answer
            })
            
            # Получаем оценку от LLM
            verification_response = self.llm.invoke(verification_message)
            
            # Парсим JSON ответ
            try:
                # Извлекаем JSON из ответа
                json_match = re.search(r'\{.*\}', verification_response.content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Если JSON не найден, возвращаем базовую оценку
                    result = {
                        "score": 0.5,
                        "reasoning": "Не удалось получить структурированную оценку",
                        "improvements": ["Требуется ручная проверка"],
                        "is_acceptable": False
                    }
            except json.JSONDecodeError:
                # Fallback оценка
                result = {
                    "score": 0.5,
                    "reasoning": "Ошибка парсинга оценки",
                    "improvements": ["Требуется ручная проверка"],
                    "is_acceptable": False
                }
            
            return result
            
        except Exception as e:
            print(f"Ошибка при верификации: {e}")
            return {
                "score": 0.0,
                "reasoning": f"Ошибка верификации: {str(e)}",
                "improvements": ["Требуется ручная проверка"],
                "is_acceptable": False
            }

class QualityControlledRAG:
    """RAG система с контролем качества ответов"""
    
    def __init__(self, vector_store, llm, quality_threshold=0.9, max_iterations=3, 
                 enable_monitoring=True, enable_caching=True):
        self.vector_store = vector_store
        self.llm = llm
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.verifier = AnswerQualityVerifier(llm)
        
        # Инициализация мониторинга
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = PerformanceMonitor()
            self.analytics = QualityAnalytics(self.monitor)
            self.monitor.load_metrics()  # Загружаем существующие метрики
        
        # Инициализация кэширования
        self.enable_caching = enable_caching
        if enable_caching:
            self.cache = QueryCache()
        
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Основной RAG промпт
        self.main_prompt = ChatPromptTemplate.from_template(
            """
            Ты эксперт по вопросам криптографии с юридической квалификацией. Отвечай ТОЛЬКО на основе предоставленного контекста.
            Если информации недостаточно - вежливо откажись отвечать. Сохраняй профессиональный тон.

            **Роль:**
            - Эксперт по криптографии
            - Юрист со знаниями об криптографии

            **Инструкции:**
            1. Анализируй контекст из базы знаний
            2. Отвечай максимально конкретно на вопрос
            3. Если в контексте нет ответа: "Извините, в предоставленных документах нет информации для ответа на этот вопрос"
            4. Для сложных вопросов разбивай ответ на пункты
            5. ОБЯЗАТЕЛЬНО указывай источник информации

            **Стиль ответа:**
            - Отвечай точно и полно
            - Добавляй название файла в котором найдена информация в конце ответа
            - Используй профессиональную терминологию

            **Дополнительные требования (если есть):**
            {additional_requirements}

            Context: {context}
            Question: {input}
            """
        )
        
        # Промпт для доработки
        self.refinement_prompt = ChatPromptTemplate.from_template(
            """
            Ты эксперт по криптографии. Предыдущий ответ требует доработки.

            **Исходный вопрос:** {question}
            
            **Контекст из документов:**
            {context}
            
            **Предыдущий ответ:**
            {previous_answer}
            
            **Требования для улучшения:**
            {improvements}
            
            **Инструкция:**
            Создай улучшенный ответ, учитывая замечания. Будь более полным и точным.
            Обязательно укажи источники информации.
            """
        )
    
    def get_answer_with_quality_control(self, question: str, verbose: bool = True) -> Dict:
        """
        Получает ответ с контролем качества
        
        Args:
            question: Вопрос пользователя
            verbose: Выводить ли детальную информацию
            
        Returns:
            Dict с финальным ответом и метриками
        """
        # Проверяем кэш
        if self.enable_caching:
            cached_result = self.cache.get(question, self.quality_threshold, self.max_iterations)
            if cached_result:
                self.cache_hits += 1
                if verbose:
                    print("✅ Найден результат в кэше")
                
                # Добавляем информацию о кэше
                cached_result['from_cache'] = True
                cached_result['cache_hit'] = True
                
                # Обновляем мониторинг для кэшированных результатов
                if self.enable_monitoring:
                    start_time = self.monitor.start_query()
                    monitoring_result = self.monitor.end_query(start_time, cached_result)
                    cached_result['response_time'] = monitoring_result['response_time']
                
                return cached_result
            else:
                self.cache_misses += 1
        
        # Начинаем мониторинг времени
        start_time = None
        if self.enable_monitoring:
            start_time = self.monitor.start_query()
        
        iteration = 0
        verification_history = []
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n=== Итерация {iteration} ===")
            
            # Получаем релевантные документы
            retrieved_docs = self.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Формируем дополнительные требования для доработки
            additional_requirements = ""
            if iteration > 1:
                last_verification = verification_history[-1]
                additional_requirements = f"Учти предыдущие замечания: {', '.join(last_verification['improvements'])}"
            
            # Генерируем ответ
            if iteration == 1:
                # Первая итерация - используем основной промпт
                prompt_message = self.main_prompt.invoke({
                    "context": context,
                    "input": question,
                    "additional_requirements": additional_requirements
                })
            else:
                # Последующие итерации - используем промпт для доработки
                prompt_message = self.refinement_prompt.invoke({
                    "question": question,
                    "context": context,
                    "previous_answer": current_answer,
                    "improvements": ", ".join(verification_history[-1]["improvements"])
                })
            
            current_answer = self.llm.invoke(prompt_message).content
            
            if verbose:
                print(f"Ответ: {current_answer[:200]}...")
            
            # Проверяем качество ответа
            verification = self.verifier.verify_answer(question, context, current_answer)
            verification_history.append(verification)
            
            if verbose:
                print(f"Оценка качества: {verification['score']:.2f}")
                print(f"Обоснование: {verification['reasoning']}")
            
            # Проверяем, достаточно ли качество
            if verification["score"] >= self.quality_threshold:
                if verbose:
                    print(f"✅ Качество достаточное! Оценка: {verification['score']:.2f}")
                break
            else:
                if verbose:
                    print(f"❌ Качество недостаточное. Требуется доработка.")
                    print(f"Рекомендации: {', '.join(verification['improvements'])}")
        
        # Формируем финальный результат
        final_result = {
            "question": question,
            "final_answer": current_answer,
            "iterations": iteration,
            "final_score": verification_history[-1]["score"],
            "quality_acceptable": verification_history[-1]["score"] >= self.quality_threshold,
            "verification_history": verification_history,
            "retrieved_docs_count": len(retrieved_docs),
            "from_cache": False,
            "cache_hit": False
        }
        
        # Завершаем мониторинг
        if self.enable_monitoring and start_time is not None:
            monitoring_result = self.monitor.end_query(start_time, final_result)
            final_result['response_time'] = monitoring_result['response_time']
            
            # Сохраняем метрики каждые 10 запросов
            if self.monitor.metrics['total_queries'] % 10 == 0:
                self.monitor.save_metrics()
        
        # Сохраняем в кэш только качественные ответы
        if self.enable_caching and final_result['quality_acceptable']:
            self.cache.set(question, self.quality_threshold, self.max_iterations, final_result)
        
        return final_result
    
    def get_performance_report(self) -> Dict:
        """Получить отчет о производительности"""
        if not self.enable_monitoring:
            return {"error": "Мониторинг отключен"}
        
        summary = self.monitor.get_metrics_summary()
        patterns = self.analytics.analyze_query_patterns()
        recommendations = self.analytics.get_improvement_recommendations()
        
        return {
            "summary": summary,
            "query_patterns": patterns,
            "recommendations": recommendations
        }
    
    def reset_metrics(self):
        """Сбросить метрики мониторинга"""
        if self.enable_monitoring:
            self.monitor = PerformanceMonitor()
            self.analytics = QualityAnalytics(self.monitor)
            logger.info("Metrics reset")
    
    def get_cache_stats(self) -> Dict:
        """Получить статистику кэша"""
        if not self.enable_caching:
            return {"error": "Кэширование отключено"}
        
        cache_stats = self.cache.get_cache_stats()
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            **cache_stats,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
            "total_requests": total_requests
        }
    
    def clear_cache(self):
        """Очистить кэш"""
        if self.enable_caching:
            self.cache.clear_cache()
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("Cache cleared")
    
    def get_full_report(self) -> Dict:
        """Получить полный отчет о системе"""
        report = {
            "performance": self.get_performance_report() if self.enable_monitoring else None,
            "cache": self.get_cache_stats() if self.enable_caching else None,
            "system_config": {
                "quality_threshold": self.quality_threshold,
                "max_iterations": self.max_iterations,
                "monitoring_enabled": self.enable_monitoring,
                "caching_enabled": self.enable_caching
            }
        }
        
        return report

# Примеры использования

# 1. Старый метод (для сравнения)
def old_method_example():
    question = "что такое УЦ?"
    rag_chain = create_stuff_documents_chain(LLM, PROMPT)
    retrieval_chain = create_retrieval_chain(
        VECTOR_STORE.as_retriever(
            search_kwargs={ "k": 2 },
            search_type="similarity"
        ),
        rag_chain
    )
    
    answer = retrieval_chain.invoke({"input": question})["answer"]
    print("=== СТАРЫЙ МЕТОД ===")
    print(answer)
    return answer

# 2. Новый метод с контролем качества и мониторингом
def quality_controlled_example():
    question = "что такое УЦ?"
    
    # Создаем систему с контролем качества, мониторингом и кэшированием
    quality_rag = QualityControlledRAG(
        vector_store=VECTOR_STORE,
        llm=LLM,
        quality_threshold=0.9,  # Порог качества
        max_iterations=3,       # Максимум попыток улучшения
        enable_monitoring=True, # Включаем мониторинг
        enable_caching=True     # Включаем кэширование
    )
    
    print("=== НОВЫЙ МЕТОД С КОНТРОЛЕМ КАЧЕСТВА И МОНИТОРИНГОМ ===")
    result = quality_rag.get_answer_with_quality_control(question, verbose=True)
    
    print(f"\n📊 ФИНАЛЬНЫЕ МЕТРИКИ:")
    print(f"Итераций: {result['iterations']}")
    print(f"Финальная оценка: {result['final_score']:.2f}")
    print(f"Качество приемлемое: {'✅' if result['quality_acceptable'] else '❌'}")
    print(f"Документов найдено: {result['retrieved_docs_count']}")
    if 'response_time' in result:
        print(f"Время ответа: {result['response_time']:.2f}с")
    
    print(f"\n📝 ФИНАЛЬНЫЙ ОТВЕТ:")
    print(result['final_answer'])
    
    # Получаем отчет о производительности
    performance_report = quality_rag.get_performance_report()
    print(f"\n📈 ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ:")
    print(f"Всего запросов: {performance_report['summary']['total_queries']}")
    print(f"Success rate: {performance_report['summary']['success_rate']}")
    print(f"Среднее время ответа: {performance_report['summary']['average_response_time']}")
    print(f"Средние итерации: {performance_report['summary']['average_iterations']}")
    print(f"Средняя оценка: {performance_report['summary']['average_quality_score']}")
    
    if performance_report['recommendations']:
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        for i, rec in enumerate(performance_report['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Статистика кэша
    cache_stats = quality_rag.get_cache_stats()
    if 'error' not in cache_stats:
        print(f"\n💾 СТАТИСТИКА КЭША:")
        print(f"Hit rate: {cache_stats['hit_rate']}")
        print(f"Всего элементов: {cache_stats['total_items']}")
        print(f"Cache hits: {cache_stats['cache_hits']}")
        print(f"Cache misses: {cache_stats['cache_misses']}")
    
    return result

# 3. Функция для тестирования нескольких вопросов
def batch_quality_test():
    """Тестирование системы на наборе вопросов"""
    questions = [
        "что такое УЦ?",
        "какие требования к электронной подписи?",
        "что такое СКЗИ?",
        "какие обязанности у кредитных организаций?"
    ]
    
    quality_rag = QualityControlledRAG(
        vector_store=VECTOR_STORE,
        llm=LLM,
        quality_threshold=0.9,
        max_iterations=3,
        enable_monitoring=True,
        enable_caching=True
    )
    
    print("=== BATCH ТЕСТИРОВАНИЕ ===")
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n🔍 Вопрос {i}/{len(questions)}: {question}")
        result = quality_rag.get_answer_with_quality_control(question, verbose=False)
        results.append(result)
        
        print(f"✅ Оценка: {result['final_score']:.2f}, Итерации: {result['iterations']}")
        if 'response_time' in result:
            print(f"⏱️ Время: {result['response_time']:.2f}с")
    
    # Финальная статистика
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    performance_report = quality_rag.get_performance_report()
    summary = performance_report['summary']
    
    print(f"Всего запросов: {summary['total_queries']}")
    print(f"Success rate: {summary['success_rate']}")
    print(f"Среднее время: {summary['average_response_time']}")
    print(f"Средняя оценка: {summary['average_quality_score']}")
    
    # Распределение качества
    print(f"\n📈 РАСПРЕДЕЛЕНИЕ КАЧЕСТВА:")
    for range_name, count in summary['quality_distribution'].items():
        if count > 0:
            print(f"{range_name}: {count} запросов")
    
    # Топ улучшения
    if summary['top_improvements']:
        print(f"\n🔧 ЧАСТЫЕ УЛУЧШЕНИЯ:")
        for improvement, count in summary['top_improvements'].items():
            print(f"- {improvement}: {count} раз")
    
    return results

if __name__ == "__main__":
    # Запускаем сравнение методов
    print("Сравнение старого и нового методов:\n")
    
    # Старый метод
    old_answer = old_method_example()
    
    print("\n" + "="*80 + "\n")
    
    # Новый метод с контролем качества
    quality_result = quality_controlled_example()
    
    # Дополнительная статистика
    print(f"\n📈 ИСТОРИЯ УЛУЧШЕНИЙ:")
    for i, verification in enumerate(quality_result['verification_history'], 1):
        print(f"Итерация {i}: {verification['score']:.2f} - {verification['reasoning']}")

# Для интерактивного тестирования
def interactive_quality_test():
    """Функция для интерактивного тестирования с собственными вопросами"""
    quality_rag = QualityControlledRAG(
        vector_store=VECTOR_STORE,
        llm=LLM,
        quality_threshold=0.9,
        max_iterations=3
    )
    
    while True:
        question = input("\nВведите вопрос (или 'exit' для выхода): ")
        if question.lower() == 'exit':
            break
            
        result = quality_rag.get_answer_with_quality_control(question, verbose=True)
        print(f"\n📝 ОТВЕТ: {result['final_answer']}")
        print(f"📊 Оценка: {result['final_score']:.2f}")

# Запуск основного примера
if __name__ == "__main__":
    print("🚀 RAG система с контролем качества")
    print("Выберите режим тестирования:")
    print("1. Одиночный вопрос с детальным выводом")
    print("2. Batch тестирование (4 вопроса)")
    print("3. Интерактивный режим")
    print("4. Сравнение со старым методом")
    
    choice = input("\nВведите номер (1-4): ").strip()
    
    if choice == "1":
        quality_controlled_example()
    elif choice == "2":
        batch_quality_test()
    elif choice == "3":
        interactive_quality_test()
    elif choice == "4":
        # Старый метод
        old_answer = old_method_example()
        print("\n" + "="*80 + "\n")
        # Новый метод
        quality_result = quality_controlled_example()
        # Сравнение
        print(f"\n📋 СРАВНЕНИЕ:")
        print(f"Старый метод: базовый ответ")
        print(f"Новый метод: {quality_result['iterations']} итераций, оценка {quality_result['final_score']:.2f}")
    else:
        print("Неверный выбор. Запускаем режим по умолчанию...")
        quality_controlled_example()