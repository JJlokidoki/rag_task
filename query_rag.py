# from langchain.chains.combine_documents import create_stuff_documents_chain

import os
from dotenv import find_dotenv, load_dotenv
import getpass
import json
import re
from datetime import datetime
from typing import Dict, List

from utils.logger import logger
from rag.llm import VECTOR_STORE, LLM
from prompts.crypto_expert import CRYPTO_EXPERT_PROMPT, CRYPTO_EXPERT_REFINEMENT_PROMPT
from prompts.crypto_verificator import CRYPTO_VERIFICATOR_PROMPT
from utils.perfomance_monitor import PerformanceMonitor
from utils.query_cache import QueryCache

# load .env
load_dotenv(find_dotenv())

if "GIGACHAT_CREDENTIALS" not in os.environ:
    os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")


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


class AnswerQualityVerifier:
    """Агент для проверки качества и полноты ответов"""
    
    def __init__(self, llm):
        self.llm = llm
        self.verification_prompt = CRYPTO_VERIFICATOR_PROMPT
    
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
        self.main_prompt = CRYPTO_EXPERT_PROMPT
        
        # Промпт для доработки
        self.refinement_prompt = CRYPTO_EXPERT_REFINEMENT_PROMPT
    
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

# метод с контролем качества и мониторингом
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
    # Новый метод с контролем качества
    quality_result = quality_controlled_example()
    
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
    
    choice = input("\nВведите номер (1-4): ").strip()
    
    if choice == "1":
        quality_controlled_example()
    elif choice == "2":
        batch_quality_test()
    elif choice == "3":
        interactive_quality_test()

    else:
        print("Неверный выбор. Запускаем режим по умолчанию...")
        quality_controlled_example()