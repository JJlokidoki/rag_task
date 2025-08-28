import time
from typing import Dict, List
import json
from datetime import datetime

from .logger import logger


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
