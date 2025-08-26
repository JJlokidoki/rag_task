import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from query_rag import QualityControlledRAG, VECTOR_STORE, LLM
except ImportError as e:
    st.error(f"Ошибка импорта: {e}")
    st.stop()

# Конфигурация страницы
st.set_page_config(
    page_title="RAG Quality Control System",
    page_icon="🤖",
    layout="wide"
)

# Инициализация состояния сессии
if 'quality_rag' not in st.session_state:
    st.session_state.quality_rag = QualityControlledRAG(
        vector_store=VECTOR_STORE,
        llm=LLM,
        quality_threshold=0.9,
        max_iterations=3,
        enable_monitoring=True
    )

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def main():
    st.title("🤖 RAG система с контролем качества")
    st.markdown("Система вопросов-ответов по криптографии с автоматической проверкой качества")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        quality_threshold = st.slider(
            "Порог качества", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.9, 
            step=0.05
        )
        
        max_iterations = st.slider(
            "Максимум итераций", 
            min_value=1, 
            max_value=5, 
            value=3
        )
        
        verbose_mode = st.checkbox("Детальный вывод", value=False)
        
        # Обновляем настройки системы
        st.session_state.quality_rag.quality_threshold = quality_threshold
        st.session_state.quality_rag.max_iterations = max_iterations
        
        st.markdown("---")
        
        # Кнопки управления
        if st.button("🔄 Сбросить метрики"):
            st.session_state.quality_rag.reset_metrics()
            st.session_state.query_history = []
            st.success("Метрики сброшены!")
            st.rerun()
        
        if st.button("💾 Сохранить метрики"):
            if st.session_state.quality_rag.enable_monitoring:
                st.session_state.quality_rag.monitor.save_metrics()
                st.success("Метрики сохранены!")
    
    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Задать вопрос")
        
        # Примеры вопросов
        example_questions = [
            "что такое УЦ?",
            "какие требования к электронной подписи?",
            "что такое СКЗИ?",
            "какие обязанности у кредитных организаций?",
            "что такое квалифицированная электронная подпись?"
        ]
        
        selected_example = st.selectbox(
            "Выберите пример вопроса:",
            [""] + example_questions
        )
        
        user_question = st.text_area(
            "Ваш вопрос:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="Введите вопрос о криптографии или нормативных актах..."
        )
        
        if st.button("🚀 Получить ответ", type="primary"):
            if user_question.strip():
                with st.spinner("Обработка запроса..."):
                    result = st.session_state.quality_rag.get_answer_with_quality_control(
                        user_question, 
                        verbose=verbose_mode
                    )
                    
                    # Сохраняем в историю сессии
                    st.session_state.query_history.append(result)
                
                # Отображаем результат
                display_result(result, verbose_mode)
            else:
                st.error("Пожалуйста, введите вопрос")
    
    with col2:
        st.header("📊 Статистика")
        display_metrics()
    
    # История запросов
    if st.session_state.query_history:
        st.header("📝 История запросов")
        display_query_history()
    
    # Аналитика
    if st.session_state.quality_rag.enable_monitoring:
        metrics = st.session_state.quality_rag.monitor.metrics
        if metrics['total_queries'] > 0:
            st.header("📈 Аналитика")
            display_analytics()

def display_result(result, verbose_mode):
    """Отображение результата запроса"""
    st.markdown("---")
    st.header("📋 Результат")
    
    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_color = "🟢" if result['final_score'] >= 0.9 else "🟡" if result['final_score'] >= 0.7 else "🔴"
        st.metric(
            "Оценка качества",
            f"{result['final_score']:.2f}",
            delta=score_color
        )
    
    with col2:
        st.metric("Итераций", result['iterations'])
    
    with col3:
        if 'response_time' in result:
            st.metric("Время ответа", f"{result['response_time']:.2f}с")
    
    with col4:
        status = "✅ Принят" if result['quality_acceptable'] else "❌ Отклонен"
        st.metric("Статус", status)
    
    # Ответ
    st.subheader("💡 Ответ:")
    st.markdown(result['final_answer'])
    
    # Детальная информация
    if verbose_mode:
        with st.expander("🔍 Детальная информация"):
            st.json(result['verification_history'])

def display_metrics():
    """Отображение общих метрик"""
    if not st.session_state.quality_rag.enable_monitoring:
        st.info("Мониторинг отключен")
        return
    
    metrics = st.session_state.quality_rag.monitor.metrics
    
    if metrics['total_queries'] == 0:
        st.info("Пока нет данных")
        return
    
    # Основные показатели
    success_rate = metrics['successful_queries'] / metrics['total_queries']
    
    st.metric("Всего запросов", metrics['total_queries'])
    st.metric("Success Rate", f"{success_rate:.1%}")
    st.metric("Средняя оценка", f"{metrics['average_quality_score']:.2f}")
    st.metric("Среднее время", f"{metrics['average_response_time']:.1f}с")

def display_query_history():
    """Отображение истории запросов"""
    for i, result in enumerate(reversed(st.session_state.query_history[-5:]), 1):
        with st.expander(f"Запрос {len(st.session_state.query_history) - i + 1}: {result['question'][:50]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Ответ:** {result['final_answer'][:200]}...")
            
            with col2:
                st.metric("Оценка", f"{result['final_score']:.2f}")
                st.metric("Итерации", result['iterations'])

def display_analytics():
    """Отображение аналитических графиков"""
    metrics = st.session_state.quality_rag.monitor.metrics
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Распределение качества
        st.subheader("📊 Распределение качества")
        quality_data = metrics['quality_distribution']
        
        if sum(quality_data.values()) > 0:
            df_quality = pd.DataFrame([
                {'Диапазон': k, 'Количество': v} 
                for k, v in quality_data.items() if v > 0
            ])
            
            fig_pie = px.pie(
                df_quality, 
                values='Количество', 
                names='Диапазон',
                title="Распределение оценок качества"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # График качества по времени
        if len(metrics['query_history']) > 1:
            st.subheader("📈 Качество во времени")
            
            df_history = pd.DataFrame([
                {
                    'timestamp': datetime.fromisoformat(q['timestamp']),
                    'score': q['final_score'],
                    'question': q['question'][:30] + "..."
                }
                for q in metrics['query_history']
            ])
            
            fig_line = px.line(
                df_history, 
                x='timestamp', 
                y='score',
                title="Изменение качества ответов",
                hover_data=['question']
            )
            fig_line.add_hline(y=0.9, line_dash="dash", line_color="red")
            st.plotly_chart(fig_line, use_container_width=True)
    
    # Топ улучшений
    if metrics['improvement_types']:
        st.subheader("🔧 Частые проблемы")
        
        top_improvements = dict(sorted(
            metrics['improvement_types'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5])
        
        df_improvements = pd.DataFrame([
            {'Проблема': k, 'Частота': v} 
            for k, v in top_improvements.items()
        ])
        
        fig_bar = px.bar(
            df_improvements, 
            x='Частота', 
            y='Проблема',
            orientation='h',
            title="Наиболее частые проблемы"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Рекомендации
    performance_report = st.session_state.quality_rag.get_performance_report()
    if performance_report.get('recommendations'):
        st.subheader("💡 Рекомендации по улучшению")
        for rec in performance_report['recommendations']:
            st.info(rec)

if __name__ == "__main__":
    main()
