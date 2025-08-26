import unittest
import logging
from unittest.mock import MagicMock, patch, call 
import os
import sys

# add framework directory to path
absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
print(absolute_path)
sys.path.append(absolute_path)


from framework.RAG_tools.rerankers.v2_0_1.rerankers import LLMReranker, CrossEncoderReranker
from framework.blanks.v2_0_1.blanks import Prompts

QUERY = "Какой сегодня день?"
CANDIDATE_DOCS_SIMPLE = {
    "Док1: Солнечно": {"source": "s1"},
    "Док2: Облачно": {"source": "s2"},
    "Док3: Дождливо": {"source": "s3"}
}

PATH_TO_CROSSENCODER_CLASS = 'framework.RAG_tools.rerankers.v1_0_0.rerankers.CrossEncoder'

class TestLLMRerankerUnittest(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock() # мок LLM
        self.prompts_instance = Prompts()
        self.doc_keys = list(CANDIDATE_DOCS_SIMPLE.keys())


    def test_rerank_basic_functionality(self):
        """Тестирует правильную сортировку документов на основе оценок от LLM и возврат всех документов"""
        self.mock_llm.invoke.side_effect = [ # список значений, которые будут возвращаться при последовательных вызовах
            {"message": {"content": "Оценка: 5"}}, # Док1
            {"message": {"content": "Оценка: 3"}}, # Док2
            {"message": {"content": "Оценка: 1"}}, # Док3
        ]
        reranker = LLMReranker(llm_model=self.mock_llm)
        ranked_docs = reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=3)

        self.assertEqual(len(ranked_docs), 3)
        self.assertEqual(ranked_docs[0]["text"], self.doc_keys[0]) # Док1 (5)
        self.assertEqual(ranked_docs[1]["text"], self.doc_keys[1]) # Док2 (3)
        self.assertEqual(ranked_docs[2]["text"], self.doc_keys[2]) # Док3 (1)
        self.assertEqual(self.mock_llm.invoke.call_count, 3)

    def test_rerank_top_n_respected(self):
        """параметр top_n правильно ограничивает количество возвращаемых документов"""
        self.mock_llm.invoke.side_effect = [
            {"message": {"content": "1"}}, # Док1
            {"message": {"content": "5"}}, # Док2
            {"message": {"content": "3"}}, # Док3
        ] # Оценки: Док2 (5), Док3 (3), Док1 (1)
        reranker = LLMReranker(llm_model=self.mock_llm)
        ranked_docs = reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=2)

        self.assertEqual(len(ranked_docs), 2)
        self.assertEqual(ranked_docs[0]["text"], self.doc_keys[1]) # Док2
        self.assertEqual(ranked_docs[1]["text"], self.doc_keys[2]) # Док3

    def test_rerank_no_model_returns_original_order_top_n(self):
        """когда ллм нет"""
        reranker = LLMReranker(llm_model=None)
        ranked_docs = reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=2)
        self.assertEqual(len(ranked_docs), 2)
        self.assertEqual(ranked_docs[0]["text"], self.doc_keys[0])
        self.assertEqual(ranked_docs[1]["text"], self.doc_keys[1])

    def test_rerank_empty_candidates_or_zero_top_n(self):
        """пустой список кандидатов"""
        reranker = LLMReranker(llm_model=self.mock_llm)
        self.assertEqual(reranker.rerank(QUERY, {}, top_n=2), [])
        self.assertEqual(reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=0), [])
        self.mock_llm.invoke.assert_not_called()

    def test_parse_llm_score(self):
        reranker = LLMReranker(llm_model=self.mock_llm) # llm_model не используется этим методом
        self.assertEqual(reranker._parse_llm_score("5"), 5)
        self.assertEqual(reranker._parse_llm_score("Текст с оценкой 3 внутри."), 3)
        self.assertEqual(reranker._parse_llm_score("Нет оценки"), 1)

    def test_bad_prompt_template_init_logs_error(self):
        # self.assertLogs для проверки логов
        with self.assertLogs(logger=logging.getLogger(), level='ERROR') as log_context:
            LLMReranker(llm_model=self.mock_llm, llm_rerank_user_template="")
        
        log_output = "".join(log_context.output)
        self.assertTrue(
            "Промпты для LLM реранжирования не заданы" in log_output or \
            "не содержат плейсхолдеры" in log_output
        )

    def test_bad_prompt_template_runtime_fallback(self):
        reranker = LLMReranker(llm_model=self.mock_llm)
        reranker.llm_rerank_user_template = "Плохой шаблон"
        
        with self.assertLogs(logger=logging.getLogger(), level='ERROR') as log_context:
            ranked_docs = reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=1)
        
        log_output = "".join(log_context.output)
        self.assertIn("Шаблон пользователя для LLM реранкера не содержит необходимых плейсхолдеров", log_output)
        self.assertEqual(len(ranked_docs), 1)
        self.assertEqual(ranked_docs[0]["text"], self.doc_keys[0])
        self.mock_llm.invoke.assert_not_called()

    def test_rerank_uses_correct_prompts(self):
        """Проверяем, что используются правильные системный и пользовательский промпты."""
        reranker = LLMReranker(llm_model=self.mock_llm) # Использует дефолтные промпты
        self.mock_llm.invoke.return_value = {"message": {"content": "1"}} # Не важен ответ

        # один документ для проверки
        doc_key_to_test = self.doc_keys[0]
        doc_to_test = {doc_key_to_test: CANDIDATE_DOCS_SIMPLE[doc_key_to_test]}
        
        reranker.rerank(QUERY, doc_to_test, top_n=1)

        self.mock_llm.invoke.assert_called_once()
        actual_call_args = self.mock_llm.invoke.call_args[0]
        messages_sent_to_llm = actual_call_args[0] 

        self.assertEqual(len(messages_sent_to_llm), 2)
        self.assertEqual(messages_sent_to_llm[0]['role'], 'system')
        self.assertEqual(messages_sent_to_llm[0]['content'], self.prompts_instance.RERANK_SYSTEM_MESSAGE)
        
        self.assertEqual(messages_sent_to_llm[1]['role'], 'user')
        expected_user_content = self.prompts_instance.RERANK_TEMPLATE.format(
            query=QUERY, document=doc_key_to_test
        )
        self.assertEqual(messages_sent_to_llm[1]['content'], expected_user_content)


class TestCrossEncoderRerankerUnittest(unittest.TestCase):

    def setUp(self):
        self.doc_keys = list(CANDIDATE_DOCS_SIMPLE.keys())

    @patch(PATH_TO_CROSSENCODER_CLASS)
    def test_rerank_basic_functionality(self, MockCrossEncoderClass):
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.predict.return_value = [0.8, 0.2, 0.5] # Док1, Док3, Док2
        MockCrossEncoderClass.return_value = mock_encoder_instance # когда мокнутый CrossEncoder (MockCrossEncoderClass) вызывается, он должен вернуть наш mock_encoder_instance
        
        reranker = CrossEncoderReranker(cross_encoder_model_name="any-model")
        ranked_docs = reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=3)

        self.assertEqual(len(ranked_docs), 3)
        self.assertEqual(ranked_docs[0]["text"], self.doc_keys[0]) # Док1 (0.8)
        self.assertEqual(ranked_docs[1]["text"], self.doc_keys[2]) # Док3 (0.5)
        self.assertEqual(ranked_docs[2]["text"], self.doc_keys[1]) # Док2 (0.2)
        
        expected_predict_input = [[QUERY, key] for key in self.doc_keys]
        mock_encoder_instance.predict.assert_called_once_with(expected_predict_input, show_progress_bar=False)
        MockCrossEncoderClass.assert_called_once_with("any-model") # Проверяем вызов конструктора

    @patch(PATH_TO_CROSSENCODER_CLASS)
    def test_rerank_top_n_respected(self, MockCrossEncoderClass):
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.predict.return_value = [0.1, 0.9, 0.5] # Док2, Док3, Док1
        MockCrossEncoderClass.return_value = mock_encoder_instance # когда конструктор CrossEncoder(model_name) вызывается в коде CrossEncoderReranker, 
        # он должен вернуть наш заранее подготовленный mock_cross_encoder_instance
        
        reranker = CrossEncoderReranker(cross_encoder_model_name="any-model")
        ranked_docs = reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=2)

        self.assertEqual(len(ranked_docs), 2)
        self.assertEqual(ranked_docs[0]["text"], self.doc_keys[1]) # Док2
        self.assertEqual(ranked_docs[1]["text"], self.doc_keys[2]) # Док3

    def test_rerank_encoder_not_initialized_fallback(self):
        reranker = CrossEncoderReranker(cross_encoder_model_name=None) # Не инициализируем
        with self.assertLogs(logger=logging.getLogger(), level='INFO') as log_context:
            ranked_docs = reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=2)
        
        self.assertIn("CrossEncoder не готов", "".join(log_context.output))
        self.assertEqual(len(ranked_docs), 2)
        self.assertEqual(ranked_docs[0]["text"], self.doc_keys[0])
        self.assertEqual(ranked_docs[1]["text"], self.doc_keys[1])

    @patch(PATH_TO_CROSSENCODER_CLASS) # мокаем, чтобы predict не вызывался реально
    def test_rerank_empty_candidates_or_zero_top_n(self, MockCrossEncoderClass):
        mock_encoder_instance = MagicMock()
        MockCrossEncoderClass.return_value = mock_encoder_instance
        reranker = CrossEncoderReranker(cross_encoder_model_name="any-model")

        self.assertEqual(reranker.rerank(QUERY, {}, top_n=2), [])
        self.assertEqual(reranker.rerank(QUERY, CANDIDATE_DOCS_SIMPLE, top_n=0), [])
        mock_encoder_instance.predict.assert_not_called()



if __name__ == '__main__':
    unittest.main()