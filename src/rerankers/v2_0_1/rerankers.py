import logging
import re
from sentence_transformers.cross_encoder import CrossEncoder
from framework.blanks.v2_0_1.blanks import Prompts

from framework.RAG_tools.abstract.v1_0_0.abstract import BaseReranker


class LLMReranker(BaseReranker):
    """
    LLM для оценки релевантности каждого документа.
    """
    def __init__(
        self,
        llm_model, 
        llm_rerank_system_prompt: str | None = None,
        llm_rerank_user_template: str | None = None
    ):
        self.llm_model = llm_model
        self.prompts = Prompts()

        self.llm_rerank_system_prompt = llm_rerank_system_prompt if llm_rerank_system_prompt is not None \
            else self.prompts.RERANK_SYSTEM_MESSAGE
        self.llm_rerank_user_template = llm_rerank_user_template if llm_rerank_user_template is not None \
            else self.prompts.RERANK_TEMPLATE

        if not self.llm_model:
            logging.warning("LLM модель не предоставлена. Реранжирование с LLM будет отключено.")
        if not self.llm_rerank_system_prompt or "{query}" not in self.llm_rerank_user_template or "{document}" not in self.llm_rerank_user_template:
            logging.error(
                "Промпты для LLM реранжирования не заданы, пусты или не содержат плейсхолдеры {query}/{document}. "
                "LLM реранкер не будет работать корректно."
            )

        if self.llm_model:
            logging.info("LLM модель для реранжирования предоставлена и промпты готовы")


    def _parse_llm_score(self, llm_output_content: str):
        """Оценка релевантности (1-5) из ответа LLM."""
        match = re.search(r'\b([1-5])\b', llm_output_content.strip())
        if match:
            return int(match.group(1))
        else:
            logging.warning(f"Не удалось извлечь оценку (1-5) из ответа LLM: '{llm_output_content}'. Присвоена оценка 1.")
            return 1 # Минимальная оценка по умолчанию


    def rerank(self, query: str, candidate_docs: dict, top_n: int):
        if not self.llm_model or not candidate_docs or top_n <= 0:
            logging.info("LLM реранкер не готов, нет кандидатов или top_n <= 0. Реранжирование не выполняется.")
            doc_list = list(candidate_docs.keys()) if candidate_docs else []
            final_docs = []
            for i, doc_text in enumerate(doc_list):
                if i >= top_n:
                    break
                final_docs.append({"text": doc_text, "source": candidate_docs[doc_text]})
            return final_docs

        if "{query}" not in self.llm_rerank_user_template or "{document}" not in self.llm_rerank_user_template:
            logging.error("Шаблон пользователя для LLM реранкера не содержит необходимых плейсхолдеров {query} и {document}")
            doc_list = list(candidate_docs.keys()) if candidate_docs else []
            final_docs_fallback = []
            for i, doc_text in enumerate(doc_list):
                if i >= top_n:
                    break
                final_docs_fallback.append({"text": doc_text, "source": candidate_docs[doc_text]})
            return final_docs_fallback


        doc_texts_list = list(candidate_docs.keys())
        num_candidates = len(doc_texts_list)
        actual_top_n = min(top_n, num_candidates)

        if actual_top_n == 0:
            return []

        logging.info(f"Реранжируем {num_candidates} документов с использованием LLM, выбираем топ-{actual_top_n}")
        doc_list_with_scores = []

        for i, doc_text in enumerate(doc_texts_list):
            score = 1 # по умолчанию при ошибке
            try:
                rerank_user_content = self.llm_rerank_user_template.format(query=query, document=doc_text)
                msg = [
                     {'role': 'system', 'content': self.llm_rerank_system_prompt},
                     {'role': 'user', 'content': rerank_user_content}
                ]

                response = self.llm_model.invoke(msg)
                content = response.get('message', {}).get('content', '') 
                
                logging.debug(f"LLM ответ на реранжирование документа {i+1}/{num_candidates}: '{content}'")
                score = self._parse_llm_score(content)

            except KeyError as ke:
                logging.error(f"Ошибка форматирования промпта для LLM реранкера (документ '{doc_text}'): {ke}. Проверьте плейсхолдеры в шаблоне.")
                score = 1 # Не удалось сформировать промпт
            except Exception as e:
                logging.error(f"Ошибка LLM при реранжировании документа {i+1}/{num_candidates} ('{doc_text}'): {e}", exc_info=True)
                score = 1

            doc_list_with_scores.append({
                "text": doc_text,
                "source": candidate_docs[doc_text],
                "score": score
            })

        ranked_results = sorted(doc_list_with_scores, key=lambda x: x['score'], reverse=True)
        logging.info(f"Выбрано топ-{actual_top_n} документов после реранжирования LLM.")

        if ranked_results[:actual_top_n]:
             logging.debug(f"Источники топ-{actual_top_n} документов (LLM): {[d['source'] for d in ranked_results[:actual_top_n]]}")

        final_docs_output = []
        for item in ranked_results[:actual_top_n]:
            final_docs_output.append({"text": item["text"], "source": item["source"]})
        return final_docs_output
    

class CrossEncoderReranker(BaseReranker):
    """
    CrossEncoder для оценки релевантности.
    """
    def __init__(self, cross_encoder_model_name):
        self.cross_encoder = None
        if cross_encoder_model_name:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model_name)
                logging.info(f"CrossEncoder '{cross_encoder_model_name}' загружен.")
            except Exception as e:
                logging.error(f"Не удалось загрузить CrossEncoder '{cross_encoder_model_name}': {e}")
                self.cross_encoder = None

        if not self.cross_encoder:
            logging.warning("CrossEncoder не инициализирован. Реранжирование с CrossEncoder будет отключено.")

    def rerank(self, query, candidate_docs, top_n):
        if not self.cross_encoder or not candidate_docs or top_n <= 0:
            logging.info("CrossEncoder не готов, нет кандидатов или top_n <= 0. Реранжирование не выполняется.")
            # просто берем топ N из исходного списка (если есть)
            doc_list = list(candidate_docs.keys()) if candidate_docs else []
            final_docs = [{"text": doc, "source": candidate_docs[doc]} for doc in doc_list[:min(top_n, len(doc_list))]]
            return final_docs

        doc_list = list(candidate_docs.keys()) # исходный список текстов
        num_candidates = len(doc_list)
        actual_top_n = min(top_n, num_candidates) # не просим больше, чем есть

        logging.info(f"Реранжируем {num_candidates} документов с использованием CrossEncoder, выбираем топ-{actual_top_n}")
        reranker_input = [[query, doc] for doc in doc_list]

        try:
            scores = self.cross_encoder.predict(reranker_input, show_progress_bar=False) # порядок оценок соответствует порядку пар во входном списке

            doc_list_with_scores = []
            for i, doc_text in enumerate(doc_list):
                 doc_list_with_scores.append({
                    "text": doc_text, # текст документа
                    "source": candidate_docs[doc_text], # источник из словаря candidate_docs, используя текст как ключ
                    "score": float(scores[i])
                 })

            ranked_results = sorted(doc_list_with_scores, key=lambda x: x['score'], reverse=True)
            logging.info(f"Выбрано топ-{actual_top_n} документов после реранжирования CrossEncoder.")

            if ranked_results[:actual_top_n]:
                 logging.debug(f"Источники топ-{actual_top_n} документов (CrossEncoder): {[d['source'] for d in ranked_results[:actual_top_n]]}")

            final_docs = [{"text": item["text"], "source": item["source"]} for item in ranked_results[:actual_top_n]]
            return final_docs

        except Exception as e:
            logging.error(f"Ошибка CrossEncoder при реранжировании: {e}")
            # используем исходный порядок, если реранжирование не удалось
            doc_list = list(candidate_docs.keys())
            final_docs = [{"text": doc, "source": candidate_docs[doc]} for doc in doc_list[:actual_top_n]]
            logging.warning("CrossEncoder реранжирование не удалось, возвращаем топ N из исходных кандидатов.")
            return final_docs