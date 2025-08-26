import unittest
from unittest.mock import MagicMock, patch, call
import os
import sys
import numpy as np
import logging

absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
print(absolute_path)
sys.path.append(absolute_path)

from framework.RAG_tools.retrievers.v2_0_1.retrievers import (
    VectorRetriever,
    ChromaRetriever,
    GraphRetriever,
    FaissVectorStore,
    NetworkXGraphStore
)

class FakeEmbedder:
    def __init__(self, embedding_dim=3):
        self.embedding_dim = embedding_dim
        self.encode_call_count = 0
        self.last_encoded_texts = None

    def encode(self, texts: list[str]):
        self.encode_call_count += 1
        self.last_encoded_texts = texts
        return [[float(i + j) for j in range(self.embedding_dim)] for i in range(len(texts))]

class FakeSplitter:
    def __init__(self):
        self.split_text_call_count = 0
        self.last_split_text_input = None

    def split_text(self, text: str):
        self.split_text_call_count += 1
        self.last_split_text_input = text
        return [s.strip() for s in text.split('.') if s.strip()]


class TestFaissVectorStore(unittest.TestCase):

    def setUp(self):
        self.mock_embedder = FakeEmbedder(embedding_dim=3)
        self.mock_splitter = FakeSplitter()
        self.index_name = "test_faiss_index"
        self.redis_url = "localhost"
        self.redis_port = 6379

        # Очищаем возможные файлы индекса от предыдущих запусков
        if os.path.isfile(self.index_name + ".index"):
            os.remove(self.index_name + ".index")

    def tearDown(self):
        # Очищаем созданные файлы индекса
        if os.path.isfile(self.index_name + ".index"):
            os.remove(self.index_name + ".index")

    # Мокаем зависимости, которые FaissVectorStore использует 
    @patch('framework.RAG_tools.retrievers.v2_0_1.retrievers.l2_normalize')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.redis.Redis')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.faiss.write_index')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.faiss.read_index')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.os.path.isfile')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.faiss.IndexFlatL2')
    def test_initialization_new_index(self, MockIndexFlatL2, mock_isfile, mock_read_index, mock_write_index, MockRedis, mock_l2_normalize_func):
        mock_isfile.return_value = False # Файла индекса нет
        mock_redis_client = MagicMock()
        MockRedis.return_value = mock_redis_client # Когда redis.Redis будет вызван, он вернет наш mock_redis_client
        
        mock_faiss_index_instance = MagicMock()
        MockIndexFlatL2.return_value = mock_faiss_index_instance # Конструктор вернет наш мок

        store = FaissVectorStore(
            url=self.redis_url, port=self.redis_port,
            embedder=self.mock_embedder, index_name=self.index_name,
            splitter=self.mock_splitter
        )

        mock_isfile.assert_called_once_with(self.index_name + ".index") # FaissVectorStore при инициализации ровно один раз проверил существование файла индекса с ожидаемым именем.
        MockRedis.assert_called_once_with(host=self.redis_url, port=self.redis_port, db=0, decode_responses=True) # FaissVectorStore при инициализации ровно один раз создал клиент Redis с правильными параметрами подключения.
        self.assertEqual(store.client, mock_redis_client) # кземпляр клиента Redis, который был возвращен моком конструктора, был правильно присвоен атрибуту client вашего объекта store
        
        # Проверяем, что embedder.encode был вызван для создания структуры индекса
        self.assertEqual(self.mock_embedder.encode_call_count, 1)
        self.assertEqual(self.mock_embedder.last_encoded_texts, ['example']) # self.mock_embedder.encode был вызван один раз с ['example'] (для определения размерности индекса)
        MockIndexFlatL2.assert_called_once_with(3) # Размерность из FakeEmbedder
        
        mock_write_index.assert_called_once_with(mock_faiss_index_instance, self.index_name + ".index")
        mock_read_index.assert_not_called()
        self.assertEqual(store.max_idx, 0)
        mock_l2_normalize_func.assert_not_called()

    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.redis.Redis')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.faiss.write_index') # Не должен вызываться
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.faiss.read_index')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.os.path.isfile')
    def test_initialization_existing_index(self, mock_isfile, mock_read_index, mock_write_index, MockRedis):
        mock_isfile.return_value = True # Файл индекса существует
        mock_redis_client = MagicMock()
        MockRedis.return_value = mock_redis_client
        
        mock_loaded_index = MagicMock()
        mock_loaded_index.ntotal = 5 # В существующем индексе 5 элементов
        mock_loaded_index.d = self.mock_embedder.embedding_dim # мок загруженного индекса должен иметь размерность
        mock_read_index.return_value = mock_loaded_index # faiss.read_index вернет мок существующего индекса

        store = FaissVectorStore(
            url=self.redis_url, port=self.redis_port,
            embedder=self.mock_embedder, index_name=self.index_name
        )

        mock_isfile.assert_called_once_with(self.index_name + ".index")
        mock_read_index.assert_called_once_with(self.index_name + ".index")
        self.assertEqual(store.index, mock_loaded_index)
        self.assertEqual(store.max_idx, 5)
        mock_write_index.assert_not_called() # Не должен создаваться и записываться новый

    @patch('framework.RAG_tools.retrievers.v2_0_1.retrievers.l2_normalize')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.redis.Redis')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.faiss.write_index')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.os.path.isfile')
    @patch('framework.RAG_tools.retrievers.v1_0_0.retrievers.faiss.IndexFlatL2')
    def test_add_texts_with_splitter(self, MockIndexFlatL2, mock_isfile, mock_write_index, MockRedis, mock_l2_normalize_func):
        mock_isfile.return_value = False # Создаем новый индекс
        mock_redis_client = MagicMock()
        MockRedis.return_value = mock_redis_client

        mock_faiss_index_instance = MagicMock() # Это будет наш store.index
        # начальное значение ntotal для мока индекса, чтобы start_idx_for_new_vectors был корректным
        mock_faiss_index_instance.ntotal = 0
        MockIndexFlatL2.return_value = mock_faiss_index_instance

        normalized_vector_mock = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32) 
        mock_l2_normalize_func.return_value = normalized_vector_mock

        store = FaissVectorStore(
            url=self.redis_url, port=self.redis_port,
            embedder=self.mock_embedder, index_name=self.index_name,
            splitter=self.mock_splitter # Передаем сплиттер
        )
        # Сбрасываем счетчики после инициализации
        self.mock_embedder.encode_call_count = 0
        self.mock_splitter.split_text_call_count = 0
        mock_write_index.reset_mock() # Сброс мока записи индекса

        texts_to_add = "Это первое предложение. А это второе."
        store.add(texts_to_add)

        self.assertEqual(self.mock_splitter.split_text_call_count, 1)
        self.assertEqual(self.mock_splitter.last_split_text_input, texts_to_add)
        
        self.assertEqual(self.mock_embedder.encode_call_count, 1)
        # FakeSplitter разделит на ["Это первое предложение", "А это второе"]
        self.assertEqual(self.mock_embedder.last_encoded_texts, ["Это первое предложение", "А это второе"])

        mock_l2_normalize_func.assert_called_once()
        # l2_normalize была вызвана с сырыми эмбеддингами
        raw_embeddings_from_fake_embedder = np.array(self.mock_embedder.encode(["Это первое предложение", "А это второе"])).astype(np.float32)
        # Сбрасываем счетчик, тк вызвали encode еще раз для получения raw_embeddings
        self.mock_embedder.encode_call_count -=1 

        # cравнение массивов
        np.testing.assert_array_almost_equal(mock_l2_normalize_func.call_args[0][0], raw_embeddings_from_fake_embedder)
        
        # store.index.add был вызван с "нормализованными" векторами от мока
        mock_faiss_index_instance.add.assert_called_once()
        np.testing.assert_array_almost_equal(mock_faiss_index_instance.add.call_args[0][0], normalized_vector_mock)
        
        # store.index.add.assert_called_once() # метод add у mock faiss индекса был вызван
        # Проверяем вызовы redis client.set
        expected_redis_calls = [
            call.set(f"{self.index_name}_0", "Это первое предложение"),
            call.set(f"{self.index_name}_1", "А это второе")
        ]
        mock_redis_client.assert_has_calls(expected_redis_calls, any_order=False)
        
        self.assertEqual(store.max_idx, 2) # Было 0, добавили 2
        mock_write_index.assert_called_once() # Индекс должен быть сохранен после добавления

    @patch('framework.RAG_tools.retrievers.v2_0_1.retrievers.l2_normalize')
    @patch('framework.RAG_tools.retrievers.v2_0_1.retrievers.faiss.IndexFlatL2')
    @patch('framework.RAG_tools.retrievers.v2_0_1.retrievers.os.path.isfile')
    @patch('framework.RAG_tools.retrievers.v2_0_1.retrievers.faiss.write_index')
    @patch('framework.RAG_tools.retrievers.v2_0_1.retrievers.redis.Redis') 
    def test_get_texts(self, MockRedis, mock_faiss_write_index, mock_isfile, MockIndexFlatL2, mock_l2_normalize_func):
        mock_isfile.return_value = False
        mock_redis_client = MagicMock()
        MockRedis.return_value = mock_redis_client

        mock_faiss_index_instance = MagicMock()
        MockIndexFlatL2.return_value = mock_faiss_index_instance

        store = FaissVectorStore(
            url=self.redis_url, port=self.redis_port,
            embedder=self.mock_embedder, index_name=self.index_name
        )
        normalized_query_vector_mock = np.array([[0.7, 0.8, 0.9]], dtype=np.float32)
        mock_l2_normalize_func.return_value = normalized_query_vector_mock
        # Мокаем метод search у экземпляра индекса Faiss, чтобы он возвращал предопределенные расстояния и индексы
        store.index.search = MagicMock(return_value=(np.array([[0.1, 0.2]]), np.array([[0, 1]])))
        # Мокаем ответы от Redis
        mock_redis_client.get.side_effect = lambda key: {
            f"{self.index_name}_0": "Текст документа 0",
            f"{self.index_name}_1": "Текст документа 1"
        }.get(key)

        self.mock_embedder.encode_call_count = 0 # Сброс для этого теста

        query = "мой запрос"
        k_results = 2
        results = store.get(query, k_results)

        self.assertEqual(self.mock_embedder.encode_call_count, 1)
        self.assertEqual(self.mock_embedder.last_encoded_texts, [query])

        mock_l2_normalize_func.assert_called_once()
        raw_query_embedding = np.array(self.mock_embedder.encode([query])).astype(np.float32)
        self.mock_embedder.encode_call_count -=1 

        np.testing.assert_array_almost_equal(mock_l2_normalize_func.call_args[0][0], raw_query_embedding)
        store.index.search.assert_called_once()
        np.testing.assert_array_almost_equal(store.index.search.call_args[0][0], normalized_query_vector_mock)
        # Проверяем, что search был вызван с эмбеддингом запроса и k
        self.assertEqual(store.index.search.call_args[0][1], k_results) # k - второй аргумент search

        expected_redis_get_calls = [
            call.get(f"{self.index_name}_0"),
            call.get(f"{self.index_name}_1")
        ]
        mock_redis_client.assert_has_calls(expected_redis_get_calls, any_order=True) # Порядок может быть не важен

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], (0.1, "Текст документа 0"))
        self.assertEqual(results[1], (0.2, "Текст документа 1"))


class TestVectorRetriever(unittest.TestCase):

    def setUp(self):
        self.mock_store = MagicMock(spec=FaissVectorStore) # Указываем spec, чтобы мок имел методы VectorStore

    def test_get_filters_by_threshold_and_returns_texts(self):
        # Настраиваем, что вернет mock_store.get()
        self.mock_store.get.return_value = [
            (0.5, "Документ 1 хороший"),  # dist < threshold
            (1.6, "Документ 2 плохой"),   # dist > threshold
            (0.9, "Документ 3 нормальный"),# dist < threshold
            (2.0, "Документ 4 очень плохой"),# dist > threshold
            (1.0, None)                   # Текст None
        ]
        retriever = VectorRetriever(store=self.mock_store, t=1.5, desc="TestRetriever")
        
        query = "запрос"
        k_val = 5
        candidate_texts = retriever.get(query, k=k_val)

        self.mock_store.get.assert_called_once_with(query, k_val)
        self.assertEqual(len(candidate_texts), 2)
        self.assertIn("Документ 1 хороший", candidate_texts)
        self.assertIn("Документ 3 нормальный", candidate_texts)
        self.assertNotIn("Документ 2 плохой", candidate_texts)

    def test_get_empty_results_from_store(self):
        self.mock_store.get.return_value = []
        retriever = VectorRetriever(store=self.mock_store, t=1.5)
        candidate_texts = retriever.get("запрос", k=3)
        self.assertEqual(candidate_texts, [])

    def test_get_no_docs_pass_threshold(self):
        self.mock_store.get.return_value = [
            (1.8, "Док1"), (2.5, "Док2")
        ]
        retriever = VectorRetriever(store=self.mock_store, t=1.5)
        candidate_texts = retriever.get("запрос", k=2)
        self.assertEqual(candidate_texts, [])


class TestChromaRetriever(unittest.TestCase):

    def setUp(self):
        # ChromaRetriever использует self.store.search
        self.mock_store_for_chroma = MagicMock()

    def test_get_concatenates_page_content(self):
        # Мокаем store.search, который должен возвращать список объектов,
        # у каждого из которых есть атрибут page_content
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Содержимое первого документа."
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Содержимое второго документа."
        
        self.mock_store_for_chroma.search.return_value = [mock_doc1, mock_doc2]
        
        retriever = ChromaRetriever(store=self.mock_store_for_chroma)
        query = "запрос"
        k_val = 2
        result_string = retriever.get(query, k=k_val)

        self.mock_store_for_chroma.search.assert_called_once_with(query, search_type='similarity', k=k_val)
        expected_string = ['Содержимое первого документа.', 'Содержимое второго документа.']
        self.assertEqual(result_string, expected_string)

    def test_get_empty_search_results(self):
        self.mock_store_for_chroma.search.return_value = []
        retriever = ChromaRetriever(store=self.mock_store_for_chroma)
        result_string = retriever.get("запрос", k=3)
        self.assertEqual(result_string, [])


class TestGraphRetriever(unittest.TestCase):
    def setUp(self):
        self.store = NetworkXGraphStore()
        self.retriever = GraphRetriever(self.store)

        # Построение простого графа:
        # A -> B (FRIEND)
        # A -> C (COLLEAGUE)
        # B -> D (FAMILY)
        # C -> D (COLLEAGUE)
        self.store.add_node("A", {"name": "Alice"})
        self.store.add_node("B", {"name": "Bob"})
        self.store.add_node("C", {"name": "Charlie"})
        self.store.add_node("D", {"name": "Dave"})

        self.store.add_edge("A", "B", "FRIEND", {"since": 2010})
        self.store.add_edge("A", "C", "COLLEAGUE", None)
        self.store.add_edge("B", "D", "FAMILY", None)
        self.store.add_edge("C", "D", "COLLEAGUE", None)

    def test_get_degree_1(self):
        '''
        Тестирует получение узлов степени 1.
        '''
        
        results = self.retriever.get("A", 1, None)
        self.assertEqual(len(results), 2)
        names = {result["node"]["name"] for result in results}
        self.assertEqual(names, {"Bob", "Charlie"})

    def test_get_degree_2(self):
        '''
        Тестирует получение узлов степени 2.
        '''

        results = self.retriever.get("A", 2, None)
        self.assertEqual(len(results), 3)  # B, C, D
        names = {result["node"]["name"] for result in results}
        self.assertEqual(names, {"Bob", "Charlie", "Dave"})

        degrees = {result["degree"] for result in results}
        self.assertEqual(degrees, {1, 2})

    def test_relationship_filter(self):
        '''
        Тестирует фильтр по ребру FRIEND
        '''
        results = self.retriever.get("A", 1, "FRIEND")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["node"]["name"], "Bob")

        '''
        Тестирует фильтр по ребру COLLEAGUE
        '''
        results = self.retriever.get("A", 2, "COLLEAGUE")
        self.assertEqual(len(results), 2)  # C and D
        names = {result["node"]["name"] for result in results}
        self.assertEqual(names, {"Charlie", "Dave"})

    def test_non_existent_node(self):
        '''
        Тестирует несуществующие узлы.
        '''
        results = self.retriever.get("Z", 1, None)
        self.assertEqual(len(results), 0)

    def test_zero_degrees(self):
        '''
        Тестирует получение узлов степени 0.
        '''
        results = self.retriever.get("A", 0, None)
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()