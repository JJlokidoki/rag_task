import os
import numpy as np
import logging
import faiss
import redis
import networkx as nx

from framework.RAG_tools.abstract.v1_0_0.abstract import Retriever, VectorStore, GraphStore, InMemoryClient

def l2_normalize(vectors):
    if vectors.ndim == 1: 
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm != 0 else vectors
    # массив векторов
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9 # чтобы нулевой вектор остался нулевым после деления
    return vectors / norms

class VectorRetriever(Retriever):

    def __init__(self, store: VectorStore, desc = None, t = 1.5, k=3):
        self.store = store
        self.threshold = t
        self.description = desc # "Vector"
        self.k = k

    def get(self, text, k=None, **kwargs):
        if not k:
            k = self.k  # Это костыль для обратной совместимости с версией 2_0_1
        results = self.store.get(text, k)
        candidate_texts = []
        if results:
            for dist, doc_text in results:
                if doc_text is not None and doc_text.strip() and dist < self.threshold:
                    candidate_texts.append(doc_text.strip())
        return candidate_texts

    # def _ranker(self, results):
    #     res = ''
    #     for elem in results:
    #         if elem[0] < self.threshold:
    #             res = res + '\n' + elem[1]
    #     if res == '':
    #         res = None
    #     return res
        
    # def get(self, text, k = 3, **kwargs):
    #     results = self.store.get(text, k)
    #     return self._ranker(results)


class  GraphRetriever(Retriever):
    """Обход графа, который извлекает узлы с заданными степенями."""

    def __init__(self, graph_store: GraphStore):
        self.store = graph_store

    def get(self, node_id, degrees, relationship_filter):
        visited = set()
        results = []
        queue = [(node_id, 0)]

        while queue:
            current_id, current_degree = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)

            if current_degree > 0:
                node_data = self.store.get_node(current_id)
                if node_data:
                    results.append({
                        'node': node_data,
                        'degree': current_degree
                    })

            if current_degree < degrees:
                neighbors = self.store.get_neighbors(current_id, relationship_filter)
                for neighbor in neighbors:
                    if neighbor['node'] is not None:
                        queue.append((neighbor['node']['id'], current_degree + 1))

        return results

'''
class SearchRetriever(Retriever):

    def __init__(self, executor, desc="WebSearch"):
        self.executor = executor
        self.description = desc
        # pass

    def get(self, text, **kwargs):
        result = self.executor.invoke(State(query=text))
        return result['global_summaries'] # TODO is it always just a string?
    
class CyberOnlineRetriever(Retriever):

    def __init__(self, executor_cyber_help, desc="CyberOnline"):
        self.executor_cyber_help = executor_cyber_help
        self.description = desc
        # pass

    def get(self, text, **kwargs):
        result = self.executor_cyber_help.invoke(StateCyberHelp(query=text))
        return result['global_summaries'] # TODO is it always just a string?
'''    
    
class ChromaRetriever(VectorRetriever):

    def get(self,text, k= 3):
        res = ''
        results = self.store.search(text, search_type='similarity', k=k)
        candidate_texts = []

        if results:
            for doc in results:
                if doc is not None:
                    candidate_texts.append(doc.page_content.strip())
        return candidate_texts
    

class FaissVectorStore(VectorStore):

    def __init__(self, url, port, embedder, index_name, splitter=None, in_memory=False):
        self.model = embedder
        self.url = url
        self.port = port
        self.client = redis.Redis(host=self.url, port=self.port, db=0, decode_responses=True)
        if in_memory:
            self.client = InMemoryClient()
        self.index_name = index_name
        self.splitter = splitter
        if os.path.isfile(index_name + '.index'):
            self.index = faiss.read_index(index_name + '.index')
            self.max_idx = self.index.ntotal
        else:
            self.index = faiss.IndexFlatL2(np.array(self.model.encode(['example'])).astype('float32').shape[1])
            self.max_idx = 0
            faiss.write_index(self.index, index_name + '.index')        

    
    def get(self, query_text, k):
        query_embedding_list = self.model.encode([query_text]) 
        query_embedding_np = np.array(query_embedding_list).astype('float32')

        normalized_query_embedding = l2_normalize(query_embedding_np)
        dist, idxs = self.index.search(normalized_query_embedding, k)
        
        # Получаем тексты по индексам
        # Убедимся, что idxs не пустой и содержит валидные индексы
        final_results = []
        if idxs.size > 0:
            # idxs[0] содержит список индексов для первого (и единственного) запроса
            # faiss может вернуть -1, если найдено меньше k соседей или если индекс пуст
            valid_indices_with_distances = []
            for i, original_idx in enumerate(idxs[0].tolist()):
                if original_idx != -1: # -1 означает, что для этой позиции не найдено соседа
                    valid_indices_with_distances.append((dist[0][i], original_idx))
            # получаем текст с нужными индексами
            for distance_val, original_idx_val in valid_indices_with_distances:
                text = self.client.get(self.index_name + '_' + str(original_idx_val))
                if text is not None: 
                    final_results.append((distance_val, text))
        return final_results

    
    def add(self, texts):
        texts_to_add = []
        if isinstance(texts, str):
            # Если передан один документ (строка)
            if self.splitter:
                logging.info(f"сплиттим документ (длина: {len(texts)})")
                texts_to_add = self.splitter.split_text(texts)
                logging.info(f"документ разбит на {len(texts_to_add)} чанков")
            else:
                logging.info("получили докумени, чанкер не инициализирован, добавлен весь текст как один чанк")
                texts_to_add = [texts] # Добавляем как один чанк
        elif isinstance(texts, list) and all(isinstance(item, str) for item in texts):
            logging.info(f"получен список из {len(texts)} предразбитых чанков")
            texts_to_add = texts
        else:
            logging.error("на вход подаем строку или список строк")
        vecs = np.array(self.model.encode(texts_to_add)).astype('float32')
        normalized_vecs = l2_normalize(vecs)
        start_idx_for_new_vectors = self.index.ntotal # текущее количество векторов в FAISS
        self.index.add(normalized_vecs)
        
        for i, t in enumerate(texts_to_add):
            actual_faiss_index = start_idx_for_new_vectors + i # порядковый номер (индекс) вектора t в FAISS-индексе после текущего добавления
            self.client.set(self.index_name + '_' + str(actual_faiss_index), t) # сохраняем текст t с ключом, включающим имя индекса и порядковый номер вектора
            
        self.max_idx = self.index.ntotal
        faiss.write_index(self.index, self.index_name + '.index')

    
    def prepare_document(self, doc):
        result = []
        return result

    
    def delete(self):
        pass


class NetworkXGraphStore(GraphStore):

    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node_id, properties):
        self.graph.add_node(node_id, **(properties or {}))

    def add_edge(self, source_id, target_id, relationship, properties):
        self.graph.add_edge(source_id, target_id, relationship=relationship, **(properties or {}))

    def get_node(self, node_id):
        if node_id in self.graph:
            node_data = self.graph.nodes[node_id]
            return {'id': node_id, **node_data}
        return None

    def get_neighbors(self, node_id, relationship):
        if node_id not in self.graph:
            return []

        neighbors = []
        for neighbor_id in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor_id)
            if relationship is None or edge_data.get('relationship') == relationship:
                neighbor_data = {
                    'node': self.get_node(neighbor_id),
                    'relationship': edge_data.get('relationship'),
                    'edge_properties': {k: v for k, v in edge_data.items() if k != 'relationship'}
                }
                neighbors.append(neighbor_data)
        return neighbors

    def delete_node(self, node_id):
        self.graph.remove_node(node_id)

    def delete_edge(self, source_id, target_id, relationship):
        if self.graph.has_edge(source_id, target_id):
            edge_data = self.graph.get_edge_data(source_id, target_id)
            if edge_data.get('relationship') == relationship:
                self.graph.remove_edge(source_id, target_id)