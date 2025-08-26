import os
import numpy as np
import logging
import faiss
import redis
import networkx as nx

from src.abstract.v1_0_0.abstract import Retriever, VectorStore, GraphStore

class FaissVectorStore(VectorStore):

    def __init__(self, url, port, embedder, index_name):
        self.model = embedder
        self.url = url
        self.port = port
        self.client = redis.Redis(host=self.url, port=self.port, db=0, decode_responses=True)
        self.index_name = index_name
        if os.path.isfile(index_name + '.index'):
            self.index = faiss.read_index(index_name + '.index')
            self.max_idx = self.index.ntotal
        else:
            self.index = faiss.IndexFlatL2(np.array(self.model.encode(['example'])).astype('float32').shape[1])
            self.max_idx = 0
            faiss.write_index(self.index, index_name + '.index')        

    
    def get(self, query_text, k):
        dist, idxs = self.index.search(np.array(self.model.encode([query_text])).astype('float32'), k)
        res_texts = [self.client.get(self.index_name + '_' + str(i)) for i in idxs[0].tolist()]
        return [x for x in zip(dist[0].tolist(), res_texts)]

    
    def add(self, texts):
        vecs = np.array(self.model.encode(texts)).astype('float32')
        self.index.add(vecs)
        for i, t in enumerate(texts):
            self.client.set(self.index_name + '_' + str(self.max_idx + i), t)
            stored_text = self.client.get(self.index_name + '_' + str(self.max_idx + i))
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