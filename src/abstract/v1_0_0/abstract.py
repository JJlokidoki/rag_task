from abc import ABC, abstractmethod
import re


class BaseReranker(ABC):

    @abstractmethod
    def rerank(self, query, candidate_docs, top_n):
        pass


class BaseTextSplitter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def split_text(self, text):
        pass

    @staticmethod
    def normalize_text(text):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text) # замена множественных пробелов на один
        return text


class Retriever(ABC):

    @abstractmethod
    def get(self):
        pass

class GraphStore(ABC):

    @abstractmethod
    def add_node(self):
        pass

    @abstractmethod
    def add_edge(self):
        pass

    @abstractmethod
    def get_node(self):
        pass

    @abstractmethod
    def get_neighbors(self):
        pass

    @abstractmethod
    def delete_node(self):
        pass

    @abstractmethod
    def delete_edge(self):
        pass

class VectorStore(ABC):

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def delete(self):
        pass

class InMemoryClient(ABC):
    def __init__(self):
        self.store = {}
    
    def set(self, key, value):
        self.store[key] = value
    
    def get(self, key):
        return self.store.get(key)