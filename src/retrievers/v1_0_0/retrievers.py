from src.abstract.v1_0_0.abstract import Retriever, VectorStore, GraphStore

class VectorRetriever(Retriever):

    def __init__(self, store: VectorStore, desc = None, t = 1.5):
        self.store = store
        self.threshold = t
        self.description = desc # "Vector"

    def get(self, text, k=3, **kwargs):
        results = self.store.get(text, k)
        candidate_texts = []
        if results:
            for dist, doc_text in results:
                if doc_text is not None and doc_text.strip() and dist < self.threshold:
                    candidate_texts.append(doc_text.strip())
        return '\n'.join(candidate_texts)

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


class GraphRetriever(Retriever):
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
                        queue.append((list(neighbor['node'].keys())[0], current_degree + 1))

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

    def get(self, text, k= 3):
        res = ''
        results = self.store.search(text, search_type='similarity', k=k)
        for elem in results:
            res = res + '\n' + elem.page_content
        if res == '':
            res = None
        return res