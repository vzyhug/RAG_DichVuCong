from .retriever import Retriever
from .query_analyzer import QueryAnalyzer
from .grader import Grader

class ContextRetriever:
    def __init__(self):
        self.retriever = Retriever()
        self.analyzer = QueryAnalyzer()
        self.grader = Grader()

    def get_context(self, query: str):
        entities = self.analyzer.extract_entities(query)
        results = self.retriever.retrieve(query)
        if self.grader.is_sufficient(results, query):
            return {"status": "success", "contexts": results, "entities": entities}
        return {"status": "insufficient", "contexts": results, "entities": entities}