from transformers import pipeline

class QueryGenerator:
    def __init__(self, model_name='gpt-4'):
        self.generator = pipeline('text-generation', model=model_name)

    def generate_similar_queries(query, model, num_queries=5):
        similar_queries = [query]
        for _ in range(num_queries):
            similar_queries.append(query + f" variation {_}")
        return similar_queries
