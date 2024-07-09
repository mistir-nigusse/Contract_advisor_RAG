from transformers import pipeline

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.reranker = pipeline('feature-extraction', model=model_name)

    def rerank_chunks(self, query, chunks):
        inputs = [{"query": query, "chunk": chunk} for chunk in chunks]
        scores = self.reranker(inputs)
        ranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        print["RANKED CHUNKS"]
        print(ranked_chunks)
        return ranked_chunks
