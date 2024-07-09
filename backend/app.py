from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder, util
# from backend.evaluation import generate_similar_queries, rerank_chunks
from backend.utils.langchain_util import MyLangChain
from backend.utils.pdf_util import MyPDF
from backend.utils.text_splitter_util import MyTextSplitter
from backend.utils.vector_store_util import MyVectorStore
# from backend.utils.query_generator import generate_queries
# from backend.utils.reranker_util import rerank_chunks
load_dotenv()

app = Flask(__name__)
CORS(app)
pdf_path = 'backend/data/Robinson_Advisory.pdf'
openai_api_key = os.getenv('OPENAI_API_KEY')

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Sentence Transformers model for query generation and CrossEncoder for reranking
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

def rerank_chunks(query, chunks, cross_encoder):
        pairs = [[query, chunk] for chunk in chunks]
        scores = cross_encoder.predict(pairs)
        sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        return sorted_chunks
def generate_queries(query,  num_queries=5):
        similar_queries = [query]
        for _ in range(5):
            similar_queries.append(query + f" variation {_}")
        return similar_queries

@app.route('/get_question', methods=['POST'])
def get_question():
    try:
        data = request.json
        question = data.get('question', '')

        if not question:
            raise ValueError("Question is missing from the request.")

        pdf_processor = MyPDF(pdf_path)
        raw_text = pdf_processor.get_pdf_text()
        # logging.debug(f"Raw text extracted from PDF: {raw_text[:500]}")  # Log only the first 500 characters

        text_splitter = MyTextSplitter(raw_text)
        text_chunks = text_splitter.get_cosine_similarity_chunks()

        vector_store = MyVectorStore()
        chroma_vector_store = vector_store.embed_text_and_return_vectorstore(text_chunks)
        # logging.debug("Vector store created successfully.")

        # Generate similar queries
        # query_generator = QueryGenerator()
        similar_queries = generate_queries(question)
        all_retrieved_chunks = []

        # Retrieve chunks for each query
        for query in similar_queries:
            retrieved_chunks = vector_store.retrieve_chunks(query, chroma_vector_store)
            all_retrieved_chunks.extend([doc.page_content for doc in retrieved_chunks])

        # Rerank chunks
        # reranker = Reranker()
        ranked_chunks = rerank_chunks(question, all_retrieved_chunks, cross_encoder)
        # logging.debug(f"Ranked chunks: {ranked_chunks[:5]}")  # Log only the first 5 chunks

        # Generate answer using top ranked chunks
        langchain = MyLangChain(api_key=openai_api_key)
        conversation_chain = langchain.generate_prompts_chain(ranked_chunks)
        result = conversation_chain.invoke({"question": question})
        response_content = result['response'].content
        # logging.debug(f"Response content: {response_content}")

        if isinstance(response_content, str):
            response_content = response_content.strip()
            try:
                answer = json.loads(response_content)
            except json.JSONDecodeError:
                answer = response_content

        return jsonify({'answer': answer}), 200

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)

