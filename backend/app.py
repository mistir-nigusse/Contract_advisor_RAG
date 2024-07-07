from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json
from utils.pdf_util import MyPDF
from utils.text_splitter_util import MyTextSplitter
from utils.vector_store_util import MyVectorStore
from backend.utils.langchain_util import MyLangChain

load_dotenv()

app = Flask(__name__)
CORS(app)
pdf_path = 'backend/data/Raptor_ontract.pdf'

openai_api_key = os.getenv('OPENAI_API_KEY')

@app.route('/get_question', methods=['POST'])
def get_question():
    data = request.json
    question = data.get('question', '')
    langchain = MyLangChain(api_key=openai_api_key)
    hypothetical_answer = langchain.generate_hypothetical_answer(question)
    combined_text = f"{question}\n{hypothetical_answer}"

    pdf_processor = MyPDF(pdf_path)
    raw_text = pdf_processor.get_pdf_text()

    text_splitter = MyTextSplitter(raw_text)
    text_chunks = text_splitter.get_cosine_similarity_chunks()
    # text_chunks = text_splitter.get_text_chunks()


    vector_store = MyVectorStore()
    chroma_vector_store = vector_store.embed_text_and_return_vectorstore(text_chunks + [combined_text])

    retriever = vector_store.get_retriever(chroma_vector_store)
    conversation_chain = langchain.generate_prompts_chain(base_retriever=retriever)

    result = conversation_chain.invoke({
        "question": question,
    })

    response_content = result['response'].content

    if isinstance(response_content, str):
        response_content = response_content.strip()
        try:
            answer = json.loads(response_content)
        except json.JSONDecodeError as e:
            return jsonify({'answer': response_content}), 200

    return jsonify({'answer': answer}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
