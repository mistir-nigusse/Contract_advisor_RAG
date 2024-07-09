import os
import json
import logging
import csv
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
)
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from backend.utils.langchain_util import MyLangChain
from backend.utils.pdf_util import MyPDF
from backend.utils.text_splitter_util import MyTextSplitter
from backend.utils.vector_store_util import MyVectorStore
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_api_key

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Initialize models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')
langchain = MyLangChain(api_key=openai_api_key)
pdf_path = 'backend/data/Raptor_ontract.pdf'

def rerank_chunks(query, chunks, cross_encoder):
    pairs = [[query, chunk] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return sorted_chunks

def generate_queries(query, num_queries=5):
    similar_queries = [query]
    for i in range(num_queries):
        similar_queries.append(query + f" variation {i}")
    return similar_queries

def process_question(question):
    try:
        if not question:
            raise ValueError("Question is missing.")

        pdf_processor = MyPDF(pdf_path)
        raw_text = pdf_processor.get_pdf_text()

        text_splitter = MyTextSplitter(raw_text)
        text_chunks = text_splitter.get_cosine_similarity_chunks()

        vector_store = MyVectorStore()
        chroma_vector_store = vector_store.embed_text_and_return_vectorstore(text_chunks)

        similar_queries = generate_queries(question)
        all_retrieved_chunks = []

        for query in similar_queries:
            retrieved_chunks = vector_store.retrieve_chunks(query, chroma_vector_store)
            all_retrieved_chunks.extend([doc.page_content for doc in retrieved_chunks])

        ranked_chunks = rerank_chunks(question, all_retrieved_chunks, cross_encoder)

        conversation_chain = langchain.generate_prompts_chain(ranked_chunks)
        result = conversation_chain.invoke({"question": question})
        response_content = result['response'].content

        if isinstance(response_content, str):
            response_content = response_content.strip()
            try:
                answer = json.loads(response_content)
            except json.JSONDecodeError:
                answer = response_content

        context_list = [chunk for chunk in ranked_chunks[:5]]  # Using top 5 chunks as context

        return question, answer, context_list

    except Exception as e:
        logging.error(f"Error processing question: {str(e)}", exc_info=True)
        return question, str(e), []

def evaluate_answer(question, generated_answer, context_list, expected_answer):
    single_item_dataset = Dataset.from_dict({
        "question": [question],
        "answer": [generated_answer],
        "contexts": [context_list],
        "ground_truth": [expected_answer]
    })

    metrics = [answer_relevancy, faithfulness, context_recall, context_precision, answer_similarity, answer_correctness]
    single_result = evaluate(single_item_dataset, metrics=metrics)

    return single_result

def main():
    input_json_path = 'backend/data/qa_pairs.json'
    output_csv_path = 'evaluation_results.csv'

    with open(input_json_path, 'r') as f:
        questions = json.load(f)

    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "answer_relevancy": [],
        "faithfulness": [],
        "context_recall": [],
        "context_precision": [],
        "answer_similarity": [],
        "answer_correctness": []
    }

    for item in questions:
        question = item['question']
        expected_answer = item['ground_truths']
        question_text, answer_text, context_list = process_question(question)

        evaluation_data["question"].append(question_text)
        evaluation_data["answer"].append(answer_text)
        evaluation_data["contexts"].append(context_list)
        evaluation_data["ground_truth"].append(expected_answer)

        single_result = evaluate_answer(question_text, answer_text, context_list, expected_answer)
        
        evaluation_data["answer_relevancy"].append(single_result['answer_relevancy'])
        evaluation_data["faithfulness"].append(single_result['faithfulness'])
        evaluation_data["context_recall"].append(single_result['context_recall'])
        evaluation_data["context_precision"].append(single_result['context_precision'])
        evaluation_data["answer_similarity"].append(single_result['answer_similarity'])
        evaluation_data["answer_correctness"].append(single_result['answer_correctness'])

    # Find the maximum length of all lists
    max_length = max(len(lst) for lst in evaluation_data.values())
    
    # Pad all lists to the maximum length
    for key in evaluation_data:
        evaluation_data[key].extend([''] * (max_length - len(evaluation_data[key])))

    df = pd.DataFrame(evaluation_data)
    df.to_csv('result_raptor.csv', index=False)
    print(df)

if __name__ == "__main__":
    main()
