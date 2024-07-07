import os
import json
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
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableParallel
from operator import itemgetter
from datasets import Dataset

# Load environment variables
from dotenv import load_dotenv
from backend.utils.langchain_util import MyLangChain
from backend.utils.pdf_util import MyPDF
from backend.utils.text_splitter_util import MyTextSplitter
from backend.utils.vector_store_util import MyVectorStore

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_api_key
langchain = MyLangChain(api_key=openai_api_key)

def load_qa_pairs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_pdf(file_path):
    pdf_processor = MyPDF(file_path)
    return pdf_processor.get_pdf_text()

def split_text(raw_text):
    text_splitter = MyTextSplitter(raw_text)
    return text_splitter.get_cosine_similarity_chunks()

def create_vector_store(question, text_chunks):
    vector_store = MyVectorStore()
    hypothetical_answer = langchain.generate_hypothetical_answer(question)
    combined_text = f"{question}\n{hypothetical_answer}"
    return vector_store.embed_text_and_return_vectorstore(text_chunks + [combined_text])

def generate_answer(question, retriever):
    
    conversation_chain = langchain.generate_prompts_chain(base_retriever=retriever)
    result = conversation_chain.invoke({"question": question})
    response_content = result['response'].content
    context_list = [document.page_content.strip() for document in result['context']]
    return response_content, context_list

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
    qa_pairs = load_qa_pairs('backend/data/qa_pairs.json')

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

    for item in qa_pairs:
        question = item['question']
        expected_answer = item['ground_truths']
        raw_text = process_pdf('backend/data/Raptor_ontract.pdf')
        text_chunks = split_text(raw_text)
        chroma_vector_store = create_vector_store(question,text_chunks)
        retriever = MyVectorStore().get_retriever(chroma_vector_store)
        generated_answer, context_list = generate_answer(question, retriever)

        evaluation_data["question"].append(question)
        evaluation_data["answer"].append(generated_answer)
        evaluation_data["contexts"].append(context_list)
        evaluation_data["ground_truth"].append(expected_answer)

        single_result = evaluate_answer(question, generated_answer, context_list, expected_answer)
        
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
    df.to_csv('evaluation_results.csv', index=False)
    print(df)

if __name__ == "__main__":
    main()
