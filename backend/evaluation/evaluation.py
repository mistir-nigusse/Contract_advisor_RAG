
import os
import json
import pandas as pd
import numpy as np
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall

from datasets import Dataset

from dotenv import load_dotenv
from backend.utils.langchain import MyLangChain
from backend.utils.pdf_util import MyPDF
from backend.utils.text_splitter_util import MyTextSplitter
from backend.utils.vector_store_util import MyVectorStore

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

with open('backend/data/qa_pairs.json', 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)

evaluation_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truths": [],
    "answer_relevancy": [],
    "faithfulness": [],
    "context_recall": []
}

for item in qa_pairs:
    question = item['question']
    expected_answer = item['ground_truths']
    pdf_processor = MyPDF('backend/data/Raptor_ontract.pdf')
    raw_text = pdf_processor.get_pdf_text()
    text_splitter = MyTextSplitter(raw_text)
    text_chunks = text_splitter.get_text_chunks()
    vector_store = MyVectorStore()
    chroma_vector_store = vector_store.embed_text_and_return_vectorstore(text_chunks)

    retriever = vector_store.get_retriever(chroma_vector_store)

    langchain = MyLangChain()
    conversation_chain = langchain.generate_prompts_chain(base_retriever=retriever)
    result = conversation_chain.invoke({
        "question": question,
    })
    response_content = result['response'].content  
    generated_answer = response_content

    print('Generated Answer:')
    print(generated_answer)

    context_list = [document.page_content.strip() for document in result['context']]
    ground_truth_list = [expected_answer.strip()]

    print('Context as String:')
    print(context_list)

    if not context_list or not ground_truth_list:
        print(f"Empty context or ground truth for question: {question}")
        continue

    normalized_context_list = [context.lower().strip() for context in context_list]
    normalized_ground_truth_list = [gt.lower().strip() for gt in ground_truth_list]

    print('Normalized Contexts:', normalized_context_list)
    print('Normalized Ground Truths:', normalized_ground_truth_list)

    evaluation_data["question"].append(question)
    evaluation_data["answer"].append(generated_answer)
    evaluation_data["contexts"].append(normalized_context_list)
    evaluation_data["ground_truths"].append(normalized_ground_truth_list)

    single_item_dataset = Dataset.from_dict({
        "question": [question],
        "answer": [generated_answer],
        "contexts": [normalized_context_list],
        "ground_truths": [normalized_ground_truth_list]
    })

    metrics = [answer_relevancy, faithfulness, context_recall]
    try:
        single_result = evaluate(single_item_dataset, metrics=metrics)
    except RuntimeWarning as e:
        print(f"Runtime warning encountered: {e}")
        single_result = {"answer_relevancy": 0.0, "faithfulness": 0.0, "context_recall": 0.0}

    evaluation_data["answer_relevancy"].append(single_result['answer_relevancy'])
    evaluation_data["faithfulness"].append(single_result['faithfulness'])
    evaluation_data["context_recall"].append(single_result['context_recall'])

df = pd.DataFrame(evaluation_data)

df.to_csv('backend/data/evaluation_results.csv', index=False)

print(df)
