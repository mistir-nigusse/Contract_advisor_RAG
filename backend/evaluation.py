
import os
import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall
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
from backend.utils.langchain import MyLangChain
from backend.utils.pdf_util import MyPDF
from backend.utils.text_splitter_util import MyTextSplitter
from backend.utils.vector_store_util import MyVectorStore

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_api_key

with open('backend/qa_pairs.json', 'r', encoding='utf-8') as f:
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
    pdf_processor = MyPDF('backend/Raptor_ontract.pdf')
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

    evaluation_data["question"].append(question)
    evaluation_data["answer"].append(generated_answer)
    evaluation_data["contexts"].append(context_list)
    evaluation_data["ground_truths"].append(ground_truth_list)

    single_item_dataset = Dataset.from_dict({
        "question": [question],
        "answer": [generated_answer],
        "contexts": [context_list],
        "ground_truths": [ground_truth_list]
    })

    metrics = [answer_relevancy, faithfulness, context_recall]
    single_result = evaluate(single_item_dataset, metrics=metrics)

    evaluation_data["answer_relevancy"].append(single_result['answer_relevancy'])
    evaluation_data["faithfulness"].append(single_result['faithfulness'])
    evaluation_data["context_recall"].append(single_result['context_recall'])

df = pd.DataFrame(evaluation_data)

df.to_csv('evaluation_results.csv', index=False)

print(df)
