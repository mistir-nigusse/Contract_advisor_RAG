from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel

class MyLangChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.primary_qa_llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    def generate_hypothetical_answer(self, question):
        prompt = f"Generate a very short and clear hypothetical answer for the following question:\n\nQuestion: {question}\n\nAnswer:"

        try:
            response = client.chat.completions.create(model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant and expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0)
            answer = response.choices[0].message.content.strip()
            print(answer)
            return answer
        except Exception as e:
            print(f"Error generating hypothetical answer: {e}")
            return "Sorry, I couldn't generate an answer."

    def generate_prompts_chain(self, base_retriever):
        template = """You are an AI assistant and expert.
        Use the following pieces of context to answer the question at the end.
        Use three up to five sentences maximum and keep the answer as concise as possible.
        ### CONTEXT
        {context}

        ### User question
        User question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        retriever = RunnableParallel({
            "context": itemgetter("question") | base_retriever,
            "question": itemgetter("question"),
        })

        retrieval_augmented_qa_chain = retriever | {
            "response": prompt | self.primary_qa_llm,
            "context": itemgetter("context"),
        }
        return retrieval_augmented_qa_chain
