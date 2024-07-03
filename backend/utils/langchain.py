from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableParallel

class MyLangChain:
    def generate_prompts_chain(self, base_retriever):
        template = """You are an AI assistant and expert.
   Use the following pieces of context to answer the question at the end.
   If you don't know the answer, just say that you don't know, don't try to make up an answer.
   Use three upto five sentences maximum and keep the answer as concise as possible.
        ### CONTEXT
        {context}

        ### User question
        User question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        retriever = RunnableParallel(
            {
                "context": itemgetter("question") | base_retriever,
                "question": itemgetter("question"),
            }
        )

        retrieval_augmented_qa_chain = retriever | {
            "response": prompt | primary_qa_llm,
            "context": itemgetter("context"),
        }
        return retrieval_augmented_qa_chain