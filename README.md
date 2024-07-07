## Lizzy AI: Legal Contract Q&A Chatbot with RAG System

This repository implements a Retriever-Augmented Generation (RAG) system for a legal contract Q&A chatbot, built by Lizzy AI. The system allows users to ask questions about legal contracts and receive informative answers.

### Project Goals

* Develop a robust RAG system for legal contract Q&A.
* Explore and evaluate different chunking strategies.
* Implement techniques to improve answer accuracy and relevance.

### Features

* Interactive chat interface for user queries.
* Knowledge base of legal contracts in PDF format.
* Text extraction and chunking from PDFs (simple and cosine similarity based).
* Retriever component using vector embeddings for efficient information retrieval.
* Generator component leveraging OpenAI's GPT-4 for answer generation.
* Prompt-based approach with context from retrieved contract sections.
* Evaluation framework to assess answer quality using various metrics.

### Getting Started

This project requires Python libraries like Flask, PyPDF2, Langchain, OpenAI API, ragas, and a compatible GPT-4 API access.

1. Clone the repository:

```bash
git clone https://github.com/your-username/lizzy-ai-legal-qa.git
```

2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Configure API Keys:

* Create an account with OpenAI and obtain an API key.
* (Optional) Set up access to a compatible GPT-4 API.

4. Run the front end application:

```bash
cd frontend
npm install
npm start
```
This will start the user interface at `http://localhost:3000/` in your web browser.

4. Run the bakend end application:

```bash

python app.py
```

### Usage

The chat interface allows users to type questions about legal contracts from the knowledge base. The system retrieves relevant contract sections and utilizes GPT-4 to generate informative answers based on the retrieved context.

### Evaluation

The project includes an evaluation script that calculates metrics like answer relevancy, faithfulness, context recall, and precision. You can modify the script to use your own evaluation datasets.

### Further Development

This project provides a foundation for a legal contract Q&A chatbot. Here are some potential areas for further development:

* Integration with a more advanced user interface framework.
* Exploration of alternative chunking algorithms.
* Implementation of multi-query retrieval with reranking for improved accuracy.
* Training a custom LLM specifically for legal contracts.
* Expanding the knowledge base with a wider variety of legal documents.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
