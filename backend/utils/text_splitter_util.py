from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer


class MyTextSplitter:
    def __init__(self, text):
        self.text = text

    def get_text_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(self.text)
        return chunks
    def get_cosine_similarity_chunks(self):
       

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(self.text)

        tfidf_vectorizer = TfidfVectorizer()
        all_tfidf_vectors = tfidf_vectorizer.fit_transform(chunks)

        chunked_text = [] 
        current_chunk = ""

        for i in range(len(chunks)):
            chunk_tfidf = all_tfidf_vectors[i]

            similarities = chunk_tfidf.dot(all_tfidf_vectors[:i].T).toarray()[0]

            if any(sim > 0.7 for sim in similarities): 
                current_chunk += " " + chunks[i]
            else:
                chunked_text.append(current_chunk.strip())
                current_chunk = chunks[i]

        if current_chunk:
            chunked_text.append(current_chunk.strip())
        print(chunked_text)
        return chunked_text