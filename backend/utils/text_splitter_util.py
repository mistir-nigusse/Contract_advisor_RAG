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
        """
        Splits the text into chunks based on cosine similarity.

        Returns:
            list: A list of semantically similar text chunks.
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(self.text)

        # Improved chunking with cosine similarity
        tfidf_vectorizer = TfidfVectorizer()
        all_tfidf_vectors = tfidf_vectorizer.fit_transform(chunks)

        chunked_text = []  # Stores semantically similar chunks
        current_chunk = ""

        for i in range(len(chunks)):
            chunk_tfidf = all_tfidf_vectors[i]

            # Calculate cosine similarity between current chunk and all previous ones
            similarities = chunk_tfidf.dot(all_tfidf_vectors[:i].T).toarray()[0]

            # If similarity is above a threshold (e.g., 0.7), add to current chunk
            if any(sim > 0.7 for sim in similarities): 
                current_chunk += " " + chunks[i]
            else:
                # If not similar, start a new chunk
                chunked_text.append(current_chunk.strip())
                current_chunk = chunks[i]

        # Add the last chunk (if any)
        if current_chunk:
            chunked_text.append(current_chunk.strip())
        print(chunked_text)
        return chunked_text