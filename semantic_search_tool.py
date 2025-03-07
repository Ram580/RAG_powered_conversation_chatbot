

pip install langchain scikit-learn
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_search(
    search_phrase: str, 
    text: str, 
    top_k: int = 3, 
    similarity_threshold: float = 0.5
):
    """
    Perform semantic search over the provided text using Azure OpenAI embeddings.

    Args:
        search_phrase: Query to search for.
        text: Text to search within (expected to contain multiple passages separated by newlines).
        top_k: Number of top results to return.
        similarity_threshold: Minimum similarity score to include in results.

    Returns:
        List of tuples containing (matched_text, similarity_score).
    """
    
    # Initialize Azure OpenAI embeddings.
    # Replace the placeholders with your actual Azure deployment details.
    embeddings = OpenAIEmbeddings(
        deployment="your_deployment_name",               # e.g., "text-embedding-ada-002"
        model="text-embedding-ada-002",
        openai_api_base="https://<your-resource-name>.openai.azure.com/",
        openai_api_version="2023-03-15-preview",
        openai_api_key="YOUR_AZURE_OPENAI_API_KEY"
    )

    # Split the input text into passages (using newlines as a simple separator)
    passages = [p.strip() for p in text.split('\n') if p.strip()]
    if not passages:
        return []

    # Compute embeddings for each passage and for the search phrase
    passage_embeddings = embeddings.embed_documents(passages)
    query_embedding = embeddings.embed_query(search_phrase)

    # Compute cosine similarities between the query and each passage
    similarities = [
        cosine_similarity([query_embedding], [passage_emb])[0][0]
        for passage_emb in passage_embeddings
    ]

    # Pair each passage with its similarity score, filtering by the threshold
    results = [
        (passage, score) 
        for passage, score in zip(passages, similarities)
        if score >= similarity_threshold
    ]

    # Sort the results in descending order by similarity and return the top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Example usage:
if __name__ == "__main__":
    sample_text = (
        "Agentic AI systems are autonomous and goal-directed.\n"
        "They have the capability to perform tasks without human intervention.\n"
        "Semantic search tools are critical for retrieving relevant information.\n"
        "This is a demonstration of using Azure OpenAI embeddings with LangChain."
    )
    
    query = "autonomous AI tasks"
    top_results = semantic_search(query, sample_text, top_k=2, similarity_threshold=0.3)
    
    for passage, score in top_results:
        print(f"Score: {score:.3f} | Passage: {passage}")




from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool

class SemanticSearchTool(BaseTool):
    name = "SemanticSearchTool"
    description = (
        "A custom tool for performing semantic search on a given text using Azure OpenAI embeddings. "
        "It returns a list of tuples (matched_text, similarity_score) where similarity is computed as 1 - distance."
    )

    def __init__(self, embeddings: OpenAIEmbeddings):
        """
        Initialize the tool with an embedding model.
        """
        self.embeddings = embeddings

    def _run(
        self,
        search_phrase: str,
        text: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5
    ):
        """
        Run the semantic search.

        Args:
            search_phrase: Query to search for.
            text: Text to search within (passages separated by newlines).
            top_k: Number of top results to return.
            similarity_threshold: Minimum similarity score (between 0 and 1) to include in results.

        Returns:
            List of tuples containing (matched_text, similarity_score).
        """
        # Split the text into passages (adjust chunk size/overlap as needed)
        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)
        passages = splitter.split_text(text)
        if not passages:
            return []

        # Create a FAISS vectorstore from the passages
        vectorstore = FAISS.from_texts(passages, self.embeddings)

        # Retrieve all results with their distance scores
        results = vectorstore.similarity_search_with_score(search_phrase, k=len(passages))

        # Convert distance to similarity (assuming cosine distance where similarity = 1 - distance)
        filtered_results = []
        for doc, distance in results:
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                filtered_results.append((doc.page_content, similarity))

        # Sort by similarity in descending order and return top_k results
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:top_k]

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Asynchronous mode is not supported.")


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the Azure OpenAI embeddings (update with your Azure details)
    embeddings = OpenAIEmbeddings(
        deployment="your_deployment_name",             # e.g., "text-embedding-ada-002"
        model="text-embedding-ada-002",
        openai_api_base="https://<your-resource-name>.openai.azure.com/",
        openai_api_version="2023-03-15-preview",
        openai_api_key="YOUR_AZURE_OPENAI_API_KEY"
    )

    # Create the semantic search tool instance
    search_tool = SemanticSearchTool(embeddings)

    sample_text = (
        "Agentic AI systems are autonomous and goal-directed.\n"
        "They have the capability to perform tasks without human intervention.\n"
        "Semantic search tools are critical for retrieving relevant information.\n"
        "This is a demonstration of using Azure OpenAI embeddings with LangChain."
    )
    
    query = "autonomous AI systems"
    results = search_tool._run(query, sample_text, top_k=2, similarity_threshold=0.3)
    
    for passage, score in results:
        print(f"Similarity: {score:.3f} | Passage: {passage}")



import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool

class SemanticSearchTool(BaseTool):
    name = "SemanticSearchTool"
    description = (
        "A custom tool for performing semantic search on a DataFrame using Azure OpenAI embeddings. "
        "It searches the text in the specified text column (e.g., 'summary') and returns the top similar "
        "results along with their ID."
    )

    def __init__(self, embeddings: OpenAIEmbeddings):
        """
        Initialize the tool with an embedding model.
        """
        self.embeddings = embeddings

    def _run(
        self,
        search_phrase: str,
        df: pd.DataFrame,
        text_column: str,
        id_column: str = "ID",
        top_k: int = 3,
        similarity_threshold: float = 0.5
    ):
        """
        Run the semantic search over a DataFrame.

        Args:
            search_phrase: Query to search for.
            df: A pandas DataFrame containing text and metadata.
            text_column: The name of the column to be used for semantic search.
            id_column: The column containing unique identifiers (default "ID").
            top_k: Number of top results to return.
            similarity_threshold: Minimum similarity score (0 to 1) to include in results.

        Returns:
            List of tuples (ID, summary_text, similarity_score).
        """
        # Ensure the text column exists and extract the list of texts
        if text_column not in df.columns or id_column not in df.columns:
            raise ValueError("DataFrame must contain the specified text and ID columns.")
            
        texts = df[text_column].tolist()
        ids = df[id_column].tolist()

        # Optionally split the text further using LangChain's splitter if needed.
        # Here we assume each row is already a discrete document.
        docs = [
            Document(page_content=text, metadata={id_column: doc_id})
            for text, doc_id in zip(texts, ids)
        ]

        # Create a FAISS vectorstore from the documents
        vectorstore = FAISS.from_documents(docs, self.embeddings)

        # Perform the similarity search over all documents
        results = vectorstore.similarity_search_with_score(search_phrase, k=len(docs))

        # Convert FAISS distance to cosine similarity (assuming normalized embeddings, similarity = 1 - distance)
        filtered_results = []
        for doc, distance in results:
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                filtered_results.append((doc.metadata.get(id_column), doc.page_content, similarity))

        # Sort results by similarity in descending order and return top_k
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        return filtered_results[:top_k]

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Asynchronous mode is not supported.")

# --- Example Usage ---
if __name__ == "__main__":
    # Sample DataFrame with ID and summary columns
    data = {
        "ID": [1, 2, 3, 4],
        "summary": [
            "Agentic AI systems are autonomous and goal-directed.",
            "They perform tasks without human intervention.",
            "Semantic search tools retrieve relevant information efficiently.",
            "This is a demonstration of using Azure OpenAI embeddings with LangChain."
        ]
    }
    df = pd.DataFrame(data)
    
    # Initialize the Azure OpenAI embeddings (update with your actual Azure details)
    embeddings = OpenAIEmbeddings(
        deployment="your_deployment_name",             # e.g., "text-embedding-ada-002"
        model="text-embedding-ada-002",
        openai_api_base="https://<your-resource-name>.openai.azure.com/",
        openai_api_version="2023-03-15-preview",
        openai_api_key="YOUR_AZURE_OPENAI_API_KEY"
    )

    # Create the semantic search tool instance
    search_tool = SemanticSearchTool(embeddings)

    query = "autonomous AI systems"
    results = search_tool._run(
        search_phrase=query,
        df=df,
        text_column="summary",
        id_column="ID",
        top_k=2,
        similarity_threshold=0.3
    )

    for doc_id, summary, score in results:
        print(f"ID: {doc_id} | Similarity: {score:.3f} | Summary: {summary}")
