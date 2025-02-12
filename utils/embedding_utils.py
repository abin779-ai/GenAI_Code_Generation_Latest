import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import pandas as pd


class EmbeddingUtils:
    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def batch_get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[torch.Tensor]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return embeddings


def faiss_search(pkl: bytes, question: str, embedding_utils: EmbeddingUtils, num_results: int) -> List[Dict[str, Any]]:
    search_index = FAISS.deserialize_from_bytes(
        embeddings=embedding_utils.get_embedding,
        serialized=pkl
    )
    context = search_index.similarity_search_with_score(question, k=num_results)
    return context


def create_search_index(docs: pd.DataFrame, embedding_utils: EmbeddingUtils) -> bytes:
    documents = [
        Document(page_content=row["text"], metadata={"page_number": row["page_number"]})
        for row in docs.to_dict('records')
    ]
    embeddings = embedding_utils.batch_get_embeddings([doc.page_content for doc in documents])
    search_index = FAISS.from_embeddings(embeddings, documents)
    return search_index.serialize_to_bytes()


# Example usage
if __name__ == "__main__":
    # This is just a demonstration, you would typically use this in your main application
    embedding_utils = EmbeddingUtils()

    # Example DataFrame
    example_docs = pd.DataFrame({
        "text": ["This is a sample text", "Another example", "Code snippet here"],
        "page_number": [1, 2, 3]
    })

    # Create search index
    pkl = create_search_index(example_docs, embedding_utils)

    # Perform a search
    question = "What is the sample text?"
    results = faiss_search(pkl, question, embedding_utils, num_results=2)

    for doc, score in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Similarity Score: {score}")
        print("---")