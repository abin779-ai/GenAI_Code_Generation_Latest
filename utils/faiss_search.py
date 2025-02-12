import faiss
import logging
from embeddings import CustomEmbeddings

logger = logging.getLogger(__name__)


def faiss_search(faiss_index_path, question, doc_type, num_results):

    custom_embeddings = CustomEmbeddings()

    # Generate the query embedding (CodeT5 or MiniLM)
    query_embedding = custom_embeddings.embed_query(question, doc_type)

    # Load the FAISS index
    search_index = faiss.read_index(faiss_index_path)

    # query embedding match the dimensionality of the FAISS index?
    if query_embedding.shape[0] != search_index.d:
        logger.error(
            f"Dimensionality mismatch: FAISS index has {search_index.d} dimensions, but the query has {query_embedding.shape[0]} dimensions.")
        raise ValueError("Dimensionality mismatch between FAISS index and query embedding.")

    # Perform similarity search
    distances, indices = search_index.search(query_embedding.reshape(1, -1), num_results)

    # Return documents with the nearest neighbors
    results = [(index, distance) for index, distance in zip(indices[0], distances[0])]
    return results
