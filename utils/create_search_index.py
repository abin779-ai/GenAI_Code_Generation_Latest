import faiss
import numpy as np
import logging
from embeddings import CustomEmbeddings

logger = logging.getLogger(__name__)


def create_search_index(docs, doc_type, show_progress=False):
    logger.info(f"Creating FAISS search index for {doc_type} documents")

    embeddings = []
    metadata = []
    custom_embeddings = CustomEmbeddings()

    try:
        texts = [doc['content'] for doc in docs]
        embeddings = custom_embeddings.embed_documents(texts, doc_type, batch_size=8)

        if not embeddings:
            logger.error("No embeddings were generated. Check the document processing step.")
            return None

    except Exception as e:
        logger.error(f"Error while embedding documents: {str(e)}")
        return None

    metadata = [doc['metadata'] for doc in docs]
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Validate that the embedding array is non-empty
    if embeddings_array.shape[0] == 0:
        logger.error("Embedding array is empty. FAISS index cannot be created.")
        return None

    d = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_array)

    faiss_index_path = f"faiss_{doc_type}.index"
    try:
        faiss.write_index(index, faiss_index_path)
        logger.info(f"FAISS index created and written to {faiss_index_path}")
    except Exception as e:
        logger.error(f"Error writing FAISS index to {faiss_index_path}: {str(e)}")
        return None

    return faiss_index_path
