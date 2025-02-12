from utils import faiss_search


def generate_context(pkl, question, embedding_model, num_results):

    results = faiss_search.faiss_search(pkl, question, embedding_model, num_results)
    return results
