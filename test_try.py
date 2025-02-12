import warnings
import logging
import os
import pickle
import time
import json
import ollama
from tqdm import tqdm
from document_processor import MultiFormatDocumentProcessor
from utils import faiss_search, generate_context
from utils.create_search_index import create_search_index
from sentence_transformers import CrossEncoder

# CrossEncoder for reranking
RERANKING_MODEL = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

EMBEDDING_MODEL_CODE = config["embedding_model"]["code"]
EMBEDDING_MODEL_NON_CODE = config["embedding_model"]["non_code"]
LLM_MODEL = config["llm_model"]
CODE_VECTOR_STORE = config["vector_stores"]["code"]
OTHER_VECTOR_STORE = config["vector_stores"]["other"]
DIRECTORY_PATH = config["directories"]["source_code_directory"]

def create_vector_stores():
    # Check embedding creation
    if os.path.exists(CODE_VECTOR_STORE):
        logger.info(f"{CODE_VECTOR_STORE} already exists. Skipping code vector store creation.")
    else:
        logger.info(f"{CODE_VECTOR_STORE} not found. Creating code vector store.")
        # Create code vector store
        process_and_create_vector_store("code", ["c", "cpp", "h", "cmake"], DIRECTORY_PATH)

    if os.path.exists(OTHER_VECTOR_STORE):
        logger.info(f"{OTHER_VECTOR_STORE} already exists. Skipping other vector store creation.")
    else:
        logger.info(f"{OTHER_VECTOR_STORE} not found. Creating other vector store.")
        # Create other vector store
        process_and_create_vector_store("other", ["txt", "json", "pdf"], DIRECTORY_PATH)


def process_and_create_vector_store(doc_type, file_types, source_path):

    logger.info(f'Processing {doc_type} files...')

    # a document processor - only processes the specified file types
    processor = MultiFormatDocumentProcessor(source_path, file_types)
    docs = processor.process()  # will filter out files not matching the given extensions

    if len(docs) == 0:
        logger.warning(f"No documents found for {doc_type}. Skipping FAISS index creation.")
        return None

    logger.info(f'Creating embeddings for {len(docs)} {doc_type} documents...')

    # appropriate embedding model
    embedding_model = EMBEDDING_MODEL_CODE if doc_type == "code" else EMBEDDING_MODEL_NON_CODE

    # Create FAISS index
    faiss_index_path = create_search_index(docs=docs, doc_type=doc_type, show_progress=True)

    vector_store = CODE_VECTOR_STORE if doc_type == "code" else OTHER_VECTOR_STORE
    with open(vector_store, 'wb') as f:
        pickle.dump(faiss_index_path, f)

    logger.info(f'Saved {doc_type} vector store to {vector_store}')
    return faiss_index_path


def load_vector_stores():
    logger.info('Loading data from stored files...')
    vector_stores = {}

    # Load code vector store if it exists
    if os.path.exists(CODE_VECTOR_STORE):
        with open(CODE_VECTOR_STORE, 'rb') as f:
            vector_stores["code"] = pickle.load(f)
        logger.info(f'Loaded code vector store from {CODE_VECTOR_STORE}')
    else:
        logger.error(f'Code vector store {CODE_VECTOR_STORE} not found. Please create embeddings first.')
        return None, None

    # Load other vector store if it exists, otherwise assign None
    if os.path.exists(OTHER_VECTOR_STORE):
        with open(OTHER_VECTOR_STORE, 'rb') as f:
            vector_stores["other"] = pickle.load(f)
        logger.info(f'Loaded other vector store from {OTHER_VECTOR_STORE}')
    else:
        logger.warning(f'{OTHER_VECTOR_STORE} not found. Assigning empty vector store for "other".')
        vector_stores["other"] = None

    # Validate if the code vector store was correctly loaded
    if not vector_stores["code"]:
        logger.error("Code vector store is empty. Unable to proceed with search.")
        return None, None

    return vector_stores["code"], vector_stores["other"]

def search_and_rerank(question, code_pkl, other_pkl, docs):
    logger.info('Starting search and rerank process')
    faiss_start = time.time()

    contexts = {}

    # Search codet5 embeddings (768 dimensions)
    if code_pkl is not None:
        logger.info("Searching in code vector store...")
        contexts["code"] = generate_context.generate_context(code_pkl, question, "code", num_results=3)

    # Search MiniLM embeddings (384 dimensions)
    if other_pkl is not None:
        logger.info("Searching in 'other' vector store...")
        contexts["other"] = generate_context.generate_context(other_pkl, question, "non_code", num_results=2)
    else:
        logger.warning('No "other" vector store found. Skipping search in "other" files.')

    if not contexts.get("code") and not contexts.get("other"):
        logger.warning("No valid vector stores found or no results in stores. Unable to perform search.")
        return []

    combined_context = contexts.get("code", []) + contexts.get("other", [])

    # Fetch and rerank
    ranking_list = []
    for context_item in combined_context:
        doc_index = context_item[0]
        if doc_index < len(docs):
            document_content = docs[doc_index]['content']
            ranking_list.append(document_content)
        else:
            logger.error(f"Document index {doc_index} out of bounds. Skipping.")

    if not ranking_list:
        logger.warning("No documents to rerank. Skipping reranking.")
        return []

    logger.info('Reranking combined results')
    results = RERANKING_MODEL.rank(question, ranking_list, return_documents=True, top_k=10)

    final_reranking_list = [r['text'] for r in results if r['score'] > 0.20]

    if not final_reranking_list and results:
        final_reranking_list = [r['text'] for r in results]  # Allow results even if scores are equal

        # Log small portion of relevant context for debugging
        truncated_context = [ctx[:200] + "..." if len(ctx) > 200 else ctx for ctx in
                             final_reranking_list]  # Limit to 200 chars
        logger.info(f"Search and rerank completed in {time.time() - faiss_start:.2f} seconds")
        logger.info(f"Relevant context (truncated): {truncated_context}")  # Log truncated content

        return final_reranking_list


def generate_answer(question, context):
    """
    Generate a response with appropriate C code and explanation using the LLM, based on the provided context and question.
    """
    logger.info('Generating answer using LLM')

    # Enhanced prompt template for the LLM
    template = f"""
    You are a skilled AI assistant specializing in generating C code based on a provided source code base and documentation. 

    Below is some context extracted from code and documentation files. Your task is to:
    1. Analyze the context carefully and generate C code based on the user's question.
    2. If the context provides enough information, use it directly to generate accurate and functional C code.
    3. After generating the code, briefly explain its functionality and the reasoning behind your implementation.
    4. If the provided context does not contain enough details, use your general knowledge of C programming to complete the code, but make sure to clarify which parts are inferred from the context and which parts are based on your general knowledge.

    Context:
    {context}

    User Question: {question}

    Instructions for you:
    1. Provide relevant and functional C code as a response to the user's question.
    2. If applicable, suggest any improvements or best practices related to the code.
    3. After the code, provide a concise explanation of how it works and its purpose.

    Answer:
    """

    # Call the LLM
    ollama_start = time.time()
    response = ollama.chat(model=LLM_MODEL, messages=[
        {
            'role': 'system',
            'content': 'You are an AI assistant specializing in software development, especially C programming and codebase analysis.',
        },
        {
            'role': 'user',
            'content': template,
        },
    ])
    ollama_stop = time.time()

    # Log the LLM's response & time taken
    logger.info('LLM response received')
    print('\nLLM response:\n\n', response['message']['content'])
    print('_____________________________________________________________')
    logger.info(f"LLM answer generated in {ollama_stop - ollama_start:.2f} seconds")


def main():
    logger.info('Starting the QA process')

    # if pickle files exist, if not, create vector stores
    create_vector_stores()

    # Load the vector stores
    code_pkl, other_pkl = load_vector_stores()

    # If code_pkl is None, exit (wasn't created or loaded)
    if code_pkl is None:
        logger.error("Failed to load vector stores. Exiting.")
        return

    # Process documents to get the original documents
    processor = MultiFormatDocumentProcessor(DIRECTORY_PATH, ['c', 'cpp', 'h', 'txt', 'json', 'pdf', 'cmake'])
    docs = processor.process()  # returns a list of documents

    while True:
        # Get user question
        question = input('\nType your question (or "exit" to quit): ').strip()
        if question.lower() == 'exit':
            break
        logger.info(f'Received user question: "{question}"')

        # Search and rerank
        context = search_and_rerank(question, code_pkl, other_pkl, docs)  # Pass the docs list
        logger.info('Search and rerank complete')
        print("\nRelevant context:\n", context)

        # Generate answer
        generate_answer(question, context)
        logger.info('Answer generation complete')

    logger.info('QA process ended')

if __name__ == "__main__":
    main()
