# Code Generation-Based Question Answering System

This project uses NLP techniques to process various document types, create embeddings, and perform efficient similarity searches to generate relevant code for answering questions.

## Features

- Multi-format document processing (code, text, JSON, PDF)
- Embedding generation using CodeT5
- Efficient similarity search using FAISS
- Reranking of search results for improved relevance
- LLM-based answer generation using Ollama

## Components

1. `test_try.py`: Main script orchestrating the entire process
2. `config.json`: Configuration file for model settings and file paths
3. `document_processor.py`: Handles processing of different document types
4. `embeddings.py`: Manages the creation of embeddings using CodeT5
5. `create_search_index.py`: Creates a FAISS search index from document embeddings
6. `faiss_search.py`: Performs similarity search using the FAISS index
7. `generate_context.py`: Generates context for a given question using FAISS search

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the configuration:
   - Edit `config.json` to specify the correct paths for your code 
   - Adjust model settings if needed



## Usage

1. Run the main script:
   ```
   python test_try.py
   ```

2. The script will process the documents, create embeddings, and build the search index.

3. Once setup is complete, you can start asking questions. Type your question when prompted.

4. To exit the program, type 'exit' when prompted for a question.

## Customization

- To add support for additional file types, modify the `MultiFormatDocumentProcessor` class in `document_processor.py`
- To use a different embedding model, update the `CustomEmbeddings` class in `embeddings.py`
- To adjust the search parameters, modify the `faiss_search` function in `faiss_search.py`

## Troubleshooting

- If you encounter "out of memory" errors, try reducing the batch size in `embeddings.py`
- For issues with specific file types, check the corresponding processor in `document_processor.py`