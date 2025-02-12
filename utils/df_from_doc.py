import os
import json
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from llama_index.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index import Document
from llama_index.node_parser import CodeSplitter
from llama_index import Document as LlamaDocument


class MultiFormatDocumentProcessor:
    def __init__(self, directory_path: str, file_type: str = None, json_directory: str = None):
        self.directory_path = directory_path
        self.file_type = file_type
        self.json_directory = json_directory
        self.supported_extensions = {
            'pdf': ['.pdf'],
            'c': ['.c', '.cpp', '.h', '.hpp'],
            'text': ['.txt', '.cmake'],  # Added .cmake here
            'json': ['.json'],
            'other': ['.py', '.java', '.js', '.html', '.css', '.xml', '.md']  # Add more as needed
        }

        self.pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        self.text_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20,
        )
        self.json_splitter = SimpleNodeParser()
        self.code_splitter = CodeSplitter(
            language="cpp",
            chunk_lines=30,
            chunk_lines_overlap=5,
            max_chars=1024,
        )

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splits = self.pdf_splitter.split_documents(pages)
        return [{"content": split.page_content, "metadata": split.metadata} for split in splits]

    def process_text(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        splits = self.text_splitter.split_text(text)
        return [{"content": split, "metadata": {"source": file_path}} for split in splits]

    def process_json(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            data = json.load(file)
        doc = Document(text=json.dumps(data), metadata={"source": file_path})
        nodes = self.json_splitter.get_nodes_from_documents([doc])
        return [{"content": node.text, "metadata": node.metadata} for node in nodes]

    def process_code(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            code = file.read()
        llama_doc = LlamaDocument(text=code, metadata={"source": file_path})
        nodes = self.code_splitter.get_nodes_from_documents([llama_doc])
        return [{"content": node.text, "metadata": node.metadata} for node in nodes]

    def process_directory(self) -> List[Dict[str, Any]]:
        results = []
        unprocessed_files = []
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                try:
                    if self.file_type == "pdf" and ext in self.supported_extensions['pdf']:
                        results.extend(self.process_pdf(file_path))
                    elif self.file_type == "c" and ext in self.supported_extensions['c']:
                        results.extend(self.process_code(file_path))
                    elif ext in self.supported_extensions['text']:
                        results.extend(self.process_text(file_path))
                    elif ext in self.supported_extensions['json']:
                        results.extend(self.process_json(file_path))
                    elif ext in self.supported_extensions['other']:
                        results.extend(self.process_text(file_path))  # Process other file types as text
                    else:
                        unprocessed_files.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    unprocessed_files.append(file_path)

        if unprocessed_files:
            print(f"Warning: The following files were not processed: {unprocessed_files}")

        return results

    def process(self) -> List[Dict[str, Any]]:
        return self.process_directory()