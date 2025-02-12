import os
import logging
from tqdm import tqdm
from llama_index.node_parser import CodeSplitter, SentenceSplitter
from llama_index import Document as LlamaDocument
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

class BaseProcessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def process(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class CodeProcessor(BaseProcessor):
    def __init__(self, directory_path):
        super().__init__(directory_path)
        self.code_splitter = CodeSplitter(language="cpp", chunk_lines=30, chunk_lines_overlap=5, max_chars=1024)

    def process(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                code = file.read()

            llama_doc = LlamaDocument(text=code, metadata={"source": file_path})
            nodes = self.code_splitter.get_nodes_from_documents([llama_doc])
            return [{"content": node.text, "metadata": node.metadata} for node in nodes]

        except Exception as e:
            logger.error(f"Error processing code {file_path}: {str(e)}")
            return []

class TextProcessor(BaseProcessor):
    def __init__(self, directory_path):
        super().__init__(directory_path)
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    def process(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            splits = self.text_splitter.split_text(text)
            return [{"content": split, "metadata": {"source": file_path}} for split in splits]
        except Exception as e:
            logger.error(f"Error processing text {file_path}: {str(e)}")
            return []

class PDFProcessor(BaseProcessor):
    def __init__(self, directory_path):
        super().__init__(directory_path)

    def process(self, file_path):
        try:
            reader = PdfReader(file_path)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()

            return [{"content": text, "metadata": {"source": file_path}}]
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []

class MultiFormatDocumentProcessor:
    def __init__(self, directory_path, file_types):
        self.directory_path = directory_path
        self.file_types = file_types

        # Set processors based on file type
        self.processors = {
            'c': CodeProcessor(directory_path),
            'cpp': CodeProcessor(directory_path),
            'h': CodeProcessor(directory_path),
            'txt': TextProcessor(directory_path),
            'json': TextProcessor(directory_path),
            'pdf': PDFProcessor(directory_path),
            'cmake': TextProcessor(directory_path)  # Added support for cmake files
        }

    def process_directory(self):
        results = []
        unprocessed_files = []

        logger.info(f"Scanning directory {self.directory_path} for files...")

        for root, _, files in tqdm(os.walk(self.directory_path), desc="Processing Directory"):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1][1:].lower()  # Get file extension without dot

                if ext in self.file_types:  # Filter files based on extensions
                    results.extend(self.processors[ext].process(file_path))
                else:
                    #logger.warning(f"Unrecognized file type: {file_path}. Adding to unprocessed list.")
                    unprocessed_files.append(file_path)

        #if unprocessed_files:
            #logger.warning(f"The following files were not processed: {unprocessed_files}")

        return results

    def process(self):
        return self.process_directory()
