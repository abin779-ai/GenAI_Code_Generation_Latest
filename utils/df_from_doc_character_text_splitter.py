import warnings
warnings.filterwarnings('ignore')
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.document_loaders import TextLoader, JSONLoader
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from utils import clean_text, est_words_tokens
import pandas as pd
import os

def df_from_doc(filepath, filetype,directory):
    if filetype == "pdf":
        #loader = PyPDFLoader(filepath)


        docs = []
        if filepath:
            loader = DirectoryLoader(filepath)
            

            # for loader in loaders:
            #  print("Loading raw document..." + loader.file_path)
            raw_documents = loader.load()

            # print("Splitting text...")
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
            )

            #print('text_splitter',text_splitter)
            documents = text_splitter.split_documents(raw_documents)
            docs.extend(documents)
        if directory:

           # loaders = [JSONLoader(os.path.join(directory, fn)) for fn in os.listdir(directory)]

            json_folder = os.listdir(directory)

            for i in json_folder:
                loader = JSONLoader(
                                file_path=directory+'/'+i,
                                jq_schema='.',
                                text_content=False)


            #  print("Loading raw document..." + loader.file_path)
                raw_documents = loader.load()

            # print("Splitting text...")
                text_splitter = CharacterTextSplitter(
                    separator="\n\n",
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len,
                )

                #print('text_splitter',text_splitter)
                documents = text_splitter.split_documents(raw_documents)
                docs.extend(documents)

        docs = pd.DataFrame(text_splitter.split_documents(docs), columns = ['text', 'page_number'])
        docs["text"] = docs["text"].apply(lambda x: x[1])#.apply(clean_text.clean_text);

    elif filetype == "txt":
        pdf_folder_path = filepath
        loaders = [TextLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
        docs = []

        for loader in loaders:
          #  print("Loading raw document..." + loader.file_path)
            raw_documents = loader.load()

           # print("Splitting text...")
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
            )

            #print('text_splitter',text_splitter)
            documents = text_splitter.split_documents(raw_documents)
            docs.extend(documents)

        print('______',len(docs))
        #loader = TextLoader(filepath)
        #documents = loader.load()
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1028, chunk_overlap=128)
        docs = pd.DataFrame(text_splitter.split_documents(docs), columns = ['text', 'page_number'])
       # docs["text"] = docs["text"].apply(lambda x: x[1]).apply(clean_text.clean_text); docs["page_number"] = 1
        docs["text"] = docs["text"].apply(lambda x: x[1])#.apply(clean_text.clean_text); docs["page_number"] = 1

        #docs = est_words_tokens.est_words_tokens(docs)
    return docs
