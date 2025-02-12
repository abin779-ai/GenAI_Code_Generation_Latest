from transformers import AutoTokenizer, T5EncoderModel, AutoModel
import torch
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Initialize CodeT5 (code documents)
code_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
code_model = T5EncoderModel.from_pretrained("Salesforce/codet5-base")

# Initialize sentence-transformers (non-code documents)
non_code_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
non_code_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def get_codet5_embedding_batch(codes):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code_model.to(device)

    inputs = code_tokenizer(codes, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = code_model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


def get_minilm_embedding_batch(texts):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    non_code_model.to(device)

    inputs = non_code_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = non_code_model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


class CustomEmbeddings:
    def __init__(self):
        pass

    def embed_documents(self, docs, doc_type, batch_size=32):

        embeddings = []
        total_batches = len(docs) // batch_size + (1 if len(docs) % batch_size != 0 else 0)

        for i in tqdm(range(0, len(docs), batch_size), desc="Embedding Documents", total=total_batches):
            batch = docs[i:i + batch_size]
            try:
                if doc_type == 'code':
                    batch_embeddings = get_codet5_embedding_batch(batch)
                else:
                    batch_embeddings = get_minilm_embedding_batch(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error while embedding batch {i}: {str(e)}")

        return embeddings


    def embed_query(self, text, doc_type):

        if doc_type == 'code':
            return get_codet5_embedding_batch([text]).squeeze(0)
        else:
            return get_minilm_embedding_batch([text]).squeeze(0)
