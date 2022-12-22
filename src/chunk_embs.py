import torch
import pickle
from haystack.schema import Document
from tqdm import tqdm

chunksize = int(1e6)

with open(f"./cache/tct_colbert_embs.pickle", "rb") as docs_file:
    docs: list[Document] = pickle.load(docs_file)
    chunk = torch.zeros((chunksize, 768))
    chunk_nr = 0
    chunk_str = f"./cache/emb_chunks/tct_colbert_embs_chunk_{chunk_nr}.pt"
    for i, doc in enumerate(tqdm(docs)):
        emb = torch.tensor(doc.embedding).unsqueeze(0)
        chunk[i % chunksize] = emb
        if (i % chunksize) + 1 == chunk.shape[0]:
            # save chunk 
            torch.save(chunk, chunk_str)
            chunk_nr += 1
            chunk_str = f"./cache/emb_chunks/tct_colbert_embs_chunk_{chunk_nr}.pt"
            chunk = torch.zeros((min(chunksize, len(docs) - (i + 1)), 768))