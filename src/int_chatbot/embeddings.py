import json
import shutil
from pathlib import Path

import torch
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange


class JinaEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        device: str = "cpu",
        torch_compile: bool = False,
        batch_size: int = 8,
        max_seq_length: int = 8192,
    ):
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, model_kwargs={"attn_implementation": "eager"}
        ).to(device)

        if torch_compile:
            self.model = torch.compile(self.model)

        self.model.eval()
        self.model.max_seq_length = max_seq_length

        self.max_seq_length = max_seq_length
        self.device = self.model.device
        self.batch_size = batch_size

    @torch.inference_mode()
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, task="retrieval.query").tolist()

    @torch.inference_mode()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeds = []
        for start_idx in trange(
            0, len(texts), self.batch_size, desc=f"Embedding documents with device {self.device}", unit="batch"
        ):
            batch = texts[start_idx : start_idx + self.batch_size]
            batch_embeds = self.model.encode(batch, task="retrieval.passage").tolist()
            embeds.extend(batch_embeds)
        return embeds


def build_vector_store(
    jsonl_file: str | Path,
    dir_store_name: str = "vector_store",
    overwrite: bool = False,
    embedder_device: str = "cpu",
    batch_size: int = 8,
    torch_compile: bool = False,
):
    """
    Build a FAISS vector store on the paragraph level.

    Args:
        jsonl_file: str | Path, the path to the JSONL file
        dir_store_name: str, the directory name to store the FAISS index
        overwrite: bool, whether to overwrite the existing store
        embedder_device: str, the device to use for the embeddings
        batch_size: int, the batch size for the embeddings
        torch_compile: bool, whether to compile the model

    Returns:
        FAISS: the FAISS index
    """
    pfvector = Path(dir_store_name)

    embeddings = JinaEmbeddings(device=embedder_device, batch_size=batch_size, torch_compile=torch_compile)

    if pfvector.exists():
        if overwrite:
            shutil.rmtree(pfvector)
        else:
            return FAISS.load_local(dir_store_name, embeddings=embeddings, allow_dangerous_deserialization=True)

    # The document's main content conists of the required columns in key: value format with newlines between them
    # required_columns = ["titel", "auteur_naam", "datum", "inhoud", "excerpt"]
    optional_columns = ["id", "link", "titel", "auteur_naam", "datum"]

    print("🔨 Voorbereiden van documenten voor de RAG vector store...", flush=True)

    documents = []
    with open(jsonl_file, "r") as fhin:
        for line in tqdm(fhin, unit="document"):
            row = json.loads(line)
            text = row["inhoud"] if row["inhoud"] else row["excerpt"]

            for para_idx, para in enumerate(text.split("\n")):
                para = para.strip()
                metadata = {k: row[k] for k in optional_columns}
                metadata["paragraph_idx"] = para_idx

                doc = Document(page_content=para, metadata=metadata)
                documents.append(doc)

    print("🔨 Bouwen van vector store voor RAG...", flush=True)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(dir_store_name)

    return vector_store
