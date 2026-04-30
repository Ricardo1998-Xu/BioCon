import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import faiss

def build_paper_text(sentence):
    return f"""
Method description from a bioinformatics paper:
{sentence}
"""


def build_code_text(chunk):
    doc = chunk["docstring"] or ""
    return f"""
Python function from a bioinformatics tool.
Function name:
{chunk["function"]}
Documentation:
{doc}
Code:
{chunk["code"]}
"""


def retrieve_pairs(sentence_dir, code_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "./pre"
    TOPK = 10

    paper_dir = Path(sentence_dir)
    code_dir = Path(code_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    def embed(text):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            output = model(**tokens)

        vec = output.last_hidden_state[:, 0, :]
        return vec.cpu().numpy()[0]


    repos = [p.stem for p in paper_dir.glob("*.json")]
    for repo in repos:
        print("\nProcessing repo:", repo)

        paper_file = paper_dir / f"{repo}.json"
        code_file = code_dir / f"{repo}.json"
        out_file = out_dir / f"{repo}.json"

        if not code_file.exists():
            print("code chunks missing:", repo)
            continue

        paper_vecs = []
        paper_meta = []
        sentences = json.load(open(paper_file, encoding="utf-8"))

        for item in tqdm(sentences, desc="paper"):
            if item["chunk_type"] not in ["method", "implement"]:
                continue

            text = build_paper_text(item["text"])
            vec = embed(text)
            paper_vecs.append(vec)
            paper_meta.append(item)

        code_vecs = []
        code_meta = []
        with open(code_file, encoding="utf-8") as f:
            for line in tqdm(f, desc="code"):
                chunk = json.loads(line)
                text = build_code_text(chunk)
                vec = embed(text)
                code_vecs.append(vec)
                code_meta.append(chunk)

        # FAISS
        paper_vecs = np.array(paper_vecs)
        code_vecs = np.array(code_vecs)

        # --- normalize for cosine similarity ---
        faiss.normalize_L2(paper_vecs)
        faiss.normalize_L2(code_vecs)
        dim = code_vecs.shape[1]

        # --- FAISS index ---
        index = faiss.IndexFlatIP(dim)
        index.add(code_vecs)

        scores, ids = index.search(paper_vecs, TOPK)
        final_output = []
        for i in tqdm(range(len(paper_meta))):
            top_funcs = []
            for score, idx in zip(scores[i], ids[i]):
                target_code = code_meta[idx]
                top_funcs.append({
                    "function": target_code["function"],
                    "class": target_code["class"],
                    "file": target_code["file"],
                    "score": float(score),
                    "code": target_code["code"]
                })

            final_output.append({
                "sentence": paper_meta[i]["text"],
                "chunk_type": paper_meta[i]["chunk_type"],
                "top_functions": top_funcs
            })

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        print(f"Done! Results saved to: {out_file}")


if __name__ == "__main__":
    sentence_dir = "BioCon/processed/sentence_chunks_typed"
    code_dir = "BioCon/processed/code_chunks"
    out_dir = "BioCon/processed/retrieval_pairs"
    retrieve_pairs(sentence_dir, code_dir, out_dir)
