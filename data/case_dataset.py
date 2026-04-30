import os
import json
from tqdm import tqdm
from pathlib import Path

paper_dir = Path("case/processed/sentence_chunks_typed")
code_dir = Path("case/processed/code_chunks")


def load_repo_functions(repo_json_path):
    functions = []

    with open(repo_json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            functions.append({
                "function": item["function"],
                "code": item["code"]
            })

    return functions


def build_dataset():
    retrieval_data = []

    repos = [p.stem for p in paper_dir.glob("*.json")]
    total_samples = 0
    for repo in repos:
        samples = 0
        print(repo)
        paper_file = paper_dir / f"{repo}.json"
        code_file = code_dir / f"{repo}.json"

        functions = load_repo_functions(code_file)
        candidates = []
        for i, func in enumerate(functions):
            candidates.append(func["code"])

        sentences = json.load(open(paper_file, encoding="utf-8"))

        for item in tqdm(sentences, desc="paper"):

            if item["chunk_type"] not in ["method", "implement"]:
                continue

            retrieval_data.append({
                "sentence": item["text"],
                "candidates": candidates,
                "repo": repo
            })
            total_samples += 1
            samples += 1

        print(samples)
    print(total_samples)
    return retrieval_data


if __name__ == "__main__":
    OUTPUT_PATH = "case/case_dataset.jsonl"
    data = build_dataset()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in data:
           f.write(json.dumps(item, ensure_ascii=False) + "\n")