import json
from tqdm import tqdm


def load_in_repo_functions(repo_json_path):
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

def build_in_retrieval_dataset(test_path, repo_dir):
    retrieval_data = []

    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        sample = json.loads(line)
        if sample["label"] == 1:
            paper = sample["paper"]
            sentence = sample["sentence"]
            gt_function = sample["function"]

            repo_json_path = f"{repo_dir}/{paper}.json"
            functions = load_in_repo_functions(repo_json_path)

            candidates = []
            gt_index = -1

            for i, func in enumerate(functions):
                candidates.append(func["code"])

                if func["code"].strip() == sample["code"].strip():
                    gt_index = i

            if gt_index == -1:
                print(f"[Warning] GT not found for {paper}")
                continue

            retrieval_data.append({
                "query": sentence,
                "candidates": candidates,
                "gt_index": gt_index,
                "repo": paper
            })

    return retrieval_data


def build_cross_retrieval_dataset(test_path, repo_dir):
    retrieval_data = []
    functions = load_cross_repo_functions(repo_dir)

    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        sample = json.loads(line)
        if sample["label"] == 1:
            paper = sample["paper"]
            sentence = sample["sentence"]

            candidates = []
            gt_index = -1

            for i, func in enumerate(functions):
                candidates.append(func["code"])

                if func["code"].strip() == sample["code"].strip():
                    gt_index = i

            if gt_index == -1:
                print(f"[Warning] GT not found for {paper}")
                continue

            retrieval_data.append({
                "query": sentence,
                "candidates": candidates,
                "gt_index": gt_index,
                "repo": paper
            })

    return retrieval_data


def load_cross_repo_functions(repo_dir):
    functions = []
    paper = ["cruzdb", "knn_smoothing", "svtools", "browsevcf", "bamsurgeon"]
    for i in paper:
        repo_json_path = f"{repo_dir}/{i}.json"
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


if __name__ == "__main__":
    code_path = "BioCon/processed/code_chunks"
    test_path = "BioCon/test.jsonl"

    in_retrieval_data = build_in_retrieval_dataset(test_path, code_path)
    with open("BioCon/retrieval_in-test.json", "w", encoding="utf-8") as f:
        json.dump(in_retrieval_data, f, ensure_ascii=False, indent=2)

    cross_retrieval_data = build_cross_retrieval_dataset(test_path, code_path)
    with open("BioCon/retrieval_cross-test.json", "w", encoding="utf-8") as f:
        json.dump(cross_retrieval_data, f, ensure_ascii=False, indent=2)
