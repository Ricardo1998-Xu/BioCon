import json
import random
from pathlib import Path
from tqdm import tqdm


def BioCon(retrieval_dir, output_file):
    retrieval_dir = Path(retrieval_dir)
    output_file = Path(output_file)
    NEG_PER_POS = 3

    def load_all_functions():
        funcs = []
        for f in retrieval_dir.glob("*.json"):
            repo = f.stem
            data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
            for item in data:
                for fn in item["top_functions"]:
                    fn["repo"] = repo
                    funcs.append(fn)

        return funcs

    all_funcs = load_all_functions()

    dataset = []

    for file in retrieval_dir.glob("*.json"):
        repo = file.stem
        data = json.loads(file.read_text(encoding="utf-8"))

        for item in tqdm(data):
            sentence = item["sentence"]
            top_funcs = item["top_functions"]

            if len(top_funcs) == 0:
                continue

            best = top_funcs[0]

            # ---- positive ----
            dataset.append({
                "paper": repo,
                "sentence": sentence,
                "repo": repo,
                "function": best["function"],
                "code": best["code"],
                "label": 1
            })

            # ---- hard negatives ----
            hard_negs = random.sample(top_funcs[5:10], 2)
            for neg in hard_negs:
                dataset.append({
                    "paper": repo,
                    "sentence": sentence,
                    "repo": repo,
                    "function": neg["function"],
                    "code": neg["code"],
                    "label": 0
                })

            # ---- random negatives ----
            for _ in range(NEG_PER_POS):
                neg = random.choice(all_funcs)
                if neg["repo"] != repo:
                    dataset.append({
                        "paper": repo,
                        "sentence": sentence,
                        "repo": neg.get("repo", "unknown"),
                        "function": neg["function"],
                        "code": neg["code"],
                        "label": 0
                    })
                else:
                    neg1 = random.choice(all_funcs)
                    dataset.append({
                        "paper": repo,
                        "sentence": sentence,
                        "repo": neg1.get("repo", "unknown"),
                        "function": neg1["function"],
                        "code": neg1["code"],
                        "label": 0
                    })

    with open(output_file, "w", encoding="utf-8") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")

    print("Saved to", output_file)


if __name__ == "__main__":
    retrieval_dir = "BioCon/processed/retrieval_pairs_verification"
    output_file = "BioCon/Biocon.jsonl"
    BioCon(retrieval_dir, output_file)
