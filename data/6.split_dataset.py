import json
import random
from pathlib import Path
from collections import defaultdict

def split_BioCon(dataset_file, output_dir):
    random.seed(2)

    dataset_file = Path(dataset_file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    with open(dataset_file, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    repo_groups = defaultdict(list)

    for d in data:
        repo_groups[d["paper"]].append(d)

    repos = list(repo_groups.keys())
    random.shuffle(repos)
    train_r = 38
    valid_r = 5

    train_repos = repos[:train_r]
    valid_repos = repos[train_r:train_r + valid_r]
    test_repos = repos[train_r + valid_r:]

    train = []
    valid = []
    test = []

    for r in train_repos:
        train.extend(repo_groups[r])

    for r in valid_repos:
        valid.extend(repo_groups[r])

    for r in test_repos:
        test.extend(repo_groups[r])

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(valid, output_dir / "valid.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")


if __name__ == "__main__":
    dataset_file = Path("BioCon/Biocon.jsonl")
    output_dir = Path("BioCon")
