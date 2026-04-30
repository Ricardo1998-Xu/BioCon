import json
import random
from collections import defaultdict
import argparse


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def group_by_software(data):
    software_dict = defaultdict(list)
    for item in data:
        software = item["paper"]
        software_dict[software].append(item)
    return software_dict


def compute_software_metrics(statements):
    total = len(statements)
    num_consistent = sum(1 for s in statements if s["label"] == 1)
    num_inconsistent = total - num_consistent

    consistency_ratio = num_consistent / total if total > 0 else 0.0

    label = 0 if num_inconsistent > 0 else 1

    return {
        "num_statements": total,
        "num_consistent": num_consistent,
        "num_inconsistent": num_inconsistent,
        "consistency_ratio": round(consistency_ratio, 4),
        "label": label
    }


def build_clean_variant(statements):
    return [s for s in statements if s["label"] == 1]


def build_mixed_variant(statements, ratio):
    consistent = [s for s in statements if s["label"] == 1]
    inconsistent = [s for s in statements if s["label"] == 0]

    total = len(statements)
    target_inconsistent = int(total * ratio)

    sampled_inconsistent = random.sample(
        inconsistent, min(len(inconsistent), target_inconsistent)
    )

    remaining = total - len(sampled_inconsistent)
    sampled_consistent = random.sample(
        consistent, min(len(consistent), remaining)
    )

    mixed = sampled_consistent + sampled_inconsistent
    random.shuffle(mixed)

    return mixed


def build_dataset(input_path, output_path, seed=42):
    random.seed(seed)

    data = load_jsonl(input_path)
    software_dict = group_by_software(data)

    output = []

    for software, statements in software_dict.items():

        metrics = compute_software_metrics(statements)
        output.append({
            "software": software,
            "variant": "original",
            "statements": statements,
            **metrics
        })

        # ===== clean =====
        clean_statements = build_clean_variant(statements)
        if len(clean_statements) > 0:
            metrics = compute_software_metrics(clean_statements)
            output.append({
                "software": software,
                "variant": "clean",
                "statements": clean_statements,
                **metrics
            })

        # ===== mixed =====
        for ratio in [0.1, 0.3]:
            mixed_statements = build_mixed_variant(statements, ratio)
            metrics = compute_software_metrics(mixed_statements)
            output.append({
                "software": software,
                "variant": f"mixed_{int(ratio*100)}",
                "statements": mixed_statements,
                **metrics
            })

    # jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    print(f"Done! Saved to {output_path}")
    print(f"Total software instances: {len(output)}")


if __name__ == "__main__":
    input_json = "BioCon/test.jsonl"
    output_json = "BioCon/software_test.jsonl"

    build_dataset(input_json, output_json)
