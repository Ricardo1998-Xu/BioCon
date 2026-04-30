import json
from pathlib import Path

METHOD_WORDS = [
    "algorithm", "approach", "method", "pipeline", "model", "propose", "introduce",
    "framework", "procedure", "strategy", "workflow"
]

IMPLEMENT_WORDS = [
    "implement", "implemented", "develop", "developed","implementation", "parameter", "hyperparameter",
    "build", "built", "construct", "use", "apply",
    "integrate", "run", "execute"
]

EVAL_WORDS = [
    "evaluate", "evaluated", "test", "tested",
    "benchmark", "compare", "performance",
    "accuracy", "precision", "recall", "result"
]


def classify_sentence(text):

    t = text.lower()

    for w in IMPLEMENT_WORDS:
        if w in t:
            return "implement"

    for w in METHOD_WORDS:
        if w in t:
            return "method"

    for w in EVAL_WORDS:
        if w in t:
            return "evaluation"

    return "other"


def sentence_type(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in in_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        out = []
        for item in data:
            label = classify_sentence(item["text"])
            item["chunk_type"] = label
            out.append(item)

        out_file = out_dir / f.name
        with open(out_file, "w", encoding="utf-8") as w:
            json.dump(out, w, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    in_dir = "BioCon/processed/sentence_chunks"
    out_dir = "BioCon/processed/sentence_chunks_typed"
    sentence_type(in_dir, out_dir)
