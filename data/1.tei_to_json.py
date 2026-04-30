from bs4 import BeautifulSoup
from pathlib import Path
import json
import re
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

KEEP_SECTIONS = [
    "method", "methods", "materials and methods", "approach", "protocol", "implementation", "analysis",
    "experiment", "experiments", "results", "benchmark", "replication", "software", "tool", "application", "methodology",
    "module", "evaluation", "data processing", "objectives", "objective", "architecture"
]

DROP_SECTIONS = [
    "introduction", "background", "discussion", "conclusion", "acknowledgement", "funding", "author",
    "ethics", "consent", "data availability", "code availability", "supplementary", "additional", "references",
]


def normalize_head(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9+ ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def section_decision(head_text):
    h = normalize_head(head_text)
    for d in DROP_SECTIONS:
        if d in h:
            return "drop"
    for k in KEEP_SECTIONS:
        if k in h:
            return "keep"
    return "unknown"


def split_sentences(text):
    """sentence tokenizer"""
    sentences = nltk.sent_tokenize(text)
    return sentences


def tei_to_json(tei_dir, out_dir):
    tei_dir = Path(tei_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tei_file in tei_dir.glob("*.tei.xml"):
        soup = BeautifulSoup(tei_file.read_text(encoding="utf-8"), "lxml-xml")

        body = soup.find("body")
        if body is None:
            print(f"[SKIP] No body in {tei_file.stem}")
            continue

        chunks = []
        current_state = "unknown"
        current_head = None

        paragraph_id = 0
        sentence_id = 0

        for div in body.find_all("div"):
            head = div.find("head", recursive=False)
            if head:
                current_state = section_decision(head.text)
                current_head = head.get_text(strip=True)

            if current_state == "keep":
                for p in div.find_all("p", recursive=False):
                    paragraph_text = p.get_text(" ", strip=True)
                    if len(paragraph_text) < 40:
                        continue

                    sentences = split_sentences(paragraph_text)
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) < 10:
                            continue

                        if len(sent) > 300:
                            continue

                        chunks.append({
                            "section_head": current_head,
                            "paragraph_id": paragraph_id,
                            "sentence_id": sentence_id,
                            "text": sent
                        })

                        sentence_id += 1

                    paragraph_id += 1

        with open(out_dir / f"{tei_file.stem}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    tei_dir = "BioCon/processed/tei_xml"
    out_dir = "BioCon/processed/sentence_chunks"
    tei_to_json(tei_dir, out_dir)
