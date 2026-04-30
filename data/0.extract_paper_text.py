import requests
import os
from pathlib import Path


GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
def extract_paper_text(pdf_dir, out_dir):
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf in pdf_dir.glob("*.pdf"):
        with open(pdf, "rb") as f:
            files = {"input": f}
            r = requests.post(GROBID_URL, files=files)
            if r.status_code == 200:
                out_path = out_dir / (pdf.stem + ".tei.xml")
                out_path.write_text(r.text, encoding="utf-8")
                print(f"[OK] {pdf.name}")
            else:
                print(f"[FAIL] {pdf.name}")


if __name__ == "__main__":
    pdf_dir = "BioCon/raw/papers_pdf"
    out_dir = "BioCon/processed/tei_xml"
    extract_paper_text(pdf_dir, out_dir)
