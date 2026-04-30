# Do Papers Tell the Whole Story? A Benchmark and Framework for Uncovering Hidden Implementation Gaps in Bioinformatics

## 🚀 Overview
Ensuring consistency between research papers and their corresponding software implementations is essential for reproducibility, reliability, and scientific transparency.

However, in bioinformatics, discrepancies between method descriptions and actual code implementations are widespread and often overlooked.

To address this problem, we propose:
- 🔍 Paper–Code Consistency Detection — a novel cross-modal task.
- 📊 BioCon — the first benchmark dataset for this task.
  - 48 bioinformatics software projects
  - 6,780 sentence–code pairs
  - Hard negative sampling strategy
- 🧠 A unified framework for semantic alignment between paper text and code.
  - sentence-level classification
  - cross-modal retrieval
  - project-level consistency assessment
- 🔬 Real-world Case Study
  - Analysis of 23 recent bioinformatics projects reveals:
    - up to 40%+ inconsistency rates in some projects
    - missing key components (e.g., loss functions, evaluation metrics)

---

## 📂 Repository Structure
The repository is organized to support dataset construction, model training, and multi-level evaluation, including sentence-level classification, cross-modal retrieval, and project-level consistency analysis.

```bash
│── 📁 data/                         # Datasets and data processing resources
│   ├── 📂 BioCon/                   # BioCon benchmark dataset
│   │   ├── 📂 raw/          
│   │   │   ├── 📂 papers_pdf/       # Raw research papers (PDF format)
│   │   │   ├── 📂 repos/            # Corresponding source code repositories
│   │   ├── 📂 processed/            # Intermediate processed artifacts
│   │   ├── 📜 train.jsonl           # Training set
│   │   ├── 📜 test.jsonl            # Test set (sentence-level classification)
│   │   ├── 📜 valid.jsonl           # Validation set
│   │   ├── 📜 BioCon.jsonl          # Full dataset (train + valid + test)
│   │   ├── 📜 retrieval_in-test.jsonl   # Intra-project retrieval dataset
│   │   ├── 📜 software_test.jsonl   # Project-level consistency evaluation dataset
│   │   └── ...
│   │
│   ├── 📂 case/                     # Case study dataset
│   │   ├── 📂 raw/          
│   │   │   ├── 📂 papers_pdf/       # Raw papers used in case study
│   │   │   ├── 📂 repos/            # Corresponding repositories
│   │   ├── 📂 processed/            # Processed case study artifacts
│   │   └── 📜 case_dataset.jsonl    # Constructed case study dataset  
│   │
│   ├── 📜 0.extract_paper_text.py   # Extract sentence-level text from papers          
│   ├── 📜 1.tei_to_json.py          # Convert TEI XML to structured JSON
│   ├── 📜 retrieval_dataset.py      # Construct retrieval datasets
│   └── ...
│
│── 📜 model.py                      # Model architecture definition
│── 📜 run.py                        # Training script
│── 📜 test.py                       # Sentence-level classification evaluation
│── 📜 retrieval_test.py             # Retrieval evaluation
│── 📜 project_test.py               # Project-level consistency evaluation
│── 📜 case_test.py                  # Case study analysis
│── 📜 environment.yaml              # Environment configuration
│── 📜 README.md                     
└── ...
```
