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

---
## 📊 BioCon: Data Format
Each sample consists of:
```JSON
{
  "paper": "...",
  "repo": "...",
  "sentence": "...",
  "code": "...",
  "label": 0 or 1
}
```

---

## 💻 Experiments
### 📥 Install
```sh
conda env create -f environment.yml
```

### 🚀 Training
```sh
python run.py \
    --num_labels=2 \
    --train_data_file=./data/BioCon/train.jsonl \
    --eval_data_file=./data/BioCon/valid.jsonl \
    --output_dir=./saved_models \
    --runs_path=./runs \
    --model_type=UniXcoder \
    --tokenizer_name=./pre \
    --model_name_or_path=./pre \
    --do_train \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --evaluate_during_training \
    --seed 123456
```

### 📈 Evaluation
#### Sentence-level
```sh
python test.py \
    --test_data_file=./data/BioCon/test.jsonl \
    --output_dir=./saved_models \
    --results_path=./results \
    --model_type=UniXcoder \
    --tokenizer_name=./pre \
    --model_name_or_path=./pre \
    --do_test \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456
```

#### Retrieval-level
```sh
python retrieval_test.py \
    --test_data_file=./data/BioCon/retrieval_in-test.jsonl \
    ...
```

#### Project-level
```sh
python retrieval_test.py \
    --test_data_file=./data/BioCon/software_test.jsonl \
    ...
```

#### Case Study
```sh
python case_test.py \
    --test_data_file=./data/case/case_dataset.jsonl \
    ...
```

## 🔮 Future Work
- Cross-project generalization
- Fine-grained consistency modeling
- Improved sentence filtering
- Integration with structured code representations

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
