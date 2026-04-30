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
```bash

│── 📁 data/                     # Contains datasets used in the study
│   ├── 📂 BioCon/                  # The BioCon dataset
│   │   ├── 📜 train.jsonl          # Training dataset
│   │   ├── 📜 test.jsonl           # Testing dataset (Sentence-level classification)
│   │   ├── 📜 valid.jsonl          # valid dataset
│   │   ├── 📜 BioCon.jsonl         # train + valid + test
│   │   ├── 📜 retrieval_in-test.jsonl         # Intra-project retrieva
│   │   ├── 📜 software_test.jsonl         # project-level consistency
│   │   └── ...
│   ├── 📂 MRPC/
│   ├── 📂 SST-5/              
│   └── ...               
│
│── 📁 Pre_Dataset/                 # Preference datasets
│   ├── 📂 CoLA/                    # The CoLA dataset
│   │   ├── 📜 train.jsonl          # Training dataset
│   │   ├── 📜 test.jsonl           # Testing dataset
│   │   ├── 📜 valid.jsonl          # valid dataset
│   │   └── ...
│   ├── 📂 MRPC/
│   ├── 📂 SST-5/              
│   └── ...                    
│
│── 📁 Code/            # Implementations of classification models
│   ├── 🤖 bert/                    # BERT model implementation
│   │   ├── 📜 clss_indices.json    # Label mapping file
│   │   ├── 📜 model.py             # Model definition
│   │   ├── 📜 RewardModel.py       # Reward model definition
│   │   ├── 📜 run.py               # Script for fine-tuning the model
│   │   ├── 📜 run_RL.py            # RL optimization
│   │   ├── 📜 run_RM.py            # Script for training the RM
│   │   ├── 📜 test.py              # Script for model evaluation
│   │   └── ...
│   ├── 🤖 codebert/                # CodeBERT model implementation
│   ├── 🤖 t5/                      # T5 model implementation
│   ├── 🤖 codet5/                  # CodeT5 model implementation
│   ├── 🤖 codet5+/                 # CodeT5+ model implementation
│   ├── 🤖 opt/                     # OPT model implementation
│   ├── 🤖 codegen/                 # CodeGen model implementation
│   └── 🤖 qwen3/                   # QWen3 model implementation
│   
│── 📜 environment.yaml             # Environment configuration file
│── 📜 README.md                    
└── ...
```
