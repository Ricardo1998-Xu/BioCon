from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
import multiprocessing
from model import Model as Model

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'UniXcoder': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
}


def convert_examples_to_features(sentence, code, tokenizer, args):
    text1 = ' '.join(sentence.split())
    text2 = ' '.join(code.split())
    text1_tokens = tokenizer.tokenize(text1)
    text2_tokens = tokenizer.tokenize(text2)
    code_len = args.block_size - 3 - len(text1_tokens)

    text2_tokens = text2_tokens[:code_len]
    # --- [CLS] sentence [SEP] code [SEP] ---
    source_tokens = [tokenizer.cls_token] + text1_tokens + [tokenizer.sep_token] + text2_tokens + [tokenizer.sep_token]

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_ids


class RetrievalDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.data = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                js = json.loads(line.strip())
                self.data.append(js)

        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def score_sentence(model, tokenizer, args, query, candidates, device):
    scores = []
    batch_size = args.eval_batch_size
    model.eval()
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch_codes = candidates[i:i + batch_size]

            input_ids = []
            for code in batch_codes:
                ids = convert_examples_to_features(query, code, tokenizer, args)
                input_ids.append(ids)

            input_ids = torch.tensor(input_ids).to(device)

            prob, _ = model(input_ids)

            scores.extend(prob[:, 1].cpu().tolist())  # consistency probability

    return scores


def test(args, model, tokenizer):
    # results path
    results_path = args.results_path
    if os.path.exists(results_path) is False:
        os.makedirs(results_path)

    dataset = RetrievalDataset(tokenizer, args, args.test_data_file)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    with open(os.path.join(results_path, 'case_results.jsonl'), 'w', encoding="utf-8") as out_f:
        for item in tqdm(dataset):
            sentence = item["sentence"]
            candidates = item["candidates"]
            repo = item["repo"]

            # ===== scoring =====
            scores = score_sentence(model, tokenizer, args, sentence, candidates, args.device)

            # ===== Top-10 =====
            pairs = list(zip(candidates, scores))
            pairs.sort(key=lambda x: x[1], reverse=True)
            top10 = pairs[:10]

            max_score = top10[0][1] if top10 else 0.0

            flag = "potentially_inconsistent" if max_score < args.tau else "likely_consistent"

            result = {
                "sentence": sentence,
                "repo": repo,
                "top10": [
                    {"code": c[:300], "score": float(s)}
                    for c, s in top10
                ],
                "max_score": float(max_score),
                "flag": flag
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument('--num_labels', type=int, default=2,
                        help='class num')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='threshold')
    parser.add_argument("--test_data_file", default='./data/case/case_dataset.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--output_dir", default='./weights/UniXcoder/FL-weight=5-lr0.00002-seed{seed}', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--results_path', type=str,
                        default='./results/case/UniXcoder/FL-weight=5-lr0.00002-seed{seed}', help='save results path')

    parser.add_argument('--seed', type=int, nargs='+', default=[1, 123, 123456],
                        help="random seed for initialization")

    parser.add_argument("--model_type", default="UniXcoder", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default='./pre', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="./pre", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")

    parser.add_argument("--do_test", default=True, action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--evaluate_during_training", default=True, action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)


    seeds = args.seed
    original_output_dir = args.output_dir
    original_results_path = args.results_path
    for seed in seeds:
        # Set seed
        set_seed(seed)
        args.output_dir = original_output_dir.format(seed=seed)
        args.results_path = original_results_path.format(seed=seed)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

        args.start_epoch = 0
        args.start_step = 0
        checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
        if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
            args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
            args.config_name = os.path.join(checkpoint_last, 'config.json')
            idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
            with open(idx_file, encoding='utf-8') as idxf:
                args.start_epoch = int(idxf.readlines()[0].strip()) + 1

            step_file = os.path.join(checkpoint_last, 'step_file.txt')
            if os.path.exists(step_file):
                with open(step_file, encoding='utf-8') as stepf:
                    args.start_step = int(stepf.readlines()[0].strip())

            logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
        config.num_labels = 1
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)

        if args.model_type == "qwen3":
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
        if args.model_name_or_path:
            model = model_class.from_pretrained(args.model_name_or_path,
                                                from_tf=bool('.ckpt' in args.model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
        else:
            model = model_class(config)

        model = Model(model, config, tokenizer, args)
        if args.local_rank == 0:
            torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

        logger.info("Test parameters %s", args)

        # test
        if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model.load_state_dict(torch.load(output_dir, map_location=torch.device(device)))
            model.to(args.device)
            test(args, model, tokenizer)


if __name__ == "__main__":
    main()
