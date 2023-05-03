# Knowledge Base Construction from Pre-trained Language Models (LM-KBC) 2nd Edition

This repository contains dataset for the LM-KBC challenge at ISWC 2023.

## Dataset v0.9

Preliminary release of the LM-KBC dataset, evaluation script, GPT-baseline



### Baselines

As baselines, we provide a script that can run masked LMs and causal LMs from Huggingface in the baseline.py
Furthermore, we also provide a GPT-3 baseline.

Running instructions for the HF baselines:
For BERT
* python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-bert.jsonl --train_data data/train.jsonl --model bert-large-cased --batch_size 32 --gpu 0

For OPT-1.3b
* python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-opt.jsonl --train_data data/train.jsonl --model facebook/opt-1.3b --batch_size 8 --gpu 0


Run instructions GPT-3 baseline:
 * Insert your OpenAI API key in line 33
 * python baseline-GPT3-IDs-directly.py
 
(To save money, by default the script only runs on a file containing 10 subjects (train_tiny.jsonl))
 
Run instructions evaluation script:
  * python evaluate.py -p train_tiny_predictions.jsonl -g train_tiny.jsonl


### Note

The released dataset is primarily for understanding the format. We are making a few quality checks and the changes to the final version will be minor. The final dataset will be added here within a few days.

### Coming soon

- More baseline scripts (BERT baseline, GPT-3 baseline w/ external entity disambiguation)
- Dataset V1 with further cleaning
