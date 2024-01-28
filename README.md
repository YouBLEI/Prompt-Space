# Prompt Space Optimizing Few-shot Reasoning Success with Large Language Models

This repository contains the official implementation of our method proposed in [Prompt Space Optimizing Few-shot Reasoning Success with Large Language Models](https://arxiv.org/abs/2306.03799)

## Introduction

![Overview of our methods](./overview.jpg)

**Abstract**:Prompt engineering is an essential technique for enhancing the abilities of large language models (LLMs) by providing explicit and specific instructions. It enables LLMs to excel in various tasks, such as arithmetic reasoning, question answering, summarization, relation extraction, machine translation, and sentiment analysis. Researchers have been actively exploring different prompt engineering strategies, such as Chain of Thought (CoT), Zero-CoT, and In-context learning. However, an unresolved problem arises from the fact that current approaches lack a solid theoretical foundation for determining optimal prompts. To address this issue in prompt engineering, we propose a new and effective approach called Prompt Space. Our methodology utilizes text embeddings to obtain basis vectors by matrix decomposition, and then constructs a space for representing all prompts. Prompt Space significantly outperforms state-of-the-art prompt paradigms on ten public reasoning benchmarks. Notably, without the help of the CoT method and the prompt "Let's think step by step", Prompt Space shows superior performance over the few-shot method. Overall, our approach provides a robust and fundamental theoretical framework for selecting simple and effective prompts. This advancement marks a significant step towards improving prompt engineering for a wide variety of applications in LLMs.

## Requirements

Python>=3.8
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
pip install sentence_transformers
pip install matplotlib
```

## Datasets

Download the datasets from the following:

```
Url_1: https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset
Url_2: https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/log
```

## Instructions

### Step 1. Construct basis:

```shell
python run_demo.py --task gsm8k --pred_file log/gsm8k_zero_shot_cot.log --demo_save_dir demos/gsm8k/base_8 --encoder all-MiniLM-L6-v2 --question_save_dir question/gms8k --num_basis 8
# Important parameter description.
# --task : Task name
# --pred_file: The path of the GSM8k dataset, which is the data downloaded from Url_2
# --demo_save_dir : save path
# --encoder: The model we use to extract text embedding.
# --num_basis: How many basis questions do we need to select?
```
The selected basis questions of GSM8k are presented as follows.

![The result of the Basis question](gsm8k_bias.jpg)

### Step 2. Run inference:

Use the prompt we previously selected to perform the inference：

```shell
python run_inference.py --dataset gsm8k --demo_path demos/gsm8k/base_8 --output_dir experiment/gsm8k/base_8_prompt_space --method prompt_space
python run_inference.py --dataset gsm8k --demo_path demos/gsm8k/base_8 --output_dir experiment/gsm8k/base_8_prompt_space_cot --method prompt_space_cot
python run_inference.py --dataset gsm8k --demo_path demos/gsm8k/base_8 --output_dir experiment/gsm8k/base_8_prompt_space_cot_zero --method prompt_space_cot_zero
```

## BibTeX

If this repository has been helpful to you, please cite as follows：

```
@article{shi2023prompt,
  title={Prompt Space Optimizing Few-shot Reasoning Success with Large Language Models},
  author={Shi, Fobo and Qing, Peijun and Yang, Dong and Wang, Nan and Lei, Youbo and Lu, Haonan and Lin, Xiaodong},
  journal={arXiv preprint arXiv:2306.03799},
  year={2023}
}
```



## Acknowledgements

Our method is a general technique for enhancing the abilities of large language models (LLMs) by providing explicit and specific instructions, which is builded upon several solid works. Thanks to [Auto-CoT](https://github.com/amazon-science/auto-cot) and [zero_shot_cot](https://github.com/kojima-takeshi188/zero_shot_cot) for their wonderful work and codebase!

