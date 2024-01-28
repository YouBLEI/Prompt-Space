#! /user/bin/env python
# coding=utf-8
import os
from tqdm import tqdm
import torch
import argparse
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaModel, LlamaTokenizer, LlamaConfig
import numpy as np


class llamaDataset(Dataset):
    def __init__(self, tokenizer):
        self.dir1 = 'compute_embedding/addsub.txt'
        self.tokenizer = tokenizer
        with open(self.dir1) as f:
            self.a = f.readlines()

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        tmp=self.a[idx]*2048
        item_tmp = self.tokenizer(
            tmp[:3000],
            return_tensors='pt'
            # return pytorch tensors                                                                                       return_tensors = 'pt',
        )
        #item['labels']=torch.tensor([[320]])
        item={}
        item['input_ids']=torch.reshape(item_tmp['input_ids'],(-1,))[:1024]
        item['labels']=item['input_ids'].clone() #torch.tensor([320])
        
        return item


def test():
    model_path = "/data_share/youbo_data_transfrom/pretrain_weights/llama7b" # input your checkpoint path
    
    # LlamaForCausalLM, LlamaTokenizer, LlamaConfig, LlamaModel
    # check this link: https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaModel
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    #config = DebertaV2Config.from_pretrained(args.model_path)
    # model1
    model1 = LlamaForCausalLM.from_pretrained(model_path)
    
    # model2
    #model2 = LlamaModel.from_pretrained(model_path)

    prompt = "Hey, are you consciours? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    #
    embed1= model1(inputs.input_ids, output_hidden_states=True)

    #embed2 = model2(inputs.input_ids, output_hidden_states=True)
    print(inputs.input_ids.shape)

    print(embed1.hidden_states[-1][:,0][0])
    print(embed1.hidden_states[-1].shape)


def embedding():
    device = torch.device('cuda:0')
    model_path = "/home/notebook/code/personal/S9052827/auto-cot/checkpoint/llama/7b" # input your checkpoint path
    
    # LlamaForCausalLM, LlamaTokenizer, LlamaConfig, LlamaModel
    # check this link: https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaModel
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    #config = DebertaV2Config.from_pretrained(args.model_path)
    # model1
    model1 = LlamaForCausalLM.from_pretrained(model_path)
    model1.to(device)
    corpus_embeddings = {}
    with open('/home/notebook/code/personal/S9052827/auto-cot/question/addsub', 'r') as f:
        corpus = f.readlines()

    
    for q in tqdm(corpus):
        llama_inputs = tokenizer(q, return_tensors="pt")
        corpus_embeddings[q] = torch.mean(model1(llama_inputs.input_ids.to(device), output_hidden_states=True).hidden_states[-1][0].detach(), dim=0)

    return corpus_embeddings
    

if __name__ == '__main__':
    embedding()
    

    #print(embed2.hidden_states[-1])

    # I have checked embed1 and embed2. Same outputs.

# corpus_embeddings = []   
# for i in range(len(corpus)):
#     q = corpus[i]
#     llama_inputs = tokenizer(q, return_tensors="pt")
#     corpus_embeddings[i] = encoder(llama_inputs.input_ids, output_hidden_states=True).hidden_states[-1][0][0][:]

