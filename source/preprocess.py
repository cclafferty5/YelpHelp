import sys
from segtok import tokenizer, segmenter
from collections import Counter
import numpy as np
import json
import os
import sentencepiece as spm
from capita import preprocess_capitalization

DATASET_FILE = "../datasets/yelp_review_training_dataset.jsonl"
PROCESSED_DATASET_FILE = "../datasets/dataset.jsonl"
VOCAB_FILE = "../datasets/wp_vocab10000.vocab"
VOCAB_MODEL_FILE = "../datasets/wp_vocab10000.model"

sp = spm.SentencePieceProcessor()
sp.Load(VOCAB_MODEL_FILE)
vocab = [line.split('\t')[0] for line in open(VOCAB_FILE, "r")]
pad_index = vocab.index('#')

def pad_sequence(numerized, pad_index, to_length):
    pad = numerized[:to_length]
    padded = pad + [pad_index] * (to_length - len(pad))
    mask = [w != pad_index for w in padded]
    return padded, mask

input_length = 300

with open(DATASET_FILE) as df:
    dataset = [json.loads(line) for line in df.readlines()]

for d in dataset:
    # Tokenize, numerize
    d['input'] = sp.EncodeAsIds(preprocess_capitalization(d['text']))
    
    # Padding    
    d['input'], d['input_mask'] =  pad_sequence(d['input'],  pad_index, input_length)

with open(PROCESSED_DATASET_FILE, "w+") as df:
    for d in dataset:
        print(json.dumps(d), file=df)
    




