import tensorflow as tf
import sys
from collections import Counter
import numpy as np
import json
import os

from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

cwd = os.getcwd()
models_dir = os.path.join(cwd, "models")
checkpoint_dir = os.path.join(models_dir, "checkpoints")

def build_model(input_length=300, rnn_size=256, use_glove=True, vocab_size=100000,
                 learning_rate=1e-3, dropout_rate=.2, use_gru=False, use_bidirectional=False, use_c2v=False):
    model = Sequential()
    if not use_c2v:
        if use_glove:
            embed = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        else:
            embed = Embedding(vocab_size, rnn_size, input_length=input_length)
        model.add(embed)
    if use_gru:
        rnn_cell = GRU(rnn_size, dropout=dropout_rate)
    else:
        rnn_cell = LSTM(rnn_size, dropout=dropout_rate)
    if use_bidirectional:
        model.add(Bidirectional(rnn_cell))
    else:
        model.add(rnn_cell)
    model.add(Dense(num_classes, activation='softmax'))
    loss = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def train_model(model, train_seqs, train_labels, num_epochs, save_as, batch_size=64, validation_split=.2):
    save_file = os.path.join(models_dir, save_as)
    checkpoint_file = os.path.join(checkpoint_dir, f"{save_as}.ckpt")
    cp_callback = ModelCheckpoint(filepath=checkpoint_file, verbose=1)
    training_result = model.fit(train_seqs, train_labels, epochs=num_epochs, batch_size=batch_size, validation_split=.2, callbacks=[cp_callback])
    model.save(save_file)

def load_model_from_source(name):
    return load_model(os.path.join(models_dir, name))

####### MODELS ########
print(models_dir)
glove_gru_bi = load_model_from_source("glove_gru_bi")
BEST_MODEL = glove_gru_bi

