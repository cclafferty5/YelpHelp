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

NUM_CLASSES = 5

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
    model.add(Dense(NUM_CLASSES, activation='softmax'))
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

def load_keras_model(name):
    return load_model(os.path.join(models_dir, name))

class YelpModel:
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def predict_ratings(self, preprocessed_inputs):
        return [np.argmax(p) + 1 for p in self.keras_model.predict(preprocessed_inputs)]

class EnsembleModel(YelpModel):
    def __init__(self, config):
        self.num_models = len(config)
        self.models = [model for model, _ in config]
        self.weights = [weight for _, weight in config]
    
    def predict_ratings(self, preprocessed_inputs):
        assert len(preprocessed_inputs) == self.num_models
        num_samples = len(preprocessed_inputs[0])
        predictions = np.zeros((num_samples, NUM_CLASSES))
        for i, inputs in enumerate(preprocessed_inputs):
            predictions += self.weights[i] * self.models[i].predict(inputs)
        return [np.argmax(p) + 1 for p in predictions]
        

####### MODELS ########
glove_gru_bi = load_keras_model("glove_gru_bi")
glove_gru_bi_char = load_keras_model("glove_gru_bi_char")

GLOVE_GRU_BI = YelpModel(glove_gru_bi)
GLOVE_GRU_BI_CHAR = YelpModel(glove_gru_bi_char)

ensemble_config = [(glove_gru_bi, .7), (glove_gru_bi_char, .3)]
CHAR_NO_CHAR_ENSEMBLE = EnsembleModel(ensemble_config)

#######################

BEST_MODEL = CHAR_NO_CHAR_ENSEMBLE

