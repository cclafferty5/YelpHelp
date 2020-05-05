import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os

cwd = os.getcwd()
tokenizers_dir = os.path.join(cwd, "models", "tokenizers")

class YelpPreprocessor:
    def preprocess(self, texts):
        raise NotImplementedError # abstract class

class SimpleTokenizerPadder(YelpPreprocessor):
    def __init__(self, tokenizer, input_length=300):
        self.tokenizer = tokenizer
        self.input_length = input_length
    def preprocess(self, texts):
        return pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.input_length)

class CharacterModelPreprocessor(YelpPreprocessor):
    def __init__(self, word_tokenizer, char_tokenizer, input_length=300, word_length=5):
        self.word_tokenizer = word_tokenizer
        self.input_length = input_length
        self.char_tokenizer = char_tokenizer
        self.word_length = word_length
    def character_preprocess(self, texts):
        char_sequences = self.char_tokenizer.texts_to_sequences(texts)
        out = np.zeros((len(char_sequences), self.input_length, self.word_length))
        space_character = self.char_tokenizer.word_index[' ']
        for i, seq in enumerate(char_sequences):
            word_index = 0
            char_index = 0
            for c in char_sequences[i]:
                if c == space_character:
                    if char_index != 0:
                        word_index += 1
                    char_index = 0
                else:
                    if char_index < self.word_length:
                        out[i, word_index, char_index] = c
                    char_index += 1
                if word_index >= self.input_length:
                    break
            if word_index < self.input_length:
                adj = 1 if char_index != 0 else 0 # if char_index is 0, we added one at the end, and word_index = num_words, else we are at example index 1 in a len 5, and we want to roll 3, not 4
                out[i] = np.roll(out[i], self.input_length - word_index - adj, axis=0)
        return out
    
    def word_preprocess(self, texts):
        return pad_sequences(self.word_tokenizer.texts_to_sequences(texts), maxlen=self.input_length)

    def preprocess(self, texts):
        return [self.word_preprocess(texts), self.character_preprocess(texts)]    

def load_tokenizer(name):
    file_path = os.path.join(tokenizers_dir, name)
    with open(file_path) as tkf:
        return tokenizer_from_json(tkf.read())

tokenizer_100000 = load_tokenizer("test_tokenizer_100000")
tokenizer_100000_with_unks = load_tokenizer("test_tokenizer_100000_with_unks")
char_tk = load_tokenizer("test_char_tokenizer")

BEST_PREPROCESSOR = CharacterModelPreprocessor(tokenizer_100000_with_unks, char_tk)
OLD_PREPROCESSOR = SimpleTokenizerPadder(tokenizer_100000)

