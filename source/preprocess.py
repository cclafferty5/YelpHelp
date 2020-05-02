import tensorflow as tf
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

def load_tokenizer(name):
    file_path = os.path.join(tokenizers_dir, name)
    with open(file_path) as tkf:
        return tokenizer_from_json(tkf.read())

tokenizer_100000 = load_tokenizer("test_tokenizer_100000")

BEST_PREPROCESSOR = SimpleTokenizerPadder(tokenizer_100000)

