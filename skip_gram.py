import sys
import utils
import nltk
import tensorflow as tf
import numpy as np
import datetime
from tensorflow import keras
from typing import List, Tuple
from random import randrange
from sklearn.preprocessing import normalize
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import string
import os


def create_phrase(text: str, n: int, vocab: utils.Vocablury):
    samples = []
    sentences = nltk.sent_tokenize(text=text)
    for sentence in sentences:
        # print(sentence)
        words = [vocab.index(word) for word in nltk.word_tokenize(
            sentence) if word not in string.punctuation]
        if len(words) >= 1:
            s = len(words)
            for i in range(s):
                for j in range(i-n, i+n+1):
                    if j < 0:
                        sample = [words[i], vocab.index(vocab.sos)]
                    elif j >= s:
                        sample = [words[i], vocab.index(vocab.eos)]
                    else:
                        sample = [words[i], words[j]]
                    samples.append(sample)
    return samples


def create_sample(phrase: tf.Tensor):
    x = phrase[0]
    y = phrase[1]
    return x, y


class EmbeddingModel(Model):
    def __init__(self, vocab_size: int, embedding_size: int = 300):
        super(EmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        self.word_embedding = tf.keras.layers.Embedding(
            vocab_size,
            self.embedding_size
        )
        self.decoder = tf.keras.layers.Dense(units=vocab_size)

    def call(self, x, training=False):
        context = self.word_embedding(x, training=training)
        logits = self.decoder(context, training=training)
        return logits


class WordEmbedder(object):
    def __init__(self, vocab: utils.Vocablury, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.text_vectorizer = utils.Text_Vectorizer(vocab=self.vocab)
        self.model = EmbeddingModel(
            vocab_size=len(self.vocab),
            embedding_size=self.embedding_size
        )

    def CreateDataset(self, text: str, neigbhourhood: int = 5, batch_size: int = 16, validation_slice: float = 0.2) -> None:
        self.text = text
        phrases = create_phrase(text=text, n=neigbhourhood, vocab=self.vocab)
        dataset = tf.data.Dataset.from_tensor_slices(phrases)
        # for x in dataset.take(20):
        #     print(tf.gather(self.vocab.index_to_term, x))
        dataset = dataset.map(
            map_func=lambda phrase: create_sample(phrase)
        )
        dataset = dataset.shuffle(buffer_size=10000)
        size = len(phrases)
        training_size = int(size*(1-validation_slice))
        self.training = dataset.take(training_size)
        self.training = self.training.batch(batch_size, drop_remainder=True)
        # for x, y in self.training.take(5):
        #     print(y)
        self.validation = dataset.skip(training_size)
        self.validation = self.validation.batch(
            batch_size, drop_remainder=True)

    def train(self, epochs: int = 10, validation_split: float = 0.0):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss = tf.keras.losses.MeanSquaredError()
        for (x, y) in self.training.take(1):
            self.model(x)
        # print(loss(y, self.model(x)))

        print(self.model.summary())
        self.model.compile(
            optimizer="adam", loss=loss)

        checkpoint_dir = "./CBOW/training_checkpoints"
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = self.model.fit(self.training, epochs=epochs, callbacks=[
            checkpoint_callback, tensorboard_callback], validation_data=self.validation)
        return history
        # TODO Create a training pipeline

    def load(self, path: str = "./negative_sampling_model/training_checkpoints"):
        self.model.load_weights(tf.train.latest_checkpoint(path))

    def get_embeddings(self):
        weights = self.model.word_embedding.get_weights()[0]
        embedding = {}
        for index, word in enumerate(self.vocab.index_to_term):
            embedding[word] = weights[index]
        return embedding

    def save(self):
        embeddings = self.get_embeddings()
        os.mkdir("results/skip_gram")
        with open("results/skip_gram/meta.tsv", mode="w") as meta_file, open("results/skip_gram/vecs.tsv", mode="w") as vec_file:
            for word, embedding in embeddings.items():
                meta_file.write(word+"\n")
                vec_file.write('\t'.join([str(x) for x in embedding]) + "\n")
