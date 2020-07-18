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
import string
import os


class NegativeSampler(object):
    def __init__(self, f: np.array):
        pmf = np.power(f, 3.0/4)
        pmf = np.squeeze(pmf)
        self.pmf = tf.convert_to_tensor(pmf)
        self.pmf = tf.math.log(self.pmf) - \
            tf.math.log(tf.math.reduce_sum(self.pmf))
        print(self.pmf.shape)

    def sample(self, n: int = 5):
        s = tf.random.categorical(
            logits=[self.pmf], num_samples=n, dtype=tf.int64)
        print(s.shape)
        s = tf.reshape(s, [n, 1])
        return s


def frequency(vector: List[int], size: int) -> np.array:
    count = np.zeros([1, size])
    for i in vector:
        count[0, i] += 1
    return count


class Sampler(object):
    def __init__(self, text_vector: str, target_space: List[List[int]], vocab_size: int):
        super().__init__()
        self.target_space = tf.constant(target_space, dtype=tf.int64)
        self.text_vector = tf.constant(text_vector, dtype=tf.int64)
        self.negative_sampler = NegativeSampler(
            f=frequency(vector=text_vector, size=vocab_size))

    def sample_positive(self, context_index):
        min_value = self.target_space[context_index, 0]
        max_value = self.target_space[context_index, 1]
        offset = tf.random.uniform(minval=min_value,
                                   maxval=max_value, dtype=tf.int64, shape=[])
        # for removing the context_index from offset
        # print(offset)
        if offset >= 0:
            offset += 1
        target_index = context_index+offset
        print(target_index.shape)
        return self.text_vector[context_index], self.text_vector[target_index]

    def sample(self, context_index: tf.Tensor, k: int):
        context, target = self.sample_positive(context_index=context_index)
        neg_s = self.negative_sampler.sample(k)
        y = tf.concat([[[1]], tf.tile([[0]], [k, 1])], axis=0)
        x = tf.concat(
            [
                tf.tile(input=[[context]], multiples=[k+1, 1]),
                tf.concat(
                    [
                        [[target]],
                        neg_s
                    ],
                    axis=0
                )
            ],
            axis=1
        )

        return x, y


def Context_target_candidates(text: str, n: int):
    target_bounds = []
    cleaned_words = []
    sentences = nltk.sent_tokenize(text=text)
    for sentence in sentences:
        # print(sentence)
        words = [word for word in nltk.word_tokenize(
            sentence) if word not in string.punctuation]
        s = len(words)
        if s > 1:
            bounds = []
            for i in range(s):
                bounds.append([max(-n, -i), min(n, (s-1)-i)])
            cleaned_words += words
            target_bounds += bounds
    return cleaned_words, target_bounds


class EmbeddingModel(Model):
    def __init__(self, vocab_size: int, embedding_size: int = 300):
        super(EmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        self.word_embedding = tf.keras.layers.Embedding(
            vocab_size,
            self.embedding_size
        )
        self.realation_embedding = tf.keras.layers.Embedding(
            vocab_size,
            self.embedding_size
        )

    def call(self, x, training=False):
        c = self.word_embedding(x[:, :, 0], training=training)
        t = self.realation_embedding(x[:, :, 1], training=training)
        c = tf.nn.l2_normalize(c, -1)
        t = tf.nn.l2_normalize(t, -1)
        # pred = tf.reduce_sum(c*t, -1, keepdims=True)
        pred = tf.math.sigmoid(tf.reduce_sum(c*t, -1, keepdims=True))
        return pred
        # return c


class WordEmbedder(object):
    def __init__(self, vocab: utils.Vocablury, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.text_vectorizer = utils.Text_Vectorizer(vocab=self.vocab)

        self.embedder = EmbeddingModel(
            embedding_size=embedding_size, vocab_size=len(self.vocab))

    def CreateDataset(self, text: str, k: int = 8, neigbhourhood: int = 5, batch_size: int = 16, validation_slice: float = 0.2) -> None:
        self.text = text
        cleaned_text, target_space = Context_target_candidates(
            text=text, n=5)
        text_vector = [self.vocab.index(term) for term in cleaned_text]
        sampler = Sampler(
            text_vector=text_vector, target_space=target_space, vocab_size=len(self.vocab))
        dataset = tf.data.Dataset.range(
            neigbhourhood, len(text_vector)-neigbhourhood)
        dataset = dataset.map(
            map_func=lambda i: sampler.sample(
                context_index=i,
                k=k
            )
        )
        dataset = dataset.shuffle(buffer_size=10000)
        size = len(text_vector)-neigbhourhood-neigbhourhood
        training_size = int(size*(1-validation_slice))
        self.training = dataset.take(training_size).batch(
            batch_size, drop_remainder=True)
        self.validation = dataset.skip(training_size).batch(
            batch_size, drop_remainder=True)

    def train(self, epochs: int = 10, validation_split: float = 0.0):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        loss = tf.keras.losses.BinaryCrossentropy()
        # loss = tf.keras.losses.MeanSquaredError()
        for (x, y) in self.training.take(1):
            self.embedder(x)
            # print(loss(y, self.embedder(x)))

        print(self.embedder.summary())
        self.embedder.compile(
            optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

        checkpoint_dir = "./negative_sampling_model/training_checkpoints"
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = self.embedder.fit(self.training, epochs=epochs, callbacks=[
            checkpoint_callback, tensorboard_callback], validation_data=self.validation)
        return history
        # TODO Create a training pipeline

    def load(self, path: str = "./negative_sampling_model/training_checkpoints"):
        self.embedder.load_weights(tf.train.latest_checkpoint(path))

    def get_embeddings(self):
        weights = self.embedder.word_embedding.get_weights()[0]
        embedding = {}
        for index, word in enumerate(self.vocab.index_to_term):
            embedding[word] = weights[index]
        return embedding

    def save(self):
        embeddings = self.get_embeddings()
        with open("meta.tsv", mode="w") as meta_file, open("vecs.tsv", mode="w") as vec_file:
            for word, embedding in embeddings.items():
                meta_file.write(word+"\n")
                vec_file.write('\t'.join([str(x) for x in embedding]) + "\n")
