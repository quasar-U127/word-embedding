import negative_sampling
import utils
import tensorflow as tf


def main():
    text = negative_sampling.get_text("source.txt")
    text = text.replace("—", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    vocab = utils.Vocablury(text=text)

    embedder = negative_sampling.WordEmbedder(vocab=vocab, embedding_size=300)
    embedder.CreateDataset(text=text, batch_size=256, k=2, neigbhourhood=1)
    embedder.train(epochs=1)
    embedder.save()


if __name__ == "__main__":
    main()
