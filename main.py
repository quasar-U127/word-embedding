# import negative_sampling
# import CBOW
import skip_gram
import utils


def main():
    # vocab = utils.Vocablury(text=utils.get_text("data/wiki_vocab.txt"))
    vocab = utils.Vocablury(text=utils.get_text("data/vocablury.txt"))

    # print("\n".join(vocab.index_to_term))

    # text = utils.get_text("data/wiki_text_sample.txt")
    text = utils.get_text("data/source.txt")

    text = text.replace("—", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace(".=", " ")
    text = text.replace("=", " ")
    text = text.replace("|", " ")
    text = text.replace("'", " ' ")
    text = text.replace("0", " 0 ")
    text = text.replace("1", " 1 ")
    text = text.replace("2", " 2 ")
    text = text.replace("3", " 3 ")
    text = text.replace("4", " 4 ")
    text = text.replace("5", " 5 ")
    text = text.replace("6", " 6 ")
    text = text.replace("7", " 7 ")
    text = text.replace("8", " 8 ")
    text = text.replace("9", " 9 ")

    # text = text[:100000]
    # print(text)

    embedder = skip_gram.WordEmbedder(vocab=vocab, embedding_size=50)
    embedder.CreateDataset(text=text, batch_size=64, neigbhourhood=2)

    # # # for x, y in embedder.training.take(1):
    # # #     print(x[0])
    # # #     print(tf.gather(vocab.index_to_term, x[0]))

    embedder.train(epochs=10)
    embedder.save()


if __name__ == "__main__":
    main()
