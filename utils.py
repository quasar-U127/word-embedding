import nltk


def get_text(file_name: str) -> str:
    text_file_name = file_name
    with open(text_file_name, "r") as text_file:
        return text_file.read().lower()


class Vocablury(object):
    unknown_symbol = "<UNK>"
    # SOS: Start Of Sentence
    sos = "start_of_sentence"
    # EOS: End Of Sentence
    eos = "end_of_sentence"

    def __init__(self, text: str):
        self.index_to_term = sorted(set(nltk.word_tokenize(text=text)))
        self.index_to_term.append(Vocablury.unknown_symbol)
        self.index_to_term.append(Vocablury.sos)
        self.index_to_term.append(Vocablury.eos)
        self.term_to_index = {}
        for i, term in enumerate(self.index_to_term):
            self.term_to_index[term] = i

    def __len__(self):
        return len(self.index_to_term)

    def index(self, term: str):
        if(term in self.term_to_index):
            return self.term_to_index[term]
        return self.term_to_index[Vocablury.unknown_symbol]

    def term(self, index: int):
        return self.index_to_term[index]


class Text_Vectorizer(object):
    def __init__(self, vocab: Vocablury):
        self.vocab = vocab

    def vector(self, text: str):
        return [self.vocab.index(term) for term in nltk.word_tokenize(text=text)]

    def text(self, vector):
        return " ".join([self.vocab.term(index) for index in vector])
