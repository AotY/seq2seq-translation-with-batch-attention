PAD_id = 0
SOS_id = 1
EOS_id = 2
UNK_id = 3

PAD = 'PAD'
SOS = 'SOS'
EOS = 'EOS'
UNK = 'UNK'

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2idx = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
        self.word2count = {}
        self.idx2word = {}
        self.n_words = 4 # because of the unk, pad, sos, and eos tokens.

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    ''' filtering words by min count'''
    def filter_words(self, min_count=3):
        sorted_list = sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)
        sorted_list = [item for item in sorted_list if item[1] > min_count]
        for word, _ in sorted_list:
            self.word2idx[word] = self.n_words
            self.n_words += 1

        # init idx2word
        self.idx2word = {v: k for k, v in self.word2idx.items()}


    def id_to_word(self, id):
        return self.idx2word.get(id, UNK)

    def ids_to_word(self, ids):
        words = [self.id_to_word(id) for id in ids]
        return words

    def word_to_id(self, word):
        return self.word2idx.get(word, UNK_id)

    def words_to_id(self, words):
        word_ids = [self.word_to_id(word) for word in words]
        return word_ids

    def get_vocab_size(self):
        return len(self.word2idx)

    def ids_to_sentence(self, ids):
        words = [self.id_to_word(id) for id in ids if id not in [PAD_id, SOS_id, EOS_id, UNK_id]]
        return ' '.join(words)

