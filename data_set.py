import os
import random
import pickle

import torch

from nltk.tokenize import word_tokenize
from vocab import Vocab
from vocab import EOS_id


class DataSet:
    def __init__(self, filename, max_len, min_count, device=None):

        self.english_vocab = Vocab('english')
        self.french_vocab = Vocab('french')

        self._pairs = []
        self._indicator = 0
        self.min_count = min_count

        self.device = device

        # load data, build vocab
        self.load_data(filename, max_len)

    def load_data(self, filename, max_len):
        if os.path.exists('./data/_pairs.pkl') and \
                os.path.exists('./data/english_vocab.pkl') and \
                os.path.exists('./data/french_vocab.pkl'):

            self._pairs = pickle.load(open('./data/_pairs.pkl', 'rb'))
            self.english_vocab = pickle.load(
                open('./data/english_vocab.pkl', 'rb'))
            self.french_vocab = pickle.load(
                open('./data/french_vocab.pkl', 'rb'))
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    english, french = line.rstrip().split('\t')

                    english_tokens = [token.lower() for token in word_tokenize(english, language='english')]
                    french_tokens = [token.lower() for token in word_tokenize(french, language='french')]

                    # french contains a sos or eos
                    if len(english_tokens) > max_len + 10 or len(french_tokens) > max_len + 10:
                        continue

                    english_tokens = english_tokens[-min(max_len - 1, len(english_tokens)): ]
                    french_tokens = french_tokens[-min(max_len - 1, len(french_tokens)): ]

                    print('english_tokens: {}'.format(english_tokens))
                    print('french_tokens: {}'.format(french_tokens))

                    self._pairs.append(tuple([english_tokens, french_tokens]))

                    # add to vocab
                    self.english_vocab.add_words(english_tokens)
                    self.french_vocab.add_words(french_tokens)

            self.english_vocab.filter_words(self.min_count)
            self.french_vocab.filter_words(self.min_count)

            pickle.dump(self._pairs, open('./data/_pairs.pkl', 'wb'))
            pickle.dump(
                self.english_vocab, open(
                    './data/english_vocab.pkl', 'wb'))
            pickle.dump(
                self.french_vocab, open(
                    './data/french_vocab.pkl', 'wb'))

        # shuffle
        random.shuffle(self._pairs)

    def shuffle(self):
        self._indicator = 0
        random.shuffle(self._pairs)

    def get_size(self):
        return len(self._pairs)

    def next_batch(self, batch_size, max_len, reverse=False):
        next_indicator = self._indicator + batch_size
        if next_indicator > len(self._pairs):
            random.shuffle(self._pairs)
            self._indicator = 0
            next_indicator = batch_size

        next_pair = self._pairs[self._indicator: next_indicator]

        batch_english_ids = torch.zeros(
            (max_len, batch_size),
            dtype=torch.long,
            device=self.device)
        batch_french_ids = torch.zeros((max_len, batch_size),
                                       dtype=torch.long,
                                       device=self.device)

        batch_english_lengths = []
        batch_french_lengths = []

        for batch_id, (english, french) in enumerate(next_pair):
            english_ids = self.english_vocab.words_to_id(english)
            french_ids = self.french_vocab.words_to_id(french)
            #  print(english_ids)
            batch_english_lengths.append(len(english_ids))
            batch_french_lengths.append(len(french_ids))

            for index, eid in enumerate(english_ids):
                batch_english_ids[index, batch_id] = eid

            french_ids.append(EOS_id)
            for index, fid in enumerate(french_ids):
                batch_french_ids[index, batch_id] = fid

        self._indicator = next_indicator

        # english -> french or french -> english
        return batch_english_ids, batch_french_ids, batch_english_lengths, batch_french_lengths

    '''for evaluate'''
    def input_to_ids(self, input, max_len, language):

        tokens = [token.lower() for token in word_tokenize(input, language=language)]

        if language == 'english':
            input_ids = self.english_vocab.words_to_id(tokens)
        elif language == 'french':
            input_ids = self.french_vocab.words_to_id(tokens)

        lengths = []
        lengths.append(len(input_ids))

        batch_english_ids = torch.zeros((max_len, 1), dtype=torch.long, device=self.device)

        for index, id in enumerate(input_ids):
            batch_english_ids[index, 0] = id

        return batch_english_ids, lengths

    def outputs_to_sentence(self, outputs_index, language):
        if language == 'english':
            sentence = self.english_vocab.ids_to_sentence(outputs_index)
        elif language == 'french':
            sentence = self.french_vocab.ids_to_sentence(outputs_index)

        return sentence
