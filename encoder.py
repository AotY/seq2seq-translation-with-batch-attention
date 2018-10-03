import math
import numpy as np
import torch
import torch.nn as nn


'''
Encoder:
    vocab_size:
    embedding_size:
    hidden_size:
    layer_nums:
    bidirectional:

'''


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, num_layers,
                 bidirectional, dropout_ratio, padding_idx):

        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx

        if bidirectional:
            #  self.hidden_size //= 2
            self.num_directions = 2

        # embedding
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_size, self.padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout_ratio)

        # LSTM
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,
                            self.num_layers, bias=True, batch_first=False,
                            dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, inputs, inputs_length, hidden_state):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor

            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]

        :return
            outputs: [seq_len, batch, num_directions * hidden_size]
            hidden_state: (h_n, c_n)
        '''

        # embedded
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        # [batch_size, seq_len, embedding_size]
        embedded = embedded.transpose(0, 1)

        # sort lengths
        inputs_length = np.asarray(inputs_length, dtype=np.long)
        sorted_indexs = np.argsort(-inputs_length)

        new_inputs_length = inputs_length[sorted_indexs]
        sorted_indexs = torch.tensor(sorted_indexs, dtype=torch.long)
        # restore to origianl indexs
        restore_indexs = torch.tensor(np.argsort(sorted_indexs), dtype=torch.long)

        # new embedded
        embedded = embedded[sorted_indexs].transpose(
            0, 1)  # [seq_len, batch_size, embedding_size]

        # pack
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, new_inputs_length)

        # lstm, outputs -> [seq, batch_size, hidden_size * 2], because of using
        # bidirection, hidden_state -> [num_directions * num_layers,
        # batch_size, hidden_size]
        outputs, hidden_state = self.lstm(packed_embedded, hidden_state)

        # unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # [seq, batch_size, hidden_size * 2] -> [seq, batch_size, hidden_size]
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]

        # to original sequence
        outputs = outputs.transpose(
            0, 1)[restore_indexs].transpose(0, 1).contiguous()
        hidden_state = tuple([item.transpose(0, 1)[restore_indexs].transpose(
            0, 1).contiguous() for item in hidden_state])

        return outputs, hidden_state

    def get_output_hidden_size(self):
        return self.hidden_size * self.num_directions

    def init_hidden(self, batch_size):
        initial_state_scale = math.sqrt(3.0 / self.hidden_size)

        initial_state1 = torch.rand(
            (self.num_directions * self.num_layers, batch_size, self.hidden_size))
        initial_state2 = torch.rand(
            (self.num_directions * self.num_layers, batch_size, self.hidden_size))

        initial_state1 = (-initial_state_scale - initial_state_scale) * \
            initial_state1 + initial_state_scale
        initial_state2 = (-initial_state_scale - initial_state_scale) * \
            initial_state2 + initial_state_scale

        return (initial_state1, initial_state2)
