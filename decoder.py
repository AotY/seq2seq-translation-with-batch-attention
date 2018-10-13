import torch
import torch.nn as nn


'''
Decoder:
    vocab_size:
    embedding_size:
    hidden_size:
    layer_nums:

'''


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, num_layers,
                 dropout_ratio, padding_idx, tied):

        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        # embedding
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_size, self.padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout_ratio)

        # LSTM
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,
                            self.num_layers, bias=True,
                            batch_first=False, dropout=dropout_ratio)

        # linear
        self.linear = nn.Linear(self.hidden_size,
                                self.vocab_size)

        if tied:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, encoder_outputs=None):
        '''
        input: [1, batch_size]  LongTensor

        hidden_state: [num_layers, batch_size, hidden_size]

        output: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
        hidden_state: (h_n, c_n)
        '''

        # embedded
        embedded = self.embedding(input) #[1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # lstm
        output, hidden_state = self.lstm(embedded, hidden_state)

        # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        output = output.squeeze(0)

        # linear
        output = self.linear(output)

        # softmax
        output = self.softmax(output)

        return output, hidden_state, None

