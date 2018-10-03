import torch
import torch.nn as nn
import torch.nn.functional as F


'''
GlobalAttn
    dot:
    general:
    concat:

'''


class GlobalAttn(nn.Module):
    def __init__(self, attn_method, hidden_size):

        super(GlobalAttn, self).__init__()

        self.attn_method = attn_method
        self.hidden_size = hidden_size

        if self.attn_method == 'general':
            self.attn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.attn_method == 'concat':
            self.attn_linear = nn.Linear(
                self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden_state, encoder_outputs):
        #  print(hidden_state.shape)
        #  print(encoder_outputs.shape)
        max_len, batch_size = encoder_outputs.shape[:2]
        #  encoder_outputs = encoder_outputs.transpose(0, 1) # [batch_size, seq_len, hidden_size]

        attn_weights = torch.zeros((batch_size, max_len))

        # For each batch of encoder outputs
        for batch_index in range(batch_size):
            #  weight for each encoder_output
            for len_index in range(max_len):
                one_encoder_output = encoder_outputs[len_index, batch_index].unsqueeze(0)
                one_hidden_state = hidden_state[batch_index, :].unsqueeze(0)

                attn_weights[batch_index, len_index] = self.score(
                    one_hidden_state,
                    one_encoder_output
                )

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        return attn_weights

    def score(self, one_hidden_state, one_encoder_output):
        #  print(one_encoder_output.shape)
        #  print(one_hidden_state.shape)

        if self.attn_method == 'general':
            #  weight = one_hidden_state.dot(self.attn_linear(one_encoder_output))
            weight = torch.dot(one_hidden_state.view(-1),
                               self.attn_linear(one_encoder_output).view(-1))

        elif self.attn_method == 'dot':
            weight = torch.dot(one_hidden_state.view(-1),
                               one_encoder_output.view(-1))
            #  weight = one_hidden_state.dot(one_encoder_output)

        elif self.attn_method == 'concat':
            #  weight = self.v.dot(self.attn_linear(
                #  torch.cat((one_hidden_state, one_encoder_output), dim=1)))
            weight = torch.dot(self.v.view(-1),
                self.attn_linear(
                    torch.cat((one_hidden_state, one_encoder_output), dim=1)).view(-1)
                    )

        return weight


'''
BahdanauAttnDecoder:
    vocab_size:
    embedding_size:
    hidden_size:
    layer_nums:

'''


class BahdanauAttnDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, num_layers,
                 dropout_ratio, padding_idx, tied):

        super(BahdanauAttnDecoder, self).__init__()

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

        # attn
        self.attn = GlobalAttn('concat', self.hidden_size)

        # hidden_size -> embedding_size, for attn
        if self.hidden_size != self.embedding_size:
            self.hidden_embedded_linear = nn.Linear(
                self.hidden_size, self.embedding_size)

        # LSTM, * 2 because using concat
        self.lstm = nn.LSTM(self.embedding_size * 2, self.hidden_size,
                            self.num_layers, bias=True,
                            batch_first=False, dropout=dropout_ratio)

        # linear
        self.linear = nn.Linear(self.hidden_size,
                                self.vocab_size)

        if tied:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, encoder_outputs):
        '''
        input: [1, batch_size]  LongTensor

        hidden_state: [num_layers, batch_size, hidden_size]

        output: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
        hidden_state: (h_n, c_n)
        '''

        # embedded
        input = input.view(1, -1)
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)
        #  print(embedded.shape)

        # attn_weights
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(hidden_state[0][-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)

        # Combine embedded input word and attened context, run through RNN
        if self.hidden_size != self.embedding_size:
            context = self.hidden_embedded_linear(context)

        print('context shape: ', context.shape)
        print('embedded shape: ', embedded.shape)
        input_combine = torch.cat((context, embedded), dim=2)

        # lstm
        output, hidden_state = self.lstm(input_combine, hidden_state)

        # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        output = output.squeeze(0)

        # linear
        output = self.linear(output)

        # softmax
        output = self.softmax(output)

        return output, hidden_state, attn_weights


'''
LuongAttnDecoder

'''


class LuongAttnDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout_ratio, padding_idx, tied, attn_method='concat'):

        super(LuongAttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.attn_method = attn_method

        # embedding
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_size, self.padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout_ratio)

        # attn
        self.attn = GlobalAttn(self.attn_method, self.hidden_size)

        # LSTM
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,
                            self.num_layers, bias=True,
                            batch_first=False, dropout=dropout_ratio)
        # concat linear
        self.concat_linear(self.hidden_size * 2, self.hidden_size)

        # linear
        self.linear = nn.Linear(self.hidden_size,
                                self.vocab_size)

        if tied:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, encoder_outputs):
        '''
        input: [1, batch_size]  LongTensor

        hidden_state: [num_layers, batch_size, hidden_size]

        output: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
        hidden_state: (h_n, c_n)
        '''

        # embedded
        input = input.view(1, -1)
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # Get current hidden state from input word and last hidden state
        output, hidden_state = self.lstm(embedded, hidden_state)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average<Paste>
        attn_weights = self.attn(output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        output = output.squeeze(0)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        concat_input = torch.cat((context, output), dim=1)
        concat_output = F.tanh(self.concat_linear(concat_input))

        # linear
        output = self.linear(concat_output)

        # softmax
        output = self.softmax(output)

        return output, hidden_state
