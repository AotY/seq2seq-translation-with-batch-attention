import random

import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from decoder_attn import BahdanauAttnDecoder
from decoder_attn import LuongAttnDecoder

from vocab import EOS_id


class Seq2seq(nn.Module):
    def __init__(self, encoder_vocab_size, encoder_embedding_size,
                 encoder_hidden_size, encoder_num_layers, encoder_bidirectional,
                 decoder_vocab_size, decoder_embedding_size,
                 decoder_hidden_size, decoder_num_layers,
                 dropout_ratio, padding_idx, tied):

        # super class init
        super(Seq2seq, self).__init__()

        # encoder
        self.encoder = Encoder(encoder_vocab_size, encoder_embedding_size,
                               encoder_hidden_size, encoder_num_layers,
                               encoder_bidirectional, dropout_ratio,
                               padding_idx)

        # decoder
        # BahdanauAttnDecoder
        self.decoder = Decoder(decoder_vocab_size, decoder_embedding_size,
                               decoder_hidden_size, decoder_num_layers,
                               dropout_ratio, padding_idx, tied)

    #  def train(self, encoder_inputs, encoder_input_lengths, decoder_input, decoder_targets, batch_size, max_len, teacher_forcing_ratio=0.5):
        #  return self.forward(encoder_inputs, encoder_input_lengths, decoder_input, decoder_targets, batch_size, max_len, teacher_forcing_ratio)

    def forward(self, encoder_inputs, encoder_input_lengths, decoder_input, decoder_targets, batch_size, max_len, teacher_forcing_ratio):
        '''
        input:
            encoder_inputs: [seq_len, batch_size]
            encoder_input_lengths: [batch_size]
            decoder_input: [1, batch_size], first step: [sos * batch_size]
            decoder_targets: [seq_len, batch_size]

        '''
        # encoder
        encoder_hidden_state = self.encoder.init_hidden(batch_size)

        encoder_outputs, encoder_hidden_state = self.encoder(
            encoder_inputs, encoder_input_lengths, encoder_hidden_state)

        # decoder

        # encoder_hidden_state -> [num_layers * num_directions, batch, hidden_size]
        decoder_hidden_state = tuple(
            [item[:2, :, :] + item[2:, :, :] for item in encoder_hidden_state])

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = torch.zeros(
            (max_len, batch_size, self.decoder.vocab_size))

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(decoder_targets.shape[0]):
                output, decoder_hidden_state, attn_weights = self.decoder(
                    decoder_input, decoder_hidden_state, encoder_outputs)

                decoder_outputs[di] = output
                decoder_input = decoder_targets[di]

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(decoder_targets.shape[0]):
                output, decoder_hidden_state, attn_weights = self.decoder(
                    decoder_input, decoder_hidden_state, encoder_outputs)

                decoder_outputs[di] = output
                #  topv, topi = output.topk(1, dim=1)
                #  decoder_input = topi.squeeze().detach()
                decoder_input = torch.argmax(output).detach()

                ni = decoder_input.item()
                if ni == EOS_id:
                    break

        return encoder_outputs, decoder_outputs

    '''evaluate'''

    def evaluate(self, encoder_inputs, encoder_input_lengths, decoder_input, max_len, batch_size):
        '''
        encoder_inputs: [seq_len, batch_size], maybe [max_len, 1]

        decoder_input: [1, batch_size], maybe: [sos * 1]

        '''
        # encoder
        encoder_hidden_state = self.encoder.init_hidden(batch_size)

        encoder_outputs, encoder_hidden_state = self.encoder(
            encoder_inputs, encoder_input_lengths, encoder_hidden_state)

        # decoder
        decoder_hidden_state = tuple(
            [item[:2, :, :] + item[2:, :, :] for item in encoder_hidden_state])

        decoder_outputs = torch.zeros(
            (max_len, batch_size, self.decoder.vocab_size))

        for di in range(encoder_inputs.shape[0]):
            output, decoder_hidden_state, attn_weights = self.decoder(
                decoder_input, decoder_hidden_state, encoder_outputs)

            print('output: ', output.shape)
            decoder_outputs[di] = output
            #  topv, topi = output.topk(1, dim=1)
            #  decoder_input = topi.squeeze().detach()
            decoder_input = torch.argmax(output).detach()

            ni = decoder_input[0].item()
            if ni == EOS_id:
                break

        return decoder_outputs

