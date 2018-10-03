import os
import sys
import shutil

import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from vocab import Vocab
from vocab import PAD_id, SOS_id, EOS_id
from seq2seq import Seq2seq
from data_set import DataSet

from train_opt import data_set_opt, model_opt, train_opt

program = os.path.basename(sys.argv[0])

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger(program)
logger.info("Running %s", ' '.join(sys.argv))

# get optional parameters
parser = argparse.ArgumentParser(description=program,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
data_set_opt(parser)
model_opt(parser)
train_opt(parser)
opt = parser.parse_args()

''' seq2seq_mode'''


def build_model(encoder_vocab, decoder_vocab):
    model = Seq2seq(encoder_vocab_size=encoder_vocab.get_vocab_size(),
                    encoder_embedding_size=opt.encoder_embedding_size,
                    encoder_hidden_size=opt.encoder_hidden_size,
                    encoder_num_layers=opt.encoder_num_layers,
                    encoder_bidirectional=opt.encoder_bidirectional,
                    decoder_vocab_size=decoder_vocab.get_vocab_size(),
                    decoder_embedding_size=opt.decoder_embedding_size,
                    decoder_hidden_size=opt.decoder_hidden_size,
                    decoder_num_layers=opt.decoder_num_layers,
                    dropout_ratio=opt.dropout_ratio,
                    padding_idx=PAD_id,
                    tied=opt.tied)

    print(model)
    return model


def build_optimizer(model):
    optimizer = optim.Adam(model.parameters(), opt.lr)
    return optimizer


def train_epochs(data_set, model, optimizer, criterion):
    model.train() # set to train state
    for epoch in range(1, opt.epochs + 1):
        logger.info('---------------- epoch: %d --------------------' % (epoch))
        data_set.shuffle()

        total_loss = 0

        iters = data_set.get_size() // opt.batch_size + 1

        for iter in range(iters):
            encoder_inputs, decoder_targets, \
                encoder_inputs_lengths, decoder_targets_length = data_set.next_batch(
                    batch_size=opt.batch_size, max_len=opt.max_len)

            loss = train(encoder_inputs, encoder_inputs_lengths,
                         decoder_targets, decoder_targets_length,
                         model, optimizer, criterion)

            total_loss += loss

            if iter % opt.log_interval == 0:
                logger.info('train epoch: %d\titer/iters: %d%%\tloss: %.4f' %
                            (epoch, iter / iters * 100, total_loss / (iter + 1)))
                # reset total_loss
                total_loss = 0

        # save model of each epoch
        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        # save checkpoint, including epoch, seq2seq_mode.state_dict() and
        # optimizer.state_dict()
        save_checkpoint(state=save_state,
                        is_best=False,
                        filename=os.path.join(opt.model_save_path, 'checkpoint.epoch-%d.pth' % epoch))

        #  logger.info('train epoch: %d\taverage loss: %.4f' %
        #  (epoch, total_loss / iters))


def train(encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length, model, optimizer, criterion):
    # decoder input
    decoder_input = torch.ones(
        (1, opt.batch_size), dtype=torch.long) * SOS_id

    #  print(encoder_inputs)
    #  print(encoder_inputs_length)
    # model forward
    encoder_outputs, decoder_outputs = model(encoder_inputs, encoder_inputs_length,
                                                   decoder_input, decoder_targets,
                                                   opt.batch_size, opt.max_len,
                                                   opt.teacher_forcing_ratio)

    loss = 0

    # clear the gradients off all optimzed
    optimizer.zero_grad()

    # compting loss
    decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])
    decoder_targets = decoder_targets.view(-1)

    #  print(decoder_outputs.shape)
    #  print(decoder_targets.shape)

    loss = criterion(decoder_outputs, decoder_targets)

    # computes the gradient of current tensor, graph leaves.
    loss.backward()

    # performs a single optimization setp.
    optimizer.step()

    #  loss = float(loss)
    #  logger.info('iter loss: %.4f' % (loss))

    return loss.item()
    #  return loss / float(sum(decoder_targets_length))


def evaluate(model, data_set, sentence):
    model.eval() # set to evaluate state
    with torch.no_grad():
        # to ids
        encoder_inputs, encoder_inputs_length = data_set.input_to_ids(
            sentence, opt.max_len, 'english')

        print(encoder_inputs)

        # to model
        decoder_input = torch.ones((1, 1), dtype=torch.long) * SOS_id

        decoder_outputs = model.evaluate(encoder_inputs, encoder_inputs_length,
                                         decoder_input, opt.max_len, 1)

        #  print(decoder_outputs.shape) # [max_len, 1, vocab_size]

        decoder_outputs = decoder_outputs.squeeze(dim=1) # [max_len, vocab_size]

        # get outputs index
        _, outputs_index = decoder_outputs.topk(1, dim=1) # outputs_index -> [max_len, 1]
        outputs_index = outputs_index.view(-1).tolist() # [max_len]
        sentence = data_set.outputs_to_sentence(outputs_index, 'french')

        return sentence


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    '''
    Saving a model in pytorch.
    :param state: is a dict object, including epoch, optimizer, model etc.
    :param is_best: whether is the best model.
    :param filename: save filename.
    :return:
    '''
    save_path = os.path.join(opt.model_save_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth')


def load_checkpoint(filename='checkpoint.pth'):
    logger.info("Loding checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    return checkpoint


if __name__ == "__main__":

    # train()
    data_set = DataSet(opt.filename, opt.max_len, opt.min_count)

    model = build_model(data_set.english_vocab, data_set.french_vocab)
    # evaluate()

    # optim
    optimizer = build_optimizer(model)

    # loss function
    criterion = nn.NLLLoss(
        ignore_index=PAD_id,
        reduction='elementwise_mean'
    )

    if opt.train_from:
        checkpoint = load_checkpoint(opt.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        #  model.eval()
        optimizer.load_state_dict(checkpoint['optimizer'])

        while True:
            english = input("Please input an english sentence: ")
            french = evaluate(model, data_set, english)
            logger.info('evaluate:  english: %s ------> french: %s ' %
                        (english, french))
    else:
        # train
        train_epochs(data_set, model, optimizer, criterion)


# --filename ./eng-fra.txt --encoder_embedding_size 100 --encoder_hidden_size 100 --encoder_num_layers 2 --encoder_bidirectional --decoder_embedding_size 100 --decoder_hidden_size 100 --decoder_num_layers 2 --tied --dropout_ratio 0.5 --max_len 20 --lr 0.001 --epochs 5 --batch_size 128 --teacher_forcing_ratio 0.5 --seed 7  --device cpu --log_interval 50 --log_file ./logs/train.log --model_save_path ./models --start_epoch 0
