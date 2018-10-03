import argparse


def data_set_opt(parser):
    # Data set options
    group = parser.add_argument_group('data set opt')
    group.add_argument('--filename',
                       type=str,
                       required=True,
                       help='path to the translation pair file.')


def model_opt(parser):
    group = parser.add_argument_group('model opt')

    group.add_argument('--encoder_embedding_size',
                       type=int,
                       default=100,
                       help='encoder embeddding size.'
                       )

    group.add_argument('--encoder_hidden_size',
                       type=int,
                       default=100,
                       help='encoder hidden size')

    group.add_argument('--encoder_num_layers',
                       type=int,
                       default=2,
                       help='encoder lstm num layers.'
                       )
    group.add_argument('--encoder_bidirectional', action='store_true',
                       help='is encoder bidirectional.')

    group.add_argument('--decoder_embedding_size',
                       type=int,
                       default=100,
                       help='decoder embeddding size.'
                       )

    group.add_argument('--decoder_hidden_size',
                       type=int,
                       default=100,
                       help='decoder hidden size')

    group.add_argument('--decoder_num_layers',
                       type=int,
                       default=2,
                       help='decoder lstm num layers.'
                       )

    group.add_argument('--teacher_forcing_ratio',
                       type=float,
                       default=0.5,
                       help='using teachder forcing when decoder.')

    group.add_argument('--tied',
                       action='store_true',
                       help='tie the word embedding and softmax weights'
                       )

    group.add_argument('--dropout_ratio',
                       type=float,
                       default=0.5,
                       help='dropout ratio.')

    group.add_argument('--max_len',
                       type=int,
                       default=20,
                       help='max seq len.')

    group.add_argument('--min_count',
                       type=int,
                       default=3,
                       help='filtering word by min count.')


def train_opt(parser):
    group = parser.add_argument_group('train opt')

    group.add_argument('--lr', type=float, default=0.001,
                       help='initial learning rate')

    group.add_argument('--epochs', type=int, default=5,
                       help='upper epoch limit')

    group.add_argument('--use_teacher_forcing', action='store_true',
                       help='is use teacher forcing.')

    group.add_argument('--seed',
                       type=int,
                       default=7,
                       help='random seed')

    group.add_argument('--device',
                       type=str,
                       default='cuda',
                       help='use cuda or cpu.')

    group.add_argument('--log_interval', type=int, default=200, metavar='N',
                       help='report interval')

    group.add_argument('--model_save_path',
                       type=str,
                       default='./models',
                       help='path to save models')

    group.add_argument('--log_file',
                       type=str,
                       help='path to save logger.')

    group.add_argument('--optim_method',
                       type=str,
                       default='adam',
                       help='''method (:obj:`str`): one of [sgd, adagrad, adadelta, adam] ''')

    group.add_argument('--batch_size',
                       type=int,
                       default=128,
                       help='batch size')

    group.add_argument('--train_from',
                       action='store_true',
                       help='loading checkpoint if we resume from a previous training.')

    group.add_argument('--checkpoint',
                       type=str, help='path to model s checkpoint.')
