""" Config class for search/augment """
import argparse
import os
from functools import partial
import torch

def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text

class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', type=str, default='cub_corrdi_v448_ps32_transfg0')
        parser.add_argument('--model_arch', type=str, default='b16', help='')
        parser.add_argument('--batch_size', type=int, default=16, help='train batch size')
        parser.add_argument('--num_classes', type=int, default=200, help='classes')
        # parser.add_argument('--lr1', type=float, default=0.0024, help='lr for weights')
        # parser.add_argument('--lr2', type=float, default=0.0012, help='lr for weights')
        parser.add_argument('--lr3', type=float, default=0.03, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        # parser.add_argument('--train_steps1', type=int, default=10000,
        #                     help='train_steps')
        # parser.add_argument('--train_steps2', type=int, default=10000,
        #                     help='train_steps')
        parser.add_argument('--train_steps3', type=int, default=10000,
                            help='train_steps')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='1', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--warmup_steps', type=int, default=500, help='warmup_steps')
        parser.add_argument('--image_size', type=int, default=448)
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--num_workers', type=int, default=8, help='# of workers')
        parser.add_argument('--patch_size', type=int, default=32, help='patch_size')
        parser.add_argument('--emb_dim', type=int, default=768, help='emb_dim')
        parser.add_argument('--mlp_dim', type=int, default=3072, help='mlp_dim')
        parser.add_argument('--num_heads', type=int, default=12, help='num_heads')
        parser.add_argument('--num_layers', type=int, default=12, help='num_layers')
        parser.add_argument('--attn_dropout_rate', type=float, default=0.0, help='attn_dropout_rate')
        parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout_rate')
        parser.add_argument('--tensorboard', action='store_true', default=False, help='tensorboard')
        parser.add_argument('--act', type=str, default='RELU', help='activation')

        # parser.add_argument('--genotype', default=None, help='Cell genotype')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.image_dir = image_dir
        
        self.checkpoint_path = checkpoint_path
        self.root_path = '/data/datasets'
        self.path = os.path.join('augments', self.name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.gpus = parse_gpus(self.gpus)
