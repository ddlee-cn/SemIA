import argparse
import os
import pickle

import torch


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Migrated from semia
        # workspace:
        parser.add_argument('--cuda', action='store_true', help='enables cuda', default=1)

        # load, input, save configurations:
        parser.add_argument('--netG', default='SemIA', help="path to netG (to continue training)")
        parser.add_argument('--netD', default='FPSE', help="path to netD (to continue training)")
        parser.add_argument('--use_aux', type=bool, default=True,
                            help='whether to use auxilary classfier for segmentation')
        parser.add_argument('--netAux', default='Deeplab')
        parser.add_argument('--load_from_opt_file', action='store_true',
                            help='load the options from checkpoints and use that as default')
        parser.add_argument('--align_seg', type=bool, help='whether to align colors of segmentation', default=False)

        # experiments
        parser.add_argument('--exp_root', default='', type=str,
                            help='exp output root path, default checkpoint_dir/G_D_input/')
        parser.add_argument('--exp_path', default='', type=str,
                            help='exp output path, default checkpoint_dir/G_D_input/expr_{}')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./models', help='models are saved here')

        # input/output sizes
        parser.add_argument('--resize', type=float, default=None, help="resize factor for image, seg")
        parser.add_argument('--im_nc', type=int, help='image # channels', default=3)
        parser.add_argument('--cond_nc', type=int, default=3, help='# of Cond channel')
        parser.add_argument('--label_nc', type=int, default=3,
                            help='# of input label classes without unknown class.')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for generator
        # since the third layer of alphas(FiLM) almost produce the same across spatial,
        # reduce num_up to 2 and increase base_nc
        parser.add_argument('--num_up', type=int, default=3)
        parser.add_argument('--base_nc', type=int, default=32)
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')
        parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
        parser.add_argument('--norm_G', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--R_patch', type=bool, default=False, help='use random patch L1 loss for reconstruction')
        parser.add_argument('--E_use_FiLM', default=True, action='store_true', help='use FiLM in Encoder')
        parser.add_argument('--E_use_attn', default=False, action='store_true', help='use Attention in Encoder')
        parser.add_argument('--D_use_FiLM', default=True, action='store_true', help='use FiLM in Decoder')
        parser.add_argument('--D_use_attn', default=False, action='store_true', help='use Attention in Decoder')
        parser.add_argument('--D_use_skip', default=False, action='store_true', help='use Skip Feats in Decoder')

        # for discriminator
        parser.add_argument('--D_base_nc', type=int, default=64,
                            help='# of discriminator filters(base_nc) in first conv layer')
        parser.add_argument('--D_patch', type=bool, default=False, help='use random patch for Discriminator')
        parser.add_argument('--Aux_base_nc', type=int, default=128)

        # normalization layer of sync_batch
        parser.add_argument('--mpdist', action='store_true', help='use distributed multiprocessing')

        # for instance-wise features
        parser.add_argument('--no_instance', default=True, help='if specified, do *not* add instance map as input')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt):
        file_name = os.path.join(opt.exp_path, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # data path
        opt.input_image_path = os.path.join(opt.input_name, 'img.' + opt.ext)
        opt.input_seg_path = os.path.join(opt.input_name, 'label.' + opt.ext)
        if opt.isTrain:
            # when training, use input_name/cond/*.png images as evaluation
            opt.target_seg_dir = os.path.join(opt.input_name, 'cond')

        if not isinstance(opt.gpu_ids, list):
            # set gpu ids
            str_ids = opt.gpu_ids.split(',')
            opt.gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    opt.gpu_ids.append(id)
            if len(opt.gpu_ids) > 0:
                torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        opt.must_divide = 2 ** (opt.num_up)
        opt.num_down = opt.num_up
        opt.gpu = opt.gpu_ids[0]

        opt.name = opt.netG + '_' + opt.netD + '_' + opt.input_name.split('/')[-1]

        if opt.isTrain:
            # experiment root path
            if opt.exp_root is '':
                opt.exp_root = os.path.join(opt.checkpoints_dir, opt.name)
            if not os.path.isdir(opt.exp_root):
                os.mkdir(opt.exp_root)
            if opt.exp_path is '':
                count = 1
                while True:
                    if os.path.isdir(os.path.join(opt.exp_root, 'expr_{}'.format(count))):
                        count += 1
                    else:
                        break
                opt.exp_path = os.path.join(opt.exp_root, 'expr_{}'.format(count))
                os.mkdir(opt.exp_path)

            opt.output_dir = os.path.join(opt.exp_path, "output")
            if not os.path.isdir(opt.output_dir):
                os.mkdir(opt.output_dir)

            print("EXP Path:", opt.exp_path)

            if opt.debug:
                opt.resize = 0.4
                opt.inference_freq = 3
                opt.display_freq = 2
                opt.print_freq = 1
                # opt.vis_patch_freq = 3
                # opt.zero_rec_freq = 2
                # opt.save_latest_freq = 2

        self.print_options(opt)

        self.opt = opt
        if save:
            self.save_options(opt)
        return self.opt
