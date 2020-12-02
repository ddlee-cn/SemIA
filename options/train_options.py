from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--input_name', help='input image dir', default='example/zebra')
        parser.add_argument('--ext', help='img file extension, jpg or png', default='png')
        parser.add_argument('--add_graph', help='visualize network graph via tensorboard', default=False, action='store_true')

        # data augmentation
        parser.add_argument('--angle', type=bool, help='rotation augmentation', default=False)
        parser.add_argument('--shift', type=bool, help='shift augmentation', default=False)
        parser.add_argument('--scale', type=bool, help='scale augmentation', default=True)
        parser.add_argument('--use_tps', type=bool, help='use TPS augmentation', default=True)
        parser.add_argument('--tps_max_vec_scale', type=float, default=0.02,
                            help='max random vector scale as width for TPS')
        parser.add_argument('--tps_points_per_dim', type=int, default=3)

        # optimization hyper parameters:
        parser.add_argument('--stable_iter', type=int, default=1000,
                            help='number of iters to train before stable augmentation')
        parser.add_argument('--zero_rec_freq', type=int, default=5, help='zero reconstruction training freq')
        parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=10)
        parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=1)
        parser.add_argument('--Asteps', type=int, help='Aux classifier inner steps', default=1)
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--max_iter', type=int, default=2500,
                            help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=1500,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0005')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

        # due to the change of augmentation and D structure, the hyperparameters may be slightly different with what we used in the paper
        parser.add_argument('--rec_alpha', default=5.0, type=float, help='reconstruction loss alpha')
        parser.add_argument('--fixedpoint_alpha', default=1.0, type=float,
                            help='reconstruction FiLM alpha and beta fixed point')
        parser.add_argument('--patch_alpha', default=1.0, type=float, help='patch loss weight for sample mode')
        parser.add_argument('--vgg_alpha', default=1.0, type=float, help='VGG loss in G weight')
        parser.add_argument('--feat_match_alpha', default=1.0, type=float, help='Feature Matching loss in G weight')
        parser.add_argument('--aux_alpha', default=1.0, type=float, help='Aux loss in G weight')

        parser.add_argument('--use_patch_seg', default=False, action='store_true',
                            help="use seg match instead of image match in PatchLoss")
        parser.add_argument('--patch_num', default=15, type=int, help='number of patches for measure from target')

        # for displays
        parser.add_argument('--inference_freq', type=int, default=50,
                            help='frequency of inference(use input_target_seg)')
        parser.add_argument('--display_freq', type=float, default=50,
                            help='frequency of showing training results on screen, scale_iter * freq')
        parser.add_argument('--print_freq', type=float, default=20,
                            help='frequency of showing training results on console, scale_iter * freq')
        parser.add_argument('--save_latest_freq', type=int, default=50,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_model_freq', type=int, default=100,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--vis_patch_freq', type=int, default=500,
                            help='visualize patch match in Seg-PatchLoss freq')
        parser.add_argument('--debug', action='store_true')

        # for discriminators
        parser.add_argument('--no_seg_embed', type=bool, default=True, help='use seg_embed layer in discriminator')
        parser.add_argument('--no_fm_loss', action='store_true', default=False,
                            help='if specified, do *not* use feature matching loss')
        parser.add_argument('--no_vgg_loss', type=bool, default=True, help='if specified, do *not* use VGG loss')
        parser.add_argument('--no_TTUR', action='store_true', help='Not use TTUR training scheme')

        # give up the restriction of search area, search over the whole image instead
        # parser.add_argument('--NN_dis', type=int, default=100,
        #                     help='Nearest Neighbour search area for PatchMatch, use min(H, W) for whole image searching')
        # parser.add_argument('--NN_num', default=10, type=int, help='Nearest Neighbour candidates from source')
        # parser.add_argument('--patch_scales', default=None, help='specified patch scales')
        # parser.add_argument('--min_scale', default=5, type=int, help='min patch scale')
        # parser.add_argument('--max_scale', default=9, type=int, help='max patch scale')
        # parser.add_argument('--scale_step', default=2, type=int, help='scale step between min and max')

        self.isTrain = True
        return parser
