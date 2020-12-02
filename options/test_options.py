from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # Migrated from semia
        parser.add_argument('--input_name', help='input image dir', default='example/zebra')
        parser.add_argument('--ext', help='img file extension, jpg or png', default='png')
        parser.add_argument('--target_seg_dir', help='target seg dir', default='example/zebra/cond')
        parser.add_argument('--output_dir', help='output folder', default='output')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')

        self.isTrain = False
        return parser
