# basic building blocks

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision

import util.util as util
from semia.base_network import BaseNetwork


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'semia.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    assert (torch.cuda.is_available())
    net.cuda(opt.gpu)
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name('conv', 'encoder')
    return create_network(netE_cls, opt)


def define_Aux(opt):
    netAux_cls = find_network_using_name(opt.netAux, 'network')
    return create_network(netAux_cls, opt)


class Debug(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


dbg = Debug()


class LocalNorm(nn.Module):
    def __init__(self, num_features):
        super(LocalNorm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.get_local_mean = nn.AvgPool2d(33, 1, 16, count_include_pad=False)

        self.get_var = nn.AvgPool2d(33, 1, 16, count_include_pad=False)

    def forward(self, input_tensor):
        local_mean = self.get_local_mean(input_tensor)
        print(local_mean)
        centered_input_tensor = input_tensor - local_mean
        print(centered_input_tensor)
        squared_diff = centered_input_tensor ** 2
        print(squared_diff)
        local_std = self.get_var(squared_diff) ** 0.5
        print(local_std)
        normalized_tensor = centered_input_tensor / (local_std + 1e-8)

        return normalized_tensor  # * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def get_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer, opt):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        elif subnorm_type == 'sync_batch' and opt.mpdist:
            norm_layer = nn.SyncBatchNorm(get_out_channel(layer), affine=True)
        else:
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class ResnetBlock(nn.Module):
    """ A single Res-Block module """

    def __init__(self, dim, use_bias, use_bn=True):
        super(ResnetBlock, self).__init__()

        # A res-block without the skip-connection, pad-conv-norm-relu-pad-conv-norm
        if use_bn:
            self.conv_block = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(dim, dim // 4, kernel_size=1, bias=use_bias)),
                nn.BatchNorm2d(dim // 4),
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=use_bias)),
                nn.BatchNorm2d(dim // 4),
                nn.LeakyReLU(0.2, True),
                nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim, kernel_size=1, bias=use_bias)),
                nn.BatchNorm2d(dim))
        else:
            self.conv_block = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(dim, dim // 4, kernel_size=1, bias=use_bias)),
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=use_bias)),
                nn.LeakyReLU(0.2, True),
                nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim, kernel_size=1, bias=use_bias)))

    def forward(self, input_tensor):
        # The skip connection is applied here
        return input_tensor + self.conv_block(input_tensor)


class ConvBaseBlock(nn.Module):
    def __init__(self, in_nc, out_nc, kernel=3, stride=1, pad=1, use_bn=True):
        # pad = (kernel_size - 1) // 2
        super(ConvBaseBlock, self).__init__()
        # use ReflectionPad instead of zero_pad default in conv2d
        # add spectral_norm outside conv2d
        if use_bn:
            self.main = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.utils.spectral_norm(nn.Conv2d(in_nc, out_nc, kernel, stride, bias=False)),
                nn.BatchNorm2d(out_nc),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.main = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.utils.spectral_norm(nn.Conv2d(in_nc, out_nc, kernel, stride, bias=False)),
                nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)


class DownConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc, pad=1):
        # normally, out_nc = in_nc * 2
        super(DownConvBlock, self).__init__()
        # add a resnet block before downscale or upscale
        self.res_block = ResnetBlock(in_nc, use_bias=True)
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_nc, in_nc, 3, stride=1, padding=pad)),
            nn.BatchNorm2d(in_nc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(in_nc, out_nc, 4, stride=2, padding=pad)),
            nn.BatchNorm2d(out_nc),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.res_block(x)
        return self.main(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc, pad=1):
        # normally, out_nc = in_nc / 2
        super(UpConvBlock, self).__init__()
        # add a resnet block before downscale or upscale
        self.res_block = ResnetBlock(in_nc, use_bias=True)
        self.main = nn.Sequential(
            # use in_nc -> out_nc*2 instead of in_nc -> in_nc
            # use kernel_size = 3 and padding = 1 for keeping out feats the same
            nn.utils.spectral_norm(nn.Conv2d(in_nc, out_nc * 2, 3, stride=1, padding=pad)),
            nn.BatchNorm2d(out_nc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # use kernel_size = 4 and padding = 1 for reverse conv2d(k=3, p=1)
            # ref: pytorch-CycleGAN-and-Pix2pix's UNet Block
            nn.utils.spectral_norm(nn.ConvTranspose2d(out_nc * 2, out_nc, 4, stride=2, padding=pad)),
            nn.BatchNorm2d(out_nc),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.res_block(x)
        return self.main(x)


class VAEBlock(nn.Module):
    def __init__(self, fc_in, fc_nc):
        super(VAEBlock, self).__init__()
        self.mu_fc = nn.Linear(fc_in, fc_nc)
        self.var_fc = nn.Linear(fc_in, fc_nc)

    def forward(self, x):
        # flatten to CxHxW, should equal to self.vae_tail_fc_in(code_c_in)
        temp = x.view(1, -1)
        mu = self.mu_fc(temp)
        logvar = self.var(temp)
        return mu, logvar


class RescaleBlock(nn.Module):
    def __init__(self, n_layers, scale=0.5, base_channels=64, use_bias=True):
        super(RescaleBlock, self).__init__()

        self.scale = scale

        self.conv_layers = [None] * n_layers

        in_channel_power = scale > 1
        out_channel_power = scale < 1
        i_range = range(n_layers) if scale < 1 else range(n_layers - 1, -1, -1)

        for i in i_range:
            self.conv_layers[i] = nn.Sequential(nn.ReflectionPad2d(1),
                                                nn.utils.spectral_norm(nn.Conv2d(
                                                    in_channels=base_channels * 2 ** (i + in_channel_power),
                                                    out_channels=base_channels * 2 ** (i + out_channel_power),
                                                    kernel_size=3,
                                                    stride=1,
                                                    bias=use_bias)),
                                                nn.BatchNorm2d(base_channels * 2 ** (i + out_channel_power)),
                                                nn.LeakyReLU(0.2, True))
            self.add_module("conv_%d" % i, self.conv_layers[i])

        if scale > 1:
            self.conv_layers = self.conv_layers[::-1]

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, input_tensor, pyramid=None, return_all_scales=False, skip=False):

        feature_map = input_tensor
        all_scales = []
        if return_all_scales:
            all_scales.append(feature_map)

        for i, conv_layer in enumerate(self.conv_layers):

            if self.scale > 1.0:
                feature_map = F.interpolate(feature_map, scale_factor=self.scale, mode='nearest')

            feature_map = conv_layer(feature_map)

            if skip:
                feature_map = feature_map + pyramid[-i - 2]

            if self.scale < 1.0:
                feature_map = self.max_pool(feature_map)

            if return_all_scales:
                all_scales.append(feature_map)

        return (feature_map, all_scales) if return_all_scales else (feature_map, None)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeeplabNetwork(BaseNetwork):
    '''
    aux network for segmentation
    remove BatchNorm since we only have 1 image per iter
    credits to https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
    '''

    def __init__(self, opt, num_classes=3):
        super(DeeplabNetwork, self).__init__()
        self.opt = opt
        # use less channels opt.nfc = 64
        N = int(opt.Aux_base_nc)

        # use [8, 16] instead of [12, 24, 36] since the max_size of image is 250
        self.body = ASPP(opt.im_nc, N, [8, 16])
        self.tail = nn.Sequential(
            nn.Conv2d(N, N, 3, padding=1, bias=False),
            nn.BatchNorm2d(N),
            nn.ReLU(),
            nn.Conv2d(N, num_classes, 1))

    def forward(self, x):
        x = self.body(x)
        return self.tail(x)


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False, local_pretrained_path='models/vgg19.pth'):
        super().__init__()
        # https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg19
        # vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        model = torchvision.models.vgg19()
        model.load_state_dict(torch.load(local_pretrained_path))
        vgg_pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
