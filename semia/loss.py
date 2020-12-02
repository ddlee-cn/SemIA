import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from semia.network import VGG19
from util.util import tensor2im, read_image


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode not in ['ls', 'hinge', 'wgan', 'original']:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            batchsize = input.size(0)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        ## computing loss is a bit complicated because |input| may not be
        ## a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


## KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


## Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu):
        super(VGGLoss, self).__init__()
        if gpu is not None:
            self.vgg = VGG19().cuda(gpu)
        else:
            self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# VGG version of SegPatchLoss
class SegPatchLoss(nn.Module):
    def __init__(self, gpu, patch_num):
        super(SegPatchLoss, self).__init__()
        # use vgg feature for patch loss, other feature extractors may also applicable
        if gpu is not None:
            self.vgg = VGG19().cuda(gpu)
        else:
            self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        # self.patch_scales = patch_scales  # patch scales
        # give the search area control, search over the whole image instead
        # self.NN_distance = NN_distance  # search area of NN
        # self.NN_num = NN_num  # patch num for NN matching
        self.patch_num = patch_num  # num of patch for measure inside single image

    def patch_concat(self, patches):
        # patches: list of [1, 3, H_p, W_p] tensor
        concated = torch.cat(patches, 0)
        return concated

    def weighted_L1_loss(self, src, tgt, weight):
        """L1 loss with given weight, for seg-aligned"""
        return torch.mean(weight * torch.abs(src - tgt))

    def forward(self, source_image, target_image, source_seg, target_seg):
        """
        Completeness: all patches of target image are from source image
        Coherence: no new patches in target image are not from source image
        source_image, target_image: [1, 3, H, W] Tensor
        """
        # accumulate loss across scales and random patches: Source->Target
        loss_cum = 0
        s_vgg, t_vgg = self.vgg(source_image), self.vgg(target_image)

        # hard coded hyper-params for patch search
        patch_scales = [3, 1]
        patch_nums = [10, 5]
        source_feats = s_vgg[0:2]
        target_feats = t_vgg[0:2]

        # only use first 2 features
        for patch_scale, source_feature, target_feature, patch_num in zip(patch_scales, source_feats, target_feats,
                                                                          patch_nums):
            H, W = source_feature.size()[2:]
            h_range = range(0, H - patch_scale)
            w_range = range(0, W - patch_scale)

            s_h_s, s_w_s = list(h_range), list(w_range)
            source_seg_patches = []
            target_seg_patches = []
            id_list = []
            for s_h in s_h_s:
                for s_w in s_w_s:
                    id_list.append([s_h, s_w])
                    source_seg_patches.append(source_feature[:, :, s_h:s_h + patch_scale, s_w:s_w + patch_scale])
            source_seg_tensor = self.patch_concat(source_seg_patches)

            for _ in range(patch_num):
                # random pick a patch from target
                t_h, t_w = random.randint(0, H - patch_scale), random.randint(0, W - patch_scale)
                # target_seg_patches = [target_seg_patch] * self.NN_num
                target_seg_patch = target_feature[:, :, t_h:t_h + patch_scale, t_w:t_w + patch_scale]

                target_seg_patches = [target_seg_patch] * (len(s_w_s) * len(s_h_s))
                # find closet seg patch
                target_seg_tensor = self.patch_concat(target_seg_patches)

                # reduction='none' for reduce per_batch difference, out_size: [NN_num, 3, patch_scale, patch_scale]
                seg_dis = F.l1_loss(source_seg_tensor, target_seg_tensor, reduction='none')
                # reduce along [3, patch_scale, patch_scale]
                seg_dis_reduce = torch.sum(seg_dis, [1, 2, 3])

                # Select top N
                top_n = 5
                best_seg_ids = list(seg_dis_reduce.argsort(0))[:top_n]
                for best_seg_id in best_seg_ids:
                    s_h, s_w = id_list[best_seg_id]
                    source_image_patch = source_feature[:, :, s_h:s_h + patch_scale, s_w:s_w + patch_scale]
                    loss_cum += F.l1_loss(source_image_patch, target_seg_patch) / top_n
        loss_cum /= self.patch_num
        return loss_cum, None
