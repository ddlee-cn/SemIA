import torch
import torch.nn as nn
import torch.nn.functional as F

from semia.base_network import BaseNetwork
from semia.network import get_norm_layer


class FPSEDiscriminator(BaseNetwork):
    """
    Feature-Pyramid Semantics Embedding Discriminator
    It's more stable compared to my naive version. However, the Segmentation map embedding helps little.
    credits to https://github.com/xh-liu/CC-FPSE
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.D_base_nc
        input_nc = 3
        label_nc = opt.label_nc + (0 if opt.no_instance else 1)

        norm_layer = get_norm_layer(opt, opt.norm_D)

        # bottom-up pathway
        self.enc1 = nn.Sequential(
            norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1), opt),
            nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1), opt),
            nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1), opt),
            nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=2, padding=1), opt),
            nn.LeakyReLU(0.2, True))
        # self.enc5 = nn.Sequential(
        #     norm_layer(nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1), opt),
        #     nn.LeakyReLU(0.2, True))

        # top-down pathway
        self.lat2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=1), opt),
            nn.LeakyReLU(0.2, True))
        self.lat3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 4, kernel_size=1), opt),
            nn.LeakyReLU(0.2, True))
        self.lat4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=1), opt),
            nn.LeakyReLU(0.2, True))
        # self.lat5 = nn.Sequential(
        #     norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=1), opt),
        #     nn.LeakyReLU(0.2, True))

        # upsampling, set align_corners=True to omit warning:
        # UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0.
        # Please specify align_corners=True if the old behavior is desired.
        # see https://pytorch.org/docs/stable/nn.html?highlight=upsample#torch.nn.Upsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # final layers
        self.final2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1), opt),
            nn.LeakyReLU(0.2, True))
        self.final3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1), opt),
            nn.LeakyReLU(0.2, True))
        self.final4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1), opt),
            nn.LeakyReLU(0.2, True))

        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf * 2, 1, kernel_size=1)
        if not self.opt.no_seg_embed:
            self.seg = nn.Conv2d(nf * 2, nf * 2, kernel_size=1)
            self.embedding = nn.Conv2d(label_nc, nf * 2, kernel_size=1)

    def forward(self, fake_and_real_img, segmap=None):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        # feat14 = self.enc4(feat13)
        # feat15 = self.enc5(feat14)
        # top-down pathway and lateral connections
        # feat25 = self.lat5(feat15)
        # feat24 = self.lat4(feat14)
        feat23 = self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)
        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        # feat34 = self.final4(feat24)
        # Patch-based True/False prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        # pred4 = self.tf(feat34)

        # remove segmap embedding
        if segmap is not None:
            seg2 = self.seg(feat32)
            seg3 = self.seg(feat33)
            # seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching loss
        feats = [feat12, feat13]#, feat14]#, feat15]

        # segmentation map embedding
        if segmap is not None:
            segemb = self.embedding(segmap)
            segemb = F.avg_pool2d(segemb, kernel_size=2, stride=2)
            segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)
            segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)
            # segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)

            # semantics embedding discriminator score
            pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
            pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
            # pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3]  # , pred4]

        return [results, feats]
