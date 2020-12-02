import random

import semia.network as networks
import util.util as util
from semia.base_model import BaseModel
from semia.loss import *
from util.util import rand_wz


class SemIAModel(BaseModel):
    def name(self):
        return 'SemIAModel'

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor

        self.netG, self.netD, self.netE, self.netAux = self.initialize_networks(opt)

        self.aux_losses = dict()

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionRec = torch.nn.L1Loss()
            # if opt.patch_scales is None:
            #    opt.patch_scales = list(range(opt.min_scale, opt.max_scale, opt.scale_step))
            # else:
            #    opt.patch_scales = [int(i) for i in opt.patch_scales.split(',')]
            self.criterionPatch = SegPatchLoss(opt.gpu, opt.patch_num)
            self.criterionAux = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = VGGLoss(self.opt.gpu)
            if not opt.no_fm_loss:
                self.criterionFeat = torch.nn.L1Loss()
            if opt.use_vae:
                self.KLDLoss = KLDLoss()
            self.old_lr = opt.lr

            self.create_optimizers(opt)
        self.generated = None

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            if opt.use_aux:
                Aux_params = list(self.netAux.parameters())
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        self.optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))

        if opt.isTrain:
            self.optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
            if opt.use_aux:
                self.optimizer_A = torch.optim.Adam(Aux_params, lr=G_lr, betas=(beta1, beta2))

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None
        netAux = networks.define_Aux(opt) if (opt.isTrain and opt.use_aux) else None
        self.model_names = ['G']
        self.model_names.append('D') if opt.isTrain else None
        self.model_names.append('E') if opt.use_vae else None
        self.model_names.append('Aux') if (opt.isTrain and opt.use_aux) else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
                if opt.use_aux:
                    netAux = util.load_network(netAux, 'Aux', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE, netAux

    def set_input(self, data, mode):
        for k, v in data.items():
            data[k] = v.cuda()
            # if self.opt.debug:
            #     print(k, v.shape)
        self.mode = mode
        if self.mode == 'generator':
            self.src_img, self.src_seg, self.tgt_img, self.tgt_seg = data['src_img'], data['src_seg'], data['tgt_img'], \
                                                                     data['tgt_seg']
            # random swap source and target
            if random.random() < 0.5:
                self.src_img, self.tgt_img = self.tgt_img, self.src_img
                self.src_seg, self.tgt_seg = self.tgt_seg, self.src_seg
        elif self.mode == 'generator_rec':
            self.src_img, self.src_seg, self.tgt_img, self.tgt_seg = data['tgt_img'], data['tgt_seg'], data[
                'tgt_img'].clone(), data['tgt_seg'].clone()
        else:
            self.src_img, self.src_seg, self.tgt_img, self.tgt_seg = data['src_img'], data['src_seg'], data['tgt_img'], \
                                                                     data['tgt_seg']

    # Entry point for all calls involving forward pass
    def forward(self, mode):
        if mode == 'generator':
            g_loss, generated, rel_feats, patch_vis = self.compute_generator_loss(
                self.src_seg, self.tgt_seg, self.src_img, self.tgt_img)
            return g_loss, generated, rel_feats, patch_vis
        elif mode == 'generator_rec':
            g_loss, generated, rel_feats, patch_vis = self.compute_generator_loss(
                self.src_seg, self.tgt_seg, self.src_img, self.tgt_img, mode='rec')
            return g_loss, generated, rel_feats, patch_vis
        elif mode == 'discriminator':
            d_loss, generated, real_patch, fake_patch, input_seg_patch, d_preds = self.compute_discriminator_loss(
                self.src_seg, self.tgt_seg, self.src_img, self.tgt_img)
            return d_loss, generated, real_patch, fake_patch, input_seg_patch, d_preds
        elif mode == 'sample':
            # when sample, produce augmented segmentation map at test time
            with torch.no_grad():
                fake_image, _, rel_feats = self.generate_fake(self.src_seg, self.tgt_seg, self.src_img)
            return fake_image, rel_feats
        elif mode == 'inference':
            # when inference, the input seg is aug_seg for generator
            with torch.no_grad():
                fake_image, _, _ = self.generate_fake(self.src_seg, self.tgt_seg, self.src_img, is_inference=True)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def shave(self, input_np, must_divide=8):
        # shave patch for U-Net like skip connection between encoder and decoder
        # the discriminator contains 3 downsampling
        input_np_shaved = input_np[:, :, :(input_np.shape[2] // must_divide) * must_divide,
                          :(input_np.shape[3] // must_divide) * must_divide]
        return input_np_shaved

    def compute_generator_loss(self, src_seg, tgt_seg, src_img, tgt_img, mode='sample'):
        G_losses = {}

        # when not use_FiLM, rel_feats is attn_maps, alpha_betas is None
        fake_image, alpha_betas, rel_feats = self.generate_fake(src_seg,
                                                                tgt_seg, src_img)

        # pairs for G: fake image be True, also use tgt_img for feat matching
        pairs = [[tgt_img, tgt_seg], [fake_image, tgt_seg]]
        d_preds, d_feats = self.discriminate(pairs)

        # G's goal: fake image be True
        G_losses['G/GAN'] = self.criterionGAN(d_preds[1], True, for_discriminator=False)

        patch_vis = None

        if mode == 'rec':
            if self.opt.E_use_FiLM or self.opt.D_use_FiLM:
                # Fixed point loss for alpha and beta from FiLM
                # the affine transform from rel_feats would be normalization if alpha=1, beta=0
                alpha_beta_targets = None
                if alpha_beta_targets is None:
                    alpha_beta_targets = []
                    for alpha, beta in alpha_betas:
                        alpha_beta_targets.append([torch.zeros_like(alpha) + 1, torch.zeros_like(beta)])
                alpha_loss = 0
                beta_loss = 0
                for a_b, a_b_t in zip(alpha_betas, alpha_beta_targets):
                    alpha_loss += self.criterionRec(a_b[0], a_b_t[0])
                    beta_loss += self.criterionRec(a_b[1], a_b_t[1])
                G_losses['G/alpha_fixedpoint'] = self.opt.fixedpoint_alpha * alpha_loss
                G_losses['G/beta_fixedpoint'] = self.opt.fixedpoint_alpha * beta_loss

            if self.opt.R_patch:
                # Random patch reconstruction loss
                wz_w, st_w, tt_w, = rand_wz(self.opt.width, low_b=0.4, up_b=0.6)
                wz_h, st_h, tt_h, = rand_wz(self.opt.height, low_b=0.4, up_b=0.6)
                aug_patch = tgt_img.clone()[:, :, tt_h:tt_h + wz_h, tt_w:tt_w + wz_w]
                fake_patch = fake_image.clone()[:, :, tt_h:tt_h + wz_h, tt_w:tt_w + wz_w]
                G_losses['G/Rec'] = self.opt.rec_alpha * self.criterionRec(aug_patch, fake_patch)

            else:
                # Whole image reconstruction loss
                G_losses['G/Rec'] = self.opt.rec_alpha * self.criterionRec(tgt_img, fake_image)
        else:
            if self.opt.use_aux:
                target = self.tgt_seg
                with torch.no_grad():
                    pred = self.netAux(fake_image.detach())
                G_aux_loss = self.criterionAux(target, pred)
                G_losses['G/AuxLoss'] = self.opt.aux_alpha * G_aux_loss
            if self.opt.use_patch_seg:
                patch_loss, patch_vis = self.criterionPatch(src_img, fake_image, src_seg, tgt_seg)
            else:
                patch_loss, patch_vis = self.criterionPatch(src_img, fake_image, src_img, fake_image)
            G_losses['G/PatchLoss'] = self.opt.patch_alpha * patch_loss

        if not self.opt.no_vgg_loss:
            G_losses['G/VGG'] = self.criterionVGG(fake_image, src_img) * self.opt.vgg_alpha

        if not self.opt.no_fm_loss:
            feat_fake, feat_real = d_feats[1], d_feats[0]
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            num_D = len(feat_fake)
            for i in range(num_D):
                GAN_Feat_loss += self.criterionFeat(
                    feat_fake[i], feat_real[i].detach()) * self.opt.feat_match_alpha / num_D
            G_losses['G/Feat'] = GAN_Feat_loss

        return G_losses, fake_image, rel_feats, patch_vis

    def compute_discriminator_loss(self, src_seg, tgt_seg, src_img, tgt_img):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _ = self.generate_fake(src_seg, tgt_seg, src_img)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        if self.opt.D_patch:
            # Random patch D loss
            wz_w, st_w, tt_w, = rand_wz(self.opt.width)
            wz_h, st_h, tt_h, = rand_wz(self.opt.height)
            real_patch = src_img.clone()[:, :, tt_h:tt_h + wz_h, tt_w:tt_w + wz_w]
            fake_patch = fake_image.clone()[:, :, tt_h:tt_h + wz_h, tt_w:tt_w + wz_w]
            input_seg_patch = src_seg.clone()[:, :, tt_h:tt_h + wz_h, tt_w:tt_w + wz_w]
            real_patch = self.shave(real_patch)
            fake_patch = self.shave(fake_patch)
            input_seg_patch = self.shave(input_seg_patch)

            pairs = [[real_patch, input_seg_patch], [fake_patch, input_seg_patch]]
            d_preds, _ = self.discriminate(pairs)
            D_losses['D/real_patch'] = self.criterionGAN(d_preds[0], True, for_discriminator=True)
            D_losses['D/fake_patch'] = self.criterionGAN(d_preds[1], False, for_discriminator=True)

            return D_losses, fake_image, real_patch, fake_patch, input_seg_patch, d_preds

        else:
            # pairs for D: True, False, False, False
            # use tgt_img as True sample instead of input_image(stays the same across all iters)
            pairs = [[tgt_img, tgt_seg], [fake_image, tgt_seg]]
            # [fake_image, src_seg], [tgt_img, src_seg]]
            d_preds, _ = self.discriminate(pairs)

            D_losses['D/real'] = self.criterionGAN(d_preds[0], True, for_discriminator=True)
            D_losses['D/fake'] = self.criterionGAN(d_preds[1], False, for_discriminator=True)
            # D_losses['D/mismatch_fake_image'] = self.criterionGAN(d_preds[2], False, for_discriminator=True)
            # D_losses['D/mismatch_tgt_img'] = self.criterionGAN(d_preds[3], False, for_discriminator=True)

            return D_losses, fake_image, tgt_img, fake_image, src_seg, d_preds

    def compute_aux_loss(self):
        pred = self.netAux(self.src_img.clone())
        aux_loss = self.criterionAux(self.src_seg.clone(), pred)
        return aux_loss

    def generate_fake(self, src_seg, tgt_seg, src_img, is_inference=False):
        # netG
        # input: real image, aug seg
        # output: fake_image

        if not is_inference:
            # Add noise to G input for better generalization (make it ignore the 1/255 binning)
            # similar to InGAN
            src_img = src_img + (torch.rand_like(src_img) - 0.5) * 2.0 / 255
        if self.opt.E_use_FiLM or self.opt.D_use_FiLM:
            fake_image, rel_feats, alpha_betas = self.netG(src_img, cond=src_seg,
                                                           cond_aug=tgt_seg)

            return fake_image, alpha_betas, rel_feats
        else:
            fake_image, attn_maps = self.netG(src_img, cond=src_seg, cond_aug=tgt_seg)

            return fake_image, None, attn_maps

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.
    def discriminate(self, pairs):
        imgs, segs = [], []
        for p in pairs:
            imgs.append(p[0])
            segs.append(p[1])

        fake_and_real_img = torch.cat(imgs, dim=0)

        if self.opt.no_seg_embed:
            discriminator_out = self.netD(fake_and_real_img,
                                          segmap=None)
        else:
            discriminator_out = self.netD(fake_and_real_img,
                                          segmap=torch.cat(segs, dim=0))

        preds, feats = self.divide_pred(discriminator_out, len(pairs))

        return preds, feats

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred, pair_length):
        preds = []
        feats = []
        for i in range(pair_length):
            # every pair has preds with length == D out layers
            preds.append([])
            feats.append([])
        # predictions
        for p in pred[0]:
            for j in range(pair_length):
                step = p.size(0) // pair_length
                preds[j].append(p[j * step:(j + 1) * step])
        # feats
        for f in pred[1]:
            for j in range(pair_length):
                step = f.size(0) // pair_length
                feats[j].append(f[j * step:(j + 1) * step])
        return preds, feats

    def run_generator_one_step(self, mode_g):
        self.optimizer_G.zero_grad()
        self.g_losses, self.generated, self.rel_feats, self.patch_vis = self.forward(mode_g)
        g_loss = sum(self.g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()

    def run_discriminator_one_step(self):
        self.optimizer_D.zero_grad()
        self.d_losses, self.generated, self.real_patch, self.fake_patch, self.input_seg_patch, self.d_preds = self.forward(
            mode='discriminator')
        d_loss = sum(self.d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()

    def run_aux_one_step(self):
        self.optimizer_A.zero_grad()
        aux_loss = self.compute_aux_loss()
        aux_loss.backward()
        self.aux_losses = {'Aux/l1_loss': aux_loss}
        self.optimizer_A.step()

    def evaluate(self):
        self.eval()
        generated = self.forward(mode='inference')
        self.train()
        return generated

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses, **self.aux_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, iter):
        util.save_network(self.netG, 'G', iter, self.opt)

    def reset_grads(self, require_grad):
        # reset grads for saved model
        for p in self.netG.parameters():
            p.requires_grad_(require_grad)

    def update_learning_rate(self, iter):
        if iter > self.opt.stable_iter:
            lrd = (self.opt.lr - self.opt.lr/10.0) / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            if iter % self.opt.inference_freq == 0:
                print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
