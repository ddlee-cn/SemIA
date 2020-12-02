from torch.autograd import Variable as Vb

from semia.network import *


class Encoder(BaseNetwork):
    """
    Encoder for both Condition Signal(Segmentation map) and Image(+Noise)
    params: num_down, base_nc, out_nc
    return: [features] + [Code]
    """

    def __init__(self, num_down, base_nc, in_nc, out_nc,
                 input_FiLM=False, out_feats=False, out_shapes=False, use_VAE=False,
                 use_attn=False, code_in=None, code_fc=None):
        super().__init__()
        self.num_down = num_down
        self.base_nc = base_nc
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.input_FiLM = input_FiLM  # using input_FiLM as affine transformation
        self.out_feats = out_feats  # output feats
        self.out_shapes = out_shapes  # output feats shape for inverse
        self.use_VAE = use_VAE  # produce distribution for code
        self.use_attn = use_attn  # use attention mechanism

        if self.use_VAE:
            self.vae_tail_fc_in = code_in  # the flattened feat_cond(smallest) length
            self.vae_tail_fc_nc = code_fc  # mu and logvar length

        # Similar to InGAN, increase kernel_size of entry block to 7
        self.head_block = ConvBaseBlock(self.in_nc, self.base_nc, kernel=7, pad=3)
        down_block = []
        for i in range(self.num_down):
            # double channels after reduce spatial size
            nc_factor = 2 ** i
            down_block.append(DownConvBlock(self.base_nc * nc_factor, self.base_nc * nc_factor * 2))
        self.down_block = nn.ModuleList(down_block)
        if self.use_VAE:
            self.tail_block = VAEBlock(self.vae_tail_fc_in, self.vae_tail_fc_nc)
        else:
            self.tail_block = ConvBaseBlock(self.base_nc * (2 ** self.num_down),
                                            self.out_nc)
        if self.use_attn:
            attn_layers = []
            for i in range(self.num_down):
                # double channels after reduce spatial size
                nc_factor = 2 ** (i + 1)
                attn_layers.append(Cond_Attn(self.base_nc * nc_factor))
            self.attn_layers = nn.ModuleList(attn_layers)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def affine_transformation(self, X, alpha, beta):
        x = X.clone()
        mean, std = self.calc_mean_std(x)
        mean = mean.expand_as(x)
        std = std.expand_as(x)
        return alpha * ((x - mean) / std) + beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Vb(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return Vb(mu.data.new(mu.size()).normal_())

    def forward(self, input, noise=None, FiLM_alphas=None, FiLM_betas=None, k_feats=None, q_feats=None):
        # input: cond or prev_img
        if self.input_FiLM:
            assert len(FiLM_alphas) == len(FiLM_betas) == self.num_down, "FiLM_alphas and FiLM_betas mismatch"
        if self.use_attn:
            assert len(k_feats) == len(q_feats) == self.num_down, "k_feats and q_feats mismatch"

        feats, shapes, attn_maps = None, None, None
        if self.out_feats:
            feats = []
        if self.out_shapes:
            shapes = []
        # do not store attn_maps
        # if self.use_attn:
        #     attn_maps = []
        if noise is not None:
            input = torch.cat((input, noise), 1)
        x = self.head_block(input)
        for i in range(self.num_down):
            if self.out_shapes:
                # Output feature shape before DownSample
                shapes.append(x.shape[-2:])
            x = self.down_block[i](x)
            if self.input_FiLM:
                x = self.affine_transformation(x, FiLM_alphas[i], FiLM_betas[i])
            if self.use_attn:
                x, attn_map = self.attn_layers[i](x, k_feats[i], q_feats[i])
                # attn_maps.append(attn_map)
            if self.out_feats:
                # Out feat after DownSample and FiLM/Attention
                feats.append(x)

        if self.use_VAE:
            mu, logvar = self.tail_block(x)
            out = self.reparameterize(mu, logvar)
            out = out.view()
        else:
            out = self.tail_block(x)

        return out, feats, shapes, attn_maps


class Decoder(BaseNetwork):
    """
    Decoder for Image
    input: feature from encoder
    parmas: num_up, base_nc, in_nc
    return: Image 

    U-Net skip connections help little.
    """

    def __init__(self, num_up, base_nc, in_nc, out_nc,
                 input_FiLM=False, out_feats=False, in_shapes=False, skip_feats=False, use_attn=False):
        super().__init__()
        self.num_up = num_up
        self.base_nc = base_nc
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.input_FiLM = input_FiLM  # whether input FiLMed factors
        self.out_feats = out_feats  # whether output decoder features
        self.in_shapes = in_shapes  # whether interpolate feats according to in_shapes
        self.skip_feats = skip_feats  # whether concat skip feats from encoder
        self.use_attn = use_attn  # use attention mechanism

        # Decoder's head block out_nc = Encoder's tail block in_nc
        # self.base_nc * (2 ** self.num_up) = self.base_nc * (2 ** self.num_down)
        self.head_block = ConvBaseBlock(self.in_nc, self.base_nc * (2 ** self.num_up))
        up_block = []
        for i in range(self.num_up):
            nc_factor = 2 ** (self.num_up - i)
            if skip_feats:
                # double UpConv input channel, and half output channel 
                # for concating skip feats from encoder
                # torch.cat(feat, skip_feat) -> feat_next
                # 256 -> 64, 128 -> 32
                up_block.append(UpConvBlock(self.base_nc * nc_factor * 2, int(self.base_nc * nc_factor // 2)))
            else:
                up_block.append(UpConvBlock(self.base_nc * nc_factor, int(self.base_nc * nc_factor // 2)))
        self.up_block = nn.ModuleList(up_block)
        # Similar to InGAN, increase kernel_size of tail block of decoder to 7
        # Due to blurry edges, reduce the tail block kernel size back to 3
        self.tail_block = ConvBaseBlock(self.base_nc, self.out_nc, kernel=3, pad=1)
        if self.use_attn:
            attn_layers = []
            for i in range(self.num_up):
                # double channels after reduce spatial size
                nc_factor = 2 ** (self.num_up - i)
                attn_layers.append(Cond_Attn(self.base_nc * nc_factor))
            self.attn_layers = nn.ModuleList(attn_layers)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def affine_transformation(self, X, alpha, beta):
        x = X.clone()
        mean, std = self.calc_mean_std(x)
        mean = mean.expand_as(x)
        std = std.expand_as(x)
        return alpha * ((x - mean) / std) + beta

    def forward(self, code, skip_feats=None, in_shapes=None, FiLM_alphas=None, FiLM_betas=None, k_feats=None,
                q_feats=None):
        # code: code_img or code_cond
        if skip_feats is not None:
            assert len(skip_feats) == self.num_up, "skip feats number mismatch"
        if self.in_shapes:
            if in_shapes is not None:
                assert len(in_shapes) == self.num_up, "in_shapes number mismatch self.num_up"
            else:
                raise ValueError("in_shapes not in Input")
        if self.input_FiLM:
            assert len(FiLM_alphas) == len(FiLM_betas) == self.num_up, "FiLM_alphas and FiLM_betas mismatch"
        if self.use_attn:
            assert len(k_feats) == len(q_feats) == self.num_up, "k_feats and q_feats mismatch"

        feats, attn_maps = None, None
        if self.out_feats:
            feats = []
        # if self.use_attn:
        #     attn_maps = []
        x = self.head_block(code)
        for i in range(self.num_up):
            if self.input_FiLM:
                x = self.affine_transformation(x, FiLM_alphas[i], FiLM_betas[i])
            if self.use_attn:
                x, attn_map = self.attn_layers[i](x, k_feats[i], q_feats[i])
                # attn_maps.append(attn_map)
            if self.out_feats:
                # Out feat before UpSample/Concat and after FiLM/Attention
                feats.append(x)
            if skip_feats is not None:
                # merge skip feats before UpSample
                skip_feat = skip_feats[self.num_up - i - 1]
                if self.input_FiLM:
                    # also apply FiLM params on skip_feats
                    skip_feat = self.affine_transformation(skip_feat,
                                                           FiLM_alphas[i], FiLM_betas[i])
                if self.use_attn:
                    skip_feat, attn_map = self.attn_layers[i](skip_feat, k_feats[i], q_feats[i])
                    # attn_maps.append(attn_map)
                x = torch.cat((x, skip_feat), 1)
            x = self.up_block[i](x)
            if self.in_shapes:
                # interpolate feature size after UpSample
                # print(x.shape, in_shapes[self.num_up-i-1])
                # torch.Size([1, 64, 6, 10]) torch.Size([6, 10])
                # torch.Size([1, 32, 12, 20]) torch.Size([12, 20])
                # torch.Size([1, 16, 24, 40]) torch.Size([25, 40])
                x = F.interpolate(x, size=in_shapes[self.num_up - i - 1], mode='nearest')

        out = self.tail_block(x)

        return out, feats


class Cond_Attn(nn.Module):
    """
    Cond-Attention Module
    Attetion module may replace SFT module, but takes much more memory and brings a lot computational burden

    cond_feats as Key
    aug_cond_feats as Query
    image_feats as Value
    """

    def __init__(self, in_dim, bottleneck_factor=32):
        super(Cond_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // bottleneck_factor, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // bottleneck_factor, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x, k, q):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                k : cond feature maps( B X C X W X H)
                q : aug cond feature maps( B X C X W X H)
                k -> q as Transformation
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(q).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X (W*H) X C
        proj_key = self.key_conv(k).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N) every pixel has W*H scores
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class FiLM(BaseNetwork):
    """
    FiLMModule(Semantic Feature Translation layer)

    Our version seperates the scaling and shiftting, just keep the original naming
    """

    def __init__(self, base_nc, num_feats, double=False, reverse=False):
        super().__init__()
        self.base_nc = base_nc
        self.num_feats = num_feats
        self.reverse = reverse  # assume feats from big to small(more ncs)
        self.double = double  # whether the rel_feats are concated, instead of diff/ratio

        bottlenecks = []
        for i in range(num_feats):
            # nc_factor
            nc_factor = 2 ** (num_feats - i)
            if reverse:
                nc_factor = 2 ** (i + 1)
            # use base_nc * nc_factor // 2 as bottleneck depth
            # while Guided-pix2pix use fixed 100 across all feature maps
            bottlenecks.append(self.bottleneck_layer(base_nc * nc_factor, base_nc * nc_factor // 2))
        self.bottlenecks = torch.nn.ModuleList(bottlenecks)

    def bottleneck_layer(self, nc, bottleneck_depth):
        if self.double:
            block_list = [ConvBaseBlock(nc * 2, nc, kernel=1, pad=0)]
        else:
            block_list = []
        # add a resnet block in bottleneck layer for alpha and beta
        # update: remove bn in FiLM module
        block_list += [ResnetBlock(nc, use_bn=False, use_bias=True),
                       nn.utils.spectral_norm(nn.Conv2d(nc, bottleneck_depth, kernel_size=1)),
                       nn.ReLU(True),
                       nn.utils.spectral_norm(nn.Conv2d(bottleneck_depth, nc, kernel_size=1))]
        main = nn.Sequential(*block_list)
        return main

    def forward(self, feats):
        assert len(feats) == self.num_feats
        params = []
        for i in range(self.num_feats):
            # attach FiLM source features to main graph instead detach()
            # update: no need for add 1 for relative feats ratio
            # alpha & beta separate
            params.append(self.bottlenecks[i](feats[i]))

        return params


class SemIAGenerator(BaseNetwork):
    """
    netG
    input: real image(src_img), input_seg(src_seg), aug_seg(tgt_seg)
    output: fake_image(tgt_img)

    also output FiLM parameters(alpha, beta) for fixed-point loss and visualization
    """

    def __init__(self, opt):
        super(SemIAGenerator, self).__init__()
        self.num_down = opt.num_down  # Encoder feat layer num
        self.num_up = opt.num_up  # Decoder feat layer num
        # self.neck_depth = neck_depth  # FiLM layer bottleneck depth
        self.base_nc = opt.base_nc  # base channel size for conv layers
        self.cond_nc = opt.cond_nc  # Condition channel size, 3 for seg
        self.im_nc = opt.im_nc  # Image channel size, commonly 3
        self.opt = opt  # use FiLM or Cond-Attn
        code_c_nc, code_i_nc = self.base_nc * (2 ** self.num_down), self.base_nc * (2 ** self.num_down)

        self.encoder_c = Encoder(self.num_down, self.base_nc, self.cond_nc, code_c_nc, out_feats=True)
        # use noise + z_prev instead of torch.cat(noise+prev, prev) as input
        self.encoder_i = Encoder(self.num_down, self.base_nc, self.im_nc, code_i_nc,
                                 input_FiLM=self.opt.E_use_FiLM, use_attn=self.opt.E_use_attn,
                                 out_feats=True, out_shapes=False)
        self.decoder_i = Decoder(self.num_up, self.base_nc, code_i_nc, self.im_nc,
                                 skip_feats=self.opt.D_use_skip, input_FiLM=self.opt.D_use_FiLM,
                                 use_attn=self.opt.D_use_attn)

        if self.opt.E_use_FiLM or self.opt.D_use_FiLM:
            self.FiLM_c2i_alpha = FiLM(self.base_nc, self.num_up, reverse=True)
            self.FiLM_c2i_beta = FiLM(self.base_nc, self.num_up, reverse=True)

    def forward(self, x, cond=None, cond_aug=None):
        # print(x.shape, cond.shape)

        # Condition + FiLM(Feat_img) -> code_cond
        # _ denotes out_feats of ecnoder_c is None
        _, feats_cond, _, _ = self.encoder_c(cond)
        _, feats_cond_aug, _, _ = self.encoder_c(cond_aug)

        if self.opt.E_use_FiLM or self.opt.D_use_FiLM:
            # Relative feats between cond and cond_aug
            rel_feats_ratio = []  # use for alpha(multiplier of FiLM)
            for f_c, f_c_a in zip(feats_cond, feats_cond_aug):
                rel_feats_ratio.append(torch.div(f_c_a + 1e-14, f_c + 1e-14))  # feats_cond_aug / feats_cond
            rel_feats_diff = []  # use for beta(bias of FiLM)
            for f_c, f_c_a in zip(feats_cond, feats_cond_aug):
                rel_feats_diff.append(torch.add(f_c_a, -f_c))  # feats_cond_aug - feats_cond_aug

            # cond2img in Decoder: apply FiLM alpha and beta
            # Feat_cond -> alpha, beta
            alpha_conds = self.FiLM_c2i_alpha(rel_feats_ratio)
            beta_conds = self.FiLM_c2i_beta(rel_feats_diff)

            rel_feats_list = []  # for visualization
            alpha_beta_list = []  # for fixed-point loss in zero-reconstruction
            for fr, fd, a, b in zip(rel_feats_ratio, rel_feats_diff, alpha_conds, beta_conds):
                # shift rel_feats_ratio, alpha to around 0 for visualization
                rel_feats_list.append([fr.clone() - 1, fd.clone(), a.clone() - 1, b.clone()])
                alpha_beta_list.append([a, b])

        E_param_dict = {"FiLM_alphas": None, "FiLM_betas": None, "k_feats": None, "q_feats": None}
        if self.opt.E_use_FiLM:
            E_param_dict["FiLM_alphas"], E_param_dict["FiLM_betas"] = alpha_conds, beta_conds
        if self.opt.E_use_attn:
            E_param_dict["k_feats"], E_param_dict["q_feats"] = feats_cond, feats_cond_aug
        # Noise + Prev_img -> Feat_img, code_img
        code_i, feats_img, _, attn_maps = self.encoder_i(x, **E_param_dict)

        if not self.opt.D_use_skip:
            feats_img = None

        # code_img + FiLM(Feat_cond) -> Fake_img
        # _ denotes out_feats of decoder_i is None
        D_param_dict = {"FiLM_alphas": None, "FiLM_betas": None, "k_feats": None, "q_feats": None}
        if self.opt.D_use_FiLM:
            alpha_conds.reverse()
            beta_conds.reverse()
            D_param_dict["FiLM_alphas"], D_param_dict["FiLM_betas"] = alpha_conds, beta_conds
        if self.opt.D_use_attn:
            feats_cond.reverse()
            feats_cond_aug.reverse()
            D_param_dict["k_feats"], D_param_dict["q_feats"] = feats_cond, feats_cond_aug
        fake_img, _ = self.decoder_i(code_i, skip_feats=feats_img, **D_param_dict)

        if self.opt.E_use_FiLM or self.opt.D_use_FiLM:
            return fake_img, rel_feats_list, alpha_beta_list
        else:
            return fake_img, attn_maps
