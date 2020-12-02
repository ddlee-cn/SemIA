import numpy as np
import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from sklearn.cluster import KMeans

from util.util import pil2np
from util.warp_image import warp_images


class ImgDataset(data.Dataset):
    def __init__(self):
        super(ImgDataset, self).__init__()

    def initialize(self, opt):
        self.opt = opt

        self.input_img = self.read_image(opt.input_image_path, resize=opt.resize)
        self.input_seg = self.read_image(opt.input_seg_path, resize=opt.resize)

        self.width, self.height = self.input_img.size

        self.input_img_t = self.pil2tensor(self.input_img, self.opt.must_divide)
        self.input_seg_t = self.pil2tensor(self.input_seg, self.opt.must_divide)

        arr = self.pil2np(self.input_seg).reshape((-1, 3))
        self.input_seg_arr = arr
        if self.opt.align_seg:
            # determinate seg color cluster centers for alignment
            kmeans = KMeans(n_clusters=6, random_state=0).fit(arr)
            labels = kmeans.labels_
            self.ori_seg_centers = kmeans.cluster_centers_
            x = self.ori_seg_centers[labels]
            self.input_seg_arr = x.reshape(self.pil2np(self.input_seg).shape)
            self.input_seg_t = self.np2tensor(self.input_seg_arr, self.opt.must_divide)

        self.init_affine_param()

    def __len__(self):
        return self.opt.max_iter

    def init_affine_param(self):
        if self.opt.angle == True:
            self.opt.angle = (-5, 5)  # random angle in range (-5, 5)
        if self.opt.shift == True:
            # self.opt.shift = (20, 3)  # maximum shift_x=20, shift_y=3
            shift_x = max(2, int(self.width * 0.1))
            shift_y = max(2, int(self.height * 0.1))
            self.opt.shift = (shift_x, shift_y)
        if self.opt.scale == True:
            self.opt.scale = 0.2

    def __getitem__(self, index):
        self.iter_factor = min(1, round(index / (self.opt.stable_iter) * 100) / 100)

        affine_param = self.getRandomAffineParam()
        flip = random.random() < (0.5 * self.iter_factor)
        apply_tps = random.random() < 0.5
        if self.opt.use_tps and apply_tps:
            tps_param = self.getTPSParam(self.input_img)
        else:
            tps_param = None
        aug_img = self.transform_image(self.input_img, affine=affine_param, tps=tps_param, flip=flip)
        aug_seg = self.transform_image(self.input_seg, affine=affine_param, tps=tps_param, is_seg=True, flip=flip)

        data_dict = {"src_img": self.input_img_t, "src_seg": self.input_seg_t,
                     "tgt_img": aug_img, "tgt_seg": aug_seg}

        return data_dict

    def read_image(self, input_image_path, resize=None):
        # read image into PIL.Image
        img = Image.open(input_image_path).convert('RGB')
        if resize:
            w, h = round(img.size[0] * resize), round(img.size[1] * resize)
            img = img.resize((w, h))
        return img

    def pil2np(self, pil_image):
        return (np.array(pil_image) / 255.0)

    def pil2tensor(self, pil_image, must_divide):
        # convert PIL.Image to torch.tensor
        input_np = (np.array(pil_image) / 255.0)

        input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide,
                          :(input_np.shape[1] // must_divide) * must_divide,
                          :]

        input_tensor = self.im2tensor(input_np_shaved)

        return input_tensor

    def np2tensor(self, input_np, must_divide):
        input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide,
                          :(input_np.shape[1] // must_divide) * must_divide,
                          :]

        input_tensor = self.im2tensor(input_np_shaved)

        return input_tensor

    def im2tensor(self, image_numpy, int_flag=False):
        # the int flag indicates whether the input image is integer (and [0,255]) or float ([0,1])
        if int_flag:
            image_numpy /= 255.0
        # Undo the tensor shifting (see tensor2im function)
        transformed_image = np.transpose(image_numpy, (2, 0, 1)) * 2.0 - 1.0
        return torch.FloatTensor(transformed_image)

    def getRandomAffineParam(self):
        angle = np.array(self.opt.angle) * self.iter_factor
        shift = np.array(self.opt.shift) * self.iter_factor
        # scale = np.array([1-self.opt.scale*self.iter_factor, 1+self.opt.scale*self.iter_factor])
        scale = np.array([1, 1 + self.opt.scale * self.iter_factor])
        if not self.opt.angle and not self.opt.scale and not self.opt.shift:
            affine_param = None
            return affine_param
        else:
            affine_param = dict()
            affine_param['angle'] = np.random.uniform(low=angle[0],
                                                      high=angle[1]) if self.opt.angle is not False else 0
            affine_param['scale'] = np.random.uniform(low=scale[0],
                                                      high=scale[1]) if self.opt.scale is not False else 1
            shift_x = np.random.uniform(low=-shift[0],
                                        high=shift[0]) if self.opt.shift is not False else 0
            shift_y = np.random.uniform(low=-shift[1],
                                        high=shift[1]) if self.opt.shift is not False else 0
            affine_param['shift'] = (shift_x, shift_y)

            return affine_param

    def getTPSParam(self, input_im):
        # input_im: PIL.Image class
        w, h = input_im.size
        np_im = np.array(input_im)
        src = _get_regular_grid(np_im,
                                points_per_dim=self.opt.tps_points_per_dim)
        vecs = _generate_random_vectors(src, scale=self.iter_factor * self.opt.tps_max_vec_scale * w)
        dst = src + vecs
        return {'src': src, 'vecs': vecs, 'dst': dst}

    def transform_image(self, image, affine=None, tps=None, is_seg=False, flip=False):
        if flip:
            image = F.hflip(image)
        if tps is not None:
            image = self.__apply_tps(image, tps)
        if affine is not None:
            angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=0)
        if is_seg and self.opt.align_seg:
            seg_arr = self.pil2np(image).reshape((-1, 3))
            # align augmented segmentation colors with original one
            # It's helpful for seg-consistent mask in weight L1 loss for patch
            # for 3*3 patches: 6/9 -> 8/9 is valid
            kmeans = KMeans(n_clusters=6, init=self.ori_seg_centers, n_init=1).fit(seg_arr)
            labels = kmeans.labels_
            x = self.ori_seg_centers[labels]
            image = x.reshape(self.pil2np(image).shape)
            image = self.np2tensor(image, self.opt.must_divide)
        image = self.pil2tensor(image, self.opt.must_divide)
        return image

    def __apply_tps(self, img, tps_params):
        np_im = np.array(img)
        np_im = tps_warp_2(np_im, tps_params['dst'], tps_params['src'])
        new_im = Image.fromarray(np_im)
        return new_im


class TestImgDataset(data.Dataset):
    def __init__(self):
        super(TestImgDataset, self).__init__()

    def initialize(self, opt):
        self.opt = opt

        self.input_img = self.read_image(opt.input_image_path, resize=opt.resize)
        self.input_seg = self.read_image(opt.input_seg_path, resize=opt.resize)

        self.width, self.height = self.input_img.size

        self.input_img_t = self.pil2tensor(self.input_img, self.opt.must_divide)
        self.input_seg_t = self.pil2tensor(self.input_seg, self.opt.must_divide)

        arr = self.pil2np(self.input_seg).reshape((-1, 3))
        self.input_seg_arr = arr
        if self.opt.align_seg:
            # determinate seg color cluster centers for alignment
            kmeans = KMeans(n_clusters=6, random_state=0).fit(arr)
            labels = kmeans.labels_
            self.ori_seg_centers = kmeans.cluster_centers_
            x = self.ori_seg_centers[labels]
            self.input_seg_arr = x.reshape(self.pil2np(self.input_seg).shape)
            self.input_seg_t = self.np2tensor(self.input_seg_arr, self.opt.must_divide)

        self.test_names, self.test_segs = [], []
        for file_name in os.listdir(opt.target_seg_dir):
            file = os.path.join(opt.target_seg_dir, file_name)
            im = self.read_image(file, resize=opt.resize)
            self.test_names.append(file_name.split('.')[0])
            self.test_segs.append(im)

    def __len__(self):
        return len(self.test_names)

    def __getitem__(self, index):
        im = self.test_segs[index]
        tgt_seg = self.transform_image(im, is_seg=True)

        data_dict = {"src_img": self.input_img_t, "src_seg": self.input_seg_t,
                     "tgt_img": self.input_img_t.clone(), "tgt_seg": tgt_seg}

        return data_dict

    def read_image(self, input_image_path, resize=None):
        # read image into PIL.Image
        img = Image.open(input_image_path).convert('RGB')
        if resize:
            w, h = round(img.size[0] * resize), round(img.size[1] * resize)
            img = img.resize((w, h))
        return img

    def pil2np(self, pil_image):
        return (np.array(pil_image) / 255.0)

    def pil2tensor(self, pil_image, must_divide):
        # convert PIL.Image to torch.tensor
        input_np = (np.array(pil_image) / 255.0)

        input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide,
                          :(input_np.shape[1] // must_divide) * must_divide,
                          :]

        input_tensor = self.im2tensor(input_np_shaved)

        return input_tensor

    def np2tensor(self, input_np, must_divide):
        input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide,
                          :(input_np.shape[1] // must_divide) * must_divide,
                          :]

        input_tensor = self.im2tensor(input_np_shaved)

        return input_tensor

    def im2tensor(self, image_numpy, int_flag=False):
        # the int flag indicates whether the input image is integer (and [0,255]) or float ([0,1])
        if int_flag:
            image_numpy /= 255.0
        # Undo the tensor shifting (see tensor2im function)
        transformed_image = np.transpose(image_numpy, (2, 0, 1)) * 2.0 - 1.0
        return torch.FloatTensor(transformed_image)

    def transform_image(self, image, is_seg=False):
        if is_seg and self.opt.align_seg:
            seg_arr = self.pil2np(image).reshape((-1, 3))
            # align augmented segmentation colors with original one
            # It's helpful for seg-consistent mask in weight L1 loss for patch
            # for 3*3 patches: 6/9 -> 8/9 is valid
            kmeans = KMeans(n_clusters=6, init=self.ori_seg_centers, n_init=1).fit(seg_arr)
            labels = kmeans.labels_
            x = self.ori_seg_centers[labels]
            image = x.reshape(self.pil2np(image).shape)
            image = self.np2tensor(image, self.opt.must_divide)
        image = self.pil2tensor(image, self.opt.must_divide)
        return image


# add TPS augmentation to improve generalization(augmentation to real-world editing)
# credits to https://github.com/eliahuhorwitz/DeepSIM
def _get_regular_grid(image, points_per_dim):
    nrows, ncols = image.shape[0], image.shape[1]
    rows = np.linspace(0, nrows, points_per_dim)
    cols = np.linspace(0, ncols, points_per_dim)
    rows, cols = np.meshgrid(rows, cols)
    return np.dstack([cols.flat, rows.flat])[0]


def _generate_random_vectors(src_points, scale):
    return np.random.uniform(-scale, scale, src_points.shape)


def _thin_plate_spline_warp(image, src_points, dst_points, keep_corners=True):
    width, height = image.shape[:2]
    if keep_corners:
        corner_points = np.array(
            [[0, 0], [0, width], [height, 0], [height, width]])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))
    out = warp_images(src_points, dst_points,
                      np.moveaxis(image, 2, 0),
                      (0, 0, width - 1, height - 1))
    return np.moveaxis(np.array(out), 0, 2)


def tps_warp(image, points_per_dim, scale):
    width, height = image.shape[:2]
    src = _get_regular_grid(image, points_per_dim=points_per_dim)
    dst = _generate_random_vectors(image, src, scale=scale * width)
    out = _thin_plate_spline_warp(image, src, dst)
    return out


def tps_warp_2(image, dst, src):
    out = _thin_plate_spline_warp(image, src, dst)
    return out
