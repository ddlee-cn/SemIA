import argparse
import importlib
import numpy as np
import os
import pickle
import random
import re
import torch
from PIL import Image
from scipy.stats import entropy, wasserstein_distance


def rand_wz(dim, low_b=0.25, up_b=0.5):
    # window_size
    wz = random.randint(int(dim * low_b), int(dim * up_b))
    # st: source_start
    st = random.randint(0, dim - wz)
    # tt: target_start
    tt = random.randint(0, dim - wz)
    return wz, st, tt


def read_image(input_image_path, resize=None):
    # read image into PIL.Image
    img = Image.open(input_image_path).convert('RGB')
    if resize:
        w, h = round(img.size[0] * resize), round(img.size[1] * resize)
        img = img.resize((w, h))
    return img


def shave(input_np, must_divide):
    input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide,
                      :(input_np.shape[1] // must_divide) * must_divide,
                      :]
    return input_np_shaved


def pil2np(pil_image):
    return (np.array(pil_image) / 255.0)


def pil2tensor(pil_image, must_divide):
    # convert PIL.Image to torch.tensor
    input_np = (np.array(pil_image) / 255.0)

    input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide,
                      :(input_np.shape[1] // must_divide) * must_divide,
                      :]

    input_tensor = im2tensor(input_np_shaved)

    return input_tensor


def np2tensor(input_np, must_divide):
    input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide,
                      :(input_np.shape[1] // must_divide) * must_divide,
                      :]

    input_tensor = im2tensor(input_np_shaved)

    return input_tensor


def js(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def ws(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p /= p.sum()
    q /= q.sum()
    return wasserstein_distance(p, q)


def move_to_gpu(t):
    t = t.to(torch.device('cuda'))
    return t


def read_shave_tensorize(image_path, must_divide):
    im = read_image(image_path)
    tensor = pil2tensor(im, must_divide)
    return tensor


def tensor2im(image_tensors, imtype=np.uint8):
    if not isinstance(image_tensors, list):
        image_tensors = [image_tensors]

    image_numpys = []
    for image_tensor in image_tensors:
        # Note that tensors are shifted to be in [-1,1]
        image_numpy = image_tensor.detach().cpu().float().numpy()

        if np.ndim(image_numpy) == 4:
            image_numpy = image_numpy.transpose((0, 2, 3, 1))

        image_numpy = np.round((image_numpy.squeeze(0) + 1) / 2.0 * 255.0)
        image_numpys.append(image_numpy.astype(imtype))

    if len(image_numpys) == 1:
        image_numpys = image_numpys[0]

    return image_numpys


def im2tensor(image_numpy, int_flag=False):
    # the int flag indicates whether the input image is integer (and [0,255]) or float ([0,1])
    if int_flag:
        image_numpy /= 255.0
    # Undo the tensor shifting (see tensor2im function)
    transformed_image = np.transpose(image_numpy, (2, 0, 1)) * 2.0 - 1.0
    return torch.FloatTensor(transformed_image).unsqueeze(0).cuda()


def image_concat(g_preds, d_preds=None, size=None):
    hsize = g_preds[0].shape[0] + 6 if size is None else size[0]
    results = []
    if d_preds is None:
        d_preds = [None] * len(g_preds)
    for g_pred, d_pred in zip(g_preds, d_preds):
        # noinspection PyUnresolvedReferences
        dsize = g_pred.shape[1] if size is None or size[1] is None else size[1]
        result = np.ones([(1 + (d_pred is not None)) * hsize, dsize, 3]) * 255
        if d_pred is not None:
            # d_pred_new = imresize((np.concatenate([d_pred] * 3, 2) - 128) * 2, g_pred.shape[0:2], interp='nearest')
            g_shape = g_pred.shape[0:2]
            d_pred_new = np.array(Image.fromarray((np.concatenate([d_pred] * 3, 2) - 128) * 2).resize((g_shape[1],
                                                                                                       g_shape[0])))
            # (256, 216, 3) (216, 256, 3), resize h, w swapped
            result[hsize - g_pred.shape[0]:hsize + g_pred.shape[0], :g_pred.shape[1], :] = np.concatenate([g_pred,
                                                                                                           d_pred_new],
                                                                                                          0)
        else:
            result[hsize - g_pred.shape[0]:, :, :] = g_pred
        results.append(np.uint8(np.round(result)))

    return np.concatenate(results, 1)


def save_image(image_tensor, image_path):
    image_pil = Image.fromarray(tensor2im(image_tensor), 'RGB')
    image_pil.save(image_path)


def get_scale_weights(i, max_i, start_factor, input_shape, min_size, num_scales_limit, scale_factor):
    num_scales = np.min([np.int(np.ceil(np.log(np.min(input_shape) * 1.0 / min_size)
                                        / np.log(scale_factor))), num_scales_limit])

    # if i > max_i * 2:
    #     i = max_i * 2

    factor = start_factor ** ((max_i - i) * 1.0 / max_i)

    un_normed_weights = factor ** np.arange(num_scales)
    weights = un_normed_weights / np.sum(un_normed_weights)
    #
    # np.clip(i, 0, max_i)
    #
    # un_normed_weights = np.exp(-((np.arange(num_scales) - (max_i - i) * num_scales * 1.0 / max_i) ** 2) / (2 * sigma ** 2))
    # weights = un_normed_weights / np.sum(un_normed_weights)

    return weights  # Python 3.x


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here
def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from 
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

        # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


# Converts a one-hot tensor into a gray label map
def tensor2labelgray(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2labelgray(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=False, gray=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    ## save to png
    if gray:
        if (image_numpy.shape) == 3:
            assert (image_numpy.shape[2] == 1)
            image_numpy = image_numpy.squeeze(2)
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path.replace('.jpg', '.png'))
    else:
        if len(image_numpy.shape) == 4:
            image_numpy = image_numpy[0]
        if len(image_numpy.shape) == 2 and not gray:
            image_numpy = np.expand_dims(image_numpy, axis=2)
        if image_numpy.shape[2] == 1 and not gray:
            image_numpy = np.repeat(image_numpy, 3, 2)
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (
            module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.exp_path, save_filename)
    torch.save(net.state_dict(), save_path)  # net.cpu() -> net


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = opt.exp_path
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path, map_location=torch.device('cpu'))
    # weights = {k: v for k, v in weights.items() if k in net.state_dict()}
    # torch.save(weights, 'checkpoints/new.pth')
    # pdb.set_trace()
    net.load_state_dict(weights)
    return net


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if N == 182:  # COCO
            important_colors = {
                'sea': (54, 62, 167),
                'sky-other': (95, 219, 255),
                'tree': (140, 104, 47),
                'clouds': (170, 170, 170),
                'grass': (29, 195, 49)
            }
            for i in range(N):
                name = util.coco.id2label(i)
                if name in important_colors:
                    color = important_colors[name]
                    cmap[i] = np.array(list(color))

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
