import os
import time
import random
import pdb

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from . import util

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# matplotlib colormaps:
# https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
# use white for zero
cmap = cm.bwr

m = cm.ScalarMappable(norm=norm, cmap=cmap)

COLORS = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

# visualize the patch match results for PatchLoss
def vis_patch_match(source, target, vis_patch, save_path, iter):
    # vis_patch: dict({'patch_scale': [[s_h, s_w], [t_h, t_w]], ....})
    # random select N pairs for visualization
    N = 5
    # source
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(util.tensor2im(source[0]))
    for k, v in vis_patch.items():
        ps = int(k)
        for i, ids in enumerate(v[:N]):
            s, t = ids
            s_w, s_h = s
            rect = plt.Rectangle((s_w, s_h), ps, ps, fill=False, edgecolor=COLORS[i])
            plt.gca().add_patch(rect)

    plt.savefig(save_path + '/pm_' + str(iter) + '_source.jpg', bbox_inches='tight')
    plt.close()
    # target
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(util.tensor2im((target[0])))
    for k, v in vis_patch.items():
        ps = int(k)
        for i, ids in enumerate(v[:N]):
            s, t = ids
            t_w, t_h = t
            rect = plt.Rectangle((t_w, t_h), ps, ps, fill=False, edgecolor=COLORS[i])
            plt.gca().add_patch(rect)

    pm_path = save_path + '/pm_' + str(iter) + '_target.jpg'
    plt.savefig(pm_path, bbox_inches='tight')
    print("saved patch match results at {}".format(save_path + '/pm_' + str(iter) + '_target.jpg'))
    plt.close()



def feat_vis(feats):
    """
    Visualize feat map across spatial as heat map
    :param feats: torch.size(1, n, h, w)
    :return: heat map
    """
    f = np.mean(feats.detach().cpu().numpy(), axis=1)[0]
    hm = m.to_rgba(f)[:, :, :3]
    return hm


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.save_path = self.opt.exp_path
        if opt.isTrain:
            self.log_name = os.path.join(self.save_path, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
            # use tensorboard when training
            self.writer = SummaryWriter(self.save_path)
        else:
            self.img_dir = self.save_path
            print('output image directory: %s' % self.img_dir)
            util.mkdirs(self.img_dir)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, step):

        # convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        if self.opt.isTrain:
            for label, image_numpy in visuals.items():
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                # convert (h, w, 3) to (3, h, w) for tensorboard writer
                self.writer.add_image(label, image_numpy.transpose([2, 0, 1]), step)
        else:
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'iter%d_%s_%d.png' % (step, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'iter%d_%s.png' % (step, label))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    util.save_image(image_numpy, img_path)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.opt.isTrain:
            for tag, value in errors.items():
                value = value.cpu().mean().float()
                self.writer.add_scalar(tag, value, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, i, errors, t):
        message = '(iters: %d, time: %.3f) ' % (i, t)
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals, gray=False):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key and not gray:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            elif 'input_label' == key and gray:
                t = util.tensor2labelgray(t, self.opt.label_nc + 2, tile=tile)
            elif isinstance(t, np.ndarray):
                # do nothing for already converted
                pass
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, output_path, visuals, name, gray=False):
        visuals = self.convert_visuals_to_numpy(visuals, gray=gray)

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(output_path, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)