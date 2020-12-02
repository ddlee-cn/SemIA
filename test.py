import os
from collections import OrderedDict
import torch

from options.test_options import TestOptions
from semia.model import SemIAModel
from semia.dataset import TestImgDataset
from util.visualizer import Visualizer
from util.util import read_image, pil2tensor, pil2np, np2tensor


if __name__ == "__main__":
    TestOptions = TestOptions()
    opt = TestOptions.parse()

    # Prepare data
    test_dataset = TestImgDataset()
    test_dataset.initialize(opt)
    # record input_image size
    opt.width, opt.height = test_dataset.width, test_dataset.height

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,drop_last=False)


    opt.gpu = 0
    opt.mpdist = False

    model = SemIAModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    out_root = os.path.join(opt.output_dir, opt.name)
    if not os.path.isdir(out_root):
        os.mkdir(out_root)

    for i, test_data in enumerate(test_dataloader):
        name = test_dataset.test_names[i]
        model.set_input(test_data, 'inference')
        eval_image = model.evaluate()
        visuals = {'./': eval_image}
        visualizer.save_images(out_root, visuals, name)