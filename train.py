import os
import pdb
import random
import time
import torch
from collections import OrderedDict

from options.train_options import TrainOptions
from semia.dataset import ImgDataset, TestImgDataset
from semia.model import SemIAModel
from util.util import read_image, pil2tensor, pil2np, np2tensor
from util.visualizer import Visualizer
from util.visualizer import feat_vis, vis_patch_match

if __name__ == "__main__":
    # Load configuration
    opt = TrainOptions().parse()

    # Load dataset from single image
    dataset = ImgDataset()
    dataset.initialize(opt)
    # record input_image size
    opt.width, opt.height = dataset.width, dataset.height

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    # Load test dataset
    test_dataset = TestImgDataset()
    test_dataset.initialize(opt)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                  drop_last=False)

    # Create complete model
    model = SemIAModel(opt)

    visualizer = Visualizer(opt)

    total_steps = 0
    # Main training loop
    for i, data in enumerate(dataloader):
        total_steps += 1
        start_time = time.time()

        if i % opt.zero_rec_freq == 0:
            # Zero reconstruction: using augmented image as input and condition
            mode_g = 'generator_rec'
            model.set_input(data, mode_g)
        else:
            # Sample mode: using input_image as input, tgt_image as condition
            mode_g = 'generator'
            model.set_input(data, mode_g)

        # train discriminator once before optimizing generator
        for j in range(opt.Dsteps):
            model.run_discriminator_one_step()
            # print([[d.mean() for d in p] for p in model.d_preds])

        # Record fake image before optimizing generator(same as sampling)
        if total_steps % opt.display_freq == 0:  # or total_steps % (opt.zero_rec_freq * 10) == 0:
            visuals = OrderedDict([('0_Sample/tgt_img', model.tgt_img),
                                   ('0_Sample/src_img', model.src_img),
                                   ('0_Sample/src_seg', model.src_seg),
                                   ('0_Sample/tgt_seg', model.tgt_seg),
                                   ('0_Sample/fake_sample', model.get_latest_generated())])
        # Training
        # train auxclassifier
        if opt.use_aux:
            if mode_g == 'generator_rec':
                for j in range(opt.Asteps):
                    model.run_aux_one_step()
        # train generator
        for j in range(opt.Gsteps):
            model.run_generator_one_step(mode_g=mode_g)

        # train discriminator once after optimizing generator
        for j in range(opt.Dsteps):
            model.run_discriminator_one_step()

        iter_time = time.time() - start_time


        # display sample results after optimization and features
        if total_steps % opt.display_freq == 0:  # or i % (opt.zero_rec_freq * 10) == 0:
            visuals.update({'0_Sample/fake_image': model.get_latest_generated()})
            if opt.E_use_FiLM or opt.D_use_FiLM:
                for j, feats in enumerate(model.rel_feats):
                    visuals.update({'Feats/rel_feats_ratio_{}'.format(str(j)): feat_vis(feats[0])})
                    visuals.update({'Feats/rel_feats_diff_{}'.format(str(j)): feat_vis(feats[1])})
                    visuals.update({'Feats/alphas_{}'.format(str(j)): feat_vis(feats[2])})
                    visuals.update({'Feats/betas_{}'.format(str(j)): feat_vis(feats[3])})
            else:
                # rel_feats is attn_maps
                pass
            visuals.update({'D_preds/d_preds_0_real_patch': model.real_patch})
            visuals.update({'D_preds/d_preds_1_fake_patch': model.fake_patch})
            for k, preds in enumerate(model.d_preds):
                for l, p in enumerate(preds):
                    visuals.update({'D_preds/d_preds_{}_{}'.format(str(k), str(l)): feat_vis(p)})
            visualizer.display_current_results(visuals, total_steps)

        # display reconstruction results
        if i % (opt.display_freq * opt.zero_rec_freq) == 0:  # or total_steps % (opt.zero_rec_freq * 10) == 0:
            visuals = OrderedDict([('Rec/src_img', model.src_img),
                                   ('Rec/fake_sample', model.get_latest_generated())])
            visualizer.display_current_results(visuals, total_steps)

        # save patch match pairs to exp_path
        if opt.debug and total_steps % opt.vis_patch_freq == 0:
            if model.patch_vis is not None:
                vis_patch_match(data['src_img'], model.get_latest_generated(), model.patch_vis,
                                opt.exp_path, total_steps)


        # inference(evaluation) during training
        if total_steps % opt.inference_freq == 0:
            visuals = OrderedDict([('Eval/0_src_img', data['src_img'])])
            for i, test_data in enumerate(test_dataloader):
                name = test_dataset.test_names[i]
                model.set_input(test_data, 'inference')
                eval_image = model.evaluate()
                # print(eval_image)
                visuals.update({'Eval/0_tgt_seg/{}'.format(name): test_data['tgt_seg'],
                                'Eval/0_tgt_img/{}'.format(name): eval_image})
            visualizer.display_current_results(visuals, total_steps)
            if total_steps > opt.stable_iter:
                save_dir = os.path.join(opt.output_dir, str(total_steps))
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                visualizer.save_images(save_dir, visuals, "img")

        # loss curve and console log
        if total_steps % opt.print_freq == 0 or i % (opt.print_freq * opt.zero_rec_freq) == 0:
            losses = model.get_latest_losses()
            visualizer.print_current_errors(total_steps,
                                            losses, iter_time)
            visualizer.plot_current_errors(losses, total_steps)
            
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (total_steps %d)' %
                  (total_steps))
            model.save('latest')

        if total_steps % opt.save_model_freq == 0:
            print('saving the model (total_steps %d)' %
                  (total_steps))
            model.save(total_steps)

        model.update_learning_rate(i)

    print('Training was successfully finished.')
