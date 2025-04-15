#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import re
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, scene2, load_iter):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # gaussians2 = GaussianModel(dataset2.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration= load_iter, shuffle=False)
    # scene2 = Scene(dataset2, gaussians2, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # (model_params, first_iter) = torch.load()
    # gaussians2.restore(model_params, opt)
    # gaussians2.load_ply('./output/vangogh0/point_cloud/iteration_30000/point_cloud.ply')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack2 = scene2.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_stack2 = scene2.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_cam2 = viewpoint_stack2.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        gt_image_og = viewpoint_cam2.original_image.cuda().detach()
        background = gt_image_og
        # background = torch.zeros_like(gt_image_og, device="cuda")

        bg = torch.rand((3), device="cuda") if opt.random_background else background


        # print(gaussians.get_xyz.shape)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # print(image.shape)
        # print(gt_image.shape)
        # img = Image.fromarray(viewpoint_cam.original_image.transpose(0,2).transpose(0,1).cpu().numpy() * 255, 'RGB')
        # img=Image.fromarray((gt_image.cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
        # img.show()
        # img.save('test.png')
        # img=Image.fromarray((image.detach().cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
        # img.save('image.png')
        # img=Image.fromarray((gt_image.cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
        # img.save('gt_image.png')
        # exit()
        # img = Image.fromarray(gt_image_og.transpose(0,2).transpose(0,1).cpu().numpy(), 'RGB')
        # img2 = Image.fromarray(image.transpose(0,2).transpose(0,1).cpu().detach().numpy(), 'RGB')
        # img.save('og.png')
        # img2.save('out.png')
        # exit()
        # print(image[:,0,0], image[:,400,800])
        # print(np.nonzero(image))
        # print(image == 0)
        # print(image.flatten(1)[:,0])
        # print(torch.isclose(image.flatten(1)[0,0],torch.tensor(0.)))
        # z = torch.tensor((0.,0.,0.)).cuda()
        # for i,x in enumerate(image.transpose(0,2)):
        #     for j,y in enumerate(x):
        #         # print(torch.isclose(y, z))
        #         if torch.all(torch.isclose(y, z)):
        #             image[:,i,j] = gt_image_og[:,i,j]
        #             print("here",i,j)

        # print(torch.isclose(image.flatten(1).transpose(0,1),bg))
        # print(torch.isclose(image.flatten(1).transpose(0,1),bg).shape)
        # print(torch.all(torch.isclose(image.flatten(1).transpose(0,1),bg),dim=1))
        # print(torch.all(torch.isclose(image.flatten(1).transpose(0,1),bg),dim=1).shape)
        # print(torch.all(torch.isclose(image.flatten(1).transpose(0,1),bg),dim=1).reshape(image.shape[1],-1).shape)
        # print(torch.all(torch.isclose(image.flatten(1).transpose(0,1),bg),dim=1).reshape(image.shape[1],-1)[None,:,:].repeat(3,1,1))
        # image[torch.where(torch.all(torch.isclose(image.flatten(1).transpose(0,1),bg),dim=1).reshape(image.shape[1],-1)[None,:,:].repeat(3,1,1))[0]] = gt_image_og.detach()
        # print(torch.all(torch.isclose(image.flatten(1).transpose(0,1),bg),dim=1).reshape(image.shape[1],-1)[None,:,:].repeat(3,1,1).shape)
        # print(torch.where(torch.all(torch.isclose(image.flatten(1).transpose(0,1),bg),dim=0))[0])
        # print(torch.where(image.flatten(1).isclose(torch.tensor(0.)).all(dim=1))[0])
        # image[torch.where(~image.any(axis=0)[0])] = gt_image_og.detach()
        # image[np.where] = gt_image_og.detach()
        # image2 = gt_image_og.detach()[np.nonzero(image)] = image
        # image = image2
        # image = torch.where(torch.all(torch.isclose(image.flatten(1).transpose(0,1),bg),dim=1).reshape(image.shape[1],-1)[None,:,:].repeat(3,1,1),gt_image_og,image+gt_image_og)
        # image += gt_image_og.detach()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Count": f"{gaussians.get_xyz.shape[0]}", "Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, pipe, (1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, scene2)
            # quit()
            if (iteration in saving_iterations):
                # img=Image.fromarray((image.detach().cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
                # img.save('image.png')
                # img=Image.fromarray((gt_image.detach().cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
                # img.save('gt_image.png')
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.output_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.output_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.output_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.output_path))
    os.makedirs(args.output_path, exist_ok = True)
    with open(os.path.join(args.output_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, pipe, renderArgs, train_test_exp, scene2):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'og' :[scene2.getTrainCameras()[idx % len(scene2.getTrainCameras())] for idx in range(0, 23, 1)],
                                    'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, 23, 1)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint_cam2 = config['og'][idx]
                    gt_image_og = viewpoint_cam2.original_image.cuda().detach()
                    # background = gt_image_og
                    # background = torch.zeros_like(gt_image_og, device="cuda")
                    background = backgrounds[idx]
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, pipe, background, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if idx == 0:
                        img=Image.fromarray((image.detach().cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
                        img.save(os.path.join(args.output_path,'image0.png'))
                        # img=Image.fromarray((gt_image.detach().cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
                        # img.save('gt_image.png')
                        # img=Image.fromarray((gt_image_og.detach().cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
                        # img.save('gt_image_og.png')
                    # img=Image.fromarray((image.detach().cpu().numpy().transpose((1,2,0)) * 255).astype(np.uint8), 'RGB')
                    # img.save(os.path.join(args.output_path,f'background{idx}.png'))
                    # torch.save(image.detach(), f'background{idx}.pt')
                    # exit()
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                global l1test
                global psnrtest
                l1test = l1_test.cpu().numpy()
                psnrtest = psnr_test.cpu().numpy()
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    args.densify_from_iter = 000
    args.densify_until_iter = 15_000
    args.iterations = 2_000
    args.output_path = './output/' + args.source_path
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.output_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # args2 = parser.parse_args(["-s", "./vangogh/vangogh0"])
    args2 = parser.parse_args(["-s", "./rpd/background"]) # rpda 29:44, distorted rpd0,16:48, undistorted 7:39
    args2.output_path = './rpd/background'
    # args2 = parser.parse_args(["-s", "./rpd/frame000029"])

    # model_path = './rpd0'
    video_path = './rpd'
    sub_paths = os.listdir(video_path)
    pattern = re.compile(r'frame(\d+)')
    frames = sorted(
        (item for item in sub_paths if pattern.match(item)),
        key=lambda x: int(pattern.match(x).group(1))
    )
    # frames=frames[args.frame_start:args.frame_end]
    # if args.frame_start==1:
    #     args.load_iteration = args.first_load_iteration
    gaussians2 = GaussianModel(lp.extract(args2).sh_degree, op.extract(args).optimizer_type)
    scene2 = Scene(lp.extract(args2), gaussians2, shuffle=False)
    load_iter = None
    l1_list = []
    psnr_list = []
    durations = []
    backgrounds = []
    # model_path = './output/rpd0 with a undistorted'

    # args = parser.parse_args(["-s", "./rpd/background"]) # rpda 29:44, distorted rpd0,16:48, undistorted 7:39
    # args.model_path = './output/rpda/'
    # load_iter = -1
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, scene2, load_iter=load_iter)
    # load_iter = -1

    # load in image from rpda
    for i in range(23):
        backgrounds.append(torch.load(f'./background/background{i}.pt'))

    for frame in frames:
        start_time = time.time()
        args.source_path = os.path.join(video_path, frame)
        # args.output_path = os.path.join(output_path, frame)
        # args.model_path = model_path
        args.output_path = './output/' + args.source_path
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, scene2, load_iter=load_iter)
        # load_iter = -1
        # model_path = args.output_path
        duration = time.time()-start_time
        print(f"Frame {frame} finished in {duration} seconds.")
        l1_list.append(l1test)
        psnr_list.append(psnrtest)
        durations.append(duration)
        torch.cuda.empty_cache()
        # if count == 1:
        #     break
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, lp.extract(args2))
    with open('results.txt', 'w') as f:
        for i in range(len(durations)):
            f.write(f"{l1_list[i]} {psnr_list[i]} {durations[i]}\n")
    plt.plot(l1_list)
    plt.savefig('l1.png')
    plt.close()
    plt.plot(psnr_list)
    plt.savefig('psnr.png')
    plt.close()
    plt.plot(durations)
    plt.savefig('duration.png')
    plt.close()

    # All done
    print("\nTraining complete.")
