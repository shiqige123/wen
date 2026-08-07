import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import lpips
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import tensorflow as tf

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):

    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d

        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape

    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])

    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    all_ret = batchify_rays(rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs,
                gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf
    loss_fn = lpips.LPIPS(net='alex').to(device).float()
    print("LPIPS model initialized.")

    if render_factor != 0:
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    t = time.time()

    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()

        rgb, disp, acc, _ = render(
            H, W, K,
            chunk=chunk,
            c2w=c2w[:3, :4],
            **render_kwargs
        )

        rgb_np = rgb.cpu().numpy()

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            gt_img_np = gt_imgs[i].cpu().numpy() if isinstance(gt_imgs[i], torch.Tensor) else gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img_np)))
            psnr_scores.append(p)
            print(f"PSNR score for image {i}：", p)

        if gt_imgs is not None:
            gt_rgb = gt_imgs[i]

            if isinstance(gt_rgb, torch.Tensor):
                gt_rgb = gt_rgb.cpu().numpy()

            print(f"Rendered image shape: {rgb_np.shape}, Ground truth shape: {gt_rgb.shape}")

            rendered_rgb = torch.from_numpy(rgb_np).unsqueeze(0).permute(0, 3, 1, 2).to(device).float()
            gt_rgb_tensor = torch.from_numpy(gt_rgb).unsqueeze(0).permute(0, 3, 1, 2).to(device).float()

            lpips_score = loss_fn(rendered_rgb, gt_rgb_tensor).item()
            lpips_scores.append(lpips_score)
            print(f"LPIPS score for image {i}: {lpips_score}")

            rendered_gray = np.mean(rgb_np, axis=-1)
            gt_gray = np.mean(gt_rgb, axis=-1)

            ssim_score = ssim(rendered_gray, gt_gray, data_range=1.0)
            ssim_scores.append(ssim_score)
            print(f"SSIM score for image {i}: {ssim_score}")

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    if gt_imgs is not None:
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        avg_lpips = np.mean(lpips_scores)

        print(f"平均 PSNR 值为: {avg_psnr}")
        print(f"平均 SSIM 得分: {avg_ssim}")
        print(f"平均 LPIPS 得分: {avg_lpips}")

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs
    ).to(device)

    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs
        ).to(device)

        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk
    )

    optimizer = torch.optim.Adam(
        params=grad_vars,
        lr=args.lrate,
        betas=(0.9, 0.999)
    )

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if 'tar' in f
        ]

    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)

        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        model.load_state_dict(ckpt['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d,
                raw_noise_std=0,
                white_bkgd=False,
                pytest=False):

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)],
        -1
    )

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])

    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)

    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10],
            -1
        ),
        -1
    )[:, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)

    disp_map = 1. / torch.max(
        1e-10 * torch.ones_like(depth_map),
        depth_map / torch.sum(weights, -1)
    )

    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):

    N_rays = ray_batch.shape[0]

    rays_o = ray_batch[:, 0:3]
    rays_d = ray_batch[:, 3:6]

    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near = bounds[..., 0]
    far = bounds[..., 1]

    t_vals = torch.linspace(0., 1., steps=N_samples)

    if not lindisp:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)

        t_rand = torch.rand(z_vals.shape)

        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    raw = network_query_fn(pts, viewdirs, network_fn)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw,
        z_vals,
        rays_d,
        raw_noise_std,
        white_bkgd,
        pytest=pytest
    )

    if N_importance > 0:
        rgb_map_0 = rgb_map
        disp_map_0 = disp_map
        acc_map_0 = acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])

        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            N_importance,
            det=(perturb == 0.),
            pytest=pytest
        )

        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine

        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw,
            z_vals,
            rays_d,
            raw_noise_std,
            white_bkgd,
            pytest=pytest
        )

    ret = {
        'rgb_map': rgb_map,
        'disp_map': disp_map,
        'acc_map': acc_map
    }

    if retraw:
        ret['raw'] = raw

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret
def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_llff_data/fern', help='input data directory')

    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')

    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4, help='batch size')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay')

    parser.add_argument("--chunk", type=int, default=1024 * 32, help='number of rays processed in parallel')
    parser.add_argument("--netchunk", type=int, default=1024 * 64, help='number of pts sent through network in parallel')

    parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights path')

    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')

    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for view directions')

    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to sigma output')

    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render')
    parser.add_argument("--render_test", action='store_true', help='render the test set')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering')

    parser.add_argument("--precrop_iters", type=int, default=0, help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float, default=.5, help='fraction of img taken for central crops')

    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, help='load 1/N images from test/val sets')

    parser.add_argument("--shape", type=str, default='greek', help='options : armchair / cube / greek / vase')

    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd')
    parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400')

    parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, help='take every 1/N images as LLFF test set')

    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000, help='frequency of render_poses video saving')

    return parser


from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()

    min_side = min(img1.shape[:2])
    win_size = min(7, min_side)
    data_range = 1.0

    ssim_value, _ = ssim(
        img1,
        img2,
        multichannel=True,
        full=True,
        win_size=win_size,
        data_range=data_range
    )

    return ssim_value
