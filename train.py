# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
from mmengine.config import Config
from mmengine.runner import Runner, find_latest_checkpoint
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from flag3d.utils import register_all_modules
# import pydevd_pycharm
# pydevd_pycharm.settrace('10.8.32.196', port=16666, stdoutToServer=True, stderrToServer=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--dump',
        type=str,
        default='results.pkl',
        help='dump predictions to a pickle file')
    parser.add_argument(
        '--sync-bn',
        action='store_true',
        help='whether to use sync bn.')
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='whether to disable the training process.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
             'actual batch size and the original batch size.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./outputs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    dump_metric = dict(type='DumpResults',
                       out_file_path=osp.join(cfg.work_dir, args.dump))
    cfg.test_evaluator = list(cfg.test_evaluator)
    cfg.test_evaluator.append(dump_metric)

    if args.sync_bn:
        cfg.sync_bn = 'torch'

    # Reproducible
    cfg.randomness = dict(seed=0, diff_rank_seed=False, deterministic=True)

    return cfg


def main():
    args = parse_args()

    register_all_modules()

    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    if not args.no_train:
        # start training
        runner.train()

    # load from best_ckpt
    best_ckpt = glob.glob(osp.join(cfg.work_dir, f'best*/*.pth'))[0]
    runner.load_checkpoint(best_ckpt)

    # start testing the best checkpoint
    runner.set_randomness(**cfg.randomness)
    runner.test()

    os.system(f'mv {osp.join(cfg.work_dir, "results.pkl")} '
              f'{osp.join(cfg.work_dir, "best_results.pkl")}')

    # load from last_ckpt
    last_ckpt = find_latest_checkpoint(cfg.work_dir)
    runner.load_checkpoint(last_ckpt)

    # start testing the last checkpoint
    runner.set_randomness(**cfg.randomness)
    runner.test()

    os.system(f'mv {osp.join(cfg.work_dir, "results.pkl")} '
              f'{osp.join(cfg.work_dir, "last_results.pkl")}')


if __name__ == '__main__':
    main()
