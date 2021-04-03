from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path as osp


import os
import sys
import random
import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# from config import opt
# from io_stream import data_manager
from reid import data_manager



from reid.models.cbn_model import CbnNetBuilder
from reid.evaluating import evaluator_manager

from reid.utils.serialization import Logger, load_previous_model
from reid.utils.transforms import TestTransform

CUDA_VISIBLE_DEVICES = 0
# torch.cuda.set_device(0)


def get_data(name, data_dir, num_bn_sample):
    dataset = None
    # Datasets
    if name == 'market1501':
        dataset_name = 'market1501'
        dataset = data_manager.init_imgreid_dataset(
            root=data_dir, name=dataset_name
        )
        dataset.images_dir = ''
    elif name == 'market_sct' or name == 'market_sct_tran' or name == 'duke_sct' or name == 'duke_sct_tran':
        dataset_name = name
        dataset = data_manager.init_imgreid_dataset(
            root=data_dir, name=dataset_name, num_bn_sample=num_bn_sample
        )
        dataset.images_dir = ''
    assert dataset is not None, "dataset is none"
    return dataset


def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    # check dir
    assert os.path.exists(args.logs_dir), "logs_dir not exists"
    # os.makedirs(args.logs_dir)
    args.logs_file = args.logs_dir + '/eval.txt'
    assert args.logs_file is not None
    sys.stdout = Logger(args.logs_file)

    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # check gpu
    use_gpu = torch.cuda.is_available()
    pin_memory = True if use_gpu else False

    # prepare data
    print('initializing dataset {}'.format(args.dataset))
    dataset = get_data(args.dataset, args.data_dir, num_bn_sample=args.batch_num_bn_estimatation * args.batch_size)

    # prepare model
    model = CbnNetBuilder(last_stride=args.last_stride)
    for i in range(args.epochs // args.evaluate_interval):
        if ((i + 1) * args.evaluate_interval) < args.start_save:
            continue
        checkpoint_path = args.logs_dir + "/checkpoint_" \
                          + str((i + 1) * args.evaluate_interval) + ".pth.tar"
        print("checkpoint path", checkpoint_path)
        print('loading model from {} ...'.format(checkpoint_path))
        model = load_previous_model(model, checkpoint_path, load_fc_layers=False)
        model.eval()

        # if use_gpu:
        #     model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        reid_evaluator = evaluator_manager.init_evaluator(args.dataset, model, flip=True)

        def _calculate_bn_and_features(all_data, sampled_data):
            time.sleep(1)
            all_features, all_ids, all_cams = [], [], []
            available_cams = list(sampled_data)

            for current_cam in tqdm.tqdm(available_cams):
                camera_samples = sampled_data[current_cam]
                data_for_camera_loader = DataLoader(
                    data_manager.init_datafolder(args.dataset, camera_samples, TestTransform(args.height, args.width)),
                    batch_size=args.batch_size, num_workers=args.workers,
                    pin_memory=False, drop_last=True
                )
                reid_evaluator.collect_sim_bn_info(data_for_camera_loader)

                camera_data = all_data[current_cam]
                data_loader = DataLoader(
                    data_manager.init_datafolder(args.dataset, camera_data, TestTransform(args.height, args.width)),
                    batch_size=args.batch_size, num_workers=args.workers,
                    pin_memory=pin_memory, shuffle=False
                )
                fs, pids, camids = reid_evaluator.produce_features(data_loader, normalize=True)
                all_features.append(fs)
                all_ids.append(pids)
                all_cams.append(camids)

            all_features = torch.cat(all_features, 0)
            all_ids = np.concatenate(all_ids, axis=0)
            all_cams = np.concatenate(all_cams, axis=0)
            time.sleep(1)
            return all_features, all_ids, all_cams

        print('Processing query features...')
        qf, q_pids, q_camids = _calculate_bn_and_features(dataset.query_per_cam, dataset.query_per_cam_sampled)
        print('Processing gallery features...')
        gf, g_pids, g_camids = _calculate_bn_and_features(dataset.gallery_per_cam,
                                                          dataset.gallery_per_cam_sampled)
        print('Computing CMC and mAP...')
        reid_evaluator.get_final_results_with_features(qf, q_pids, q_camids, gf, g_pids, g_camids)


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    parser = argparse.ArgumentParser(description="sct cbn aug")
    # data
    parser.add_argument('-d', '--dataset', type=str, default=None)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--batch_num_bn_estimatation', type=int, default=50)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('--last_stride', type=int, default=1)
    # loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin of the triplet loss, default: 0.3")
    parser.add_argument('--margin1', type=float, default=0.1,
                        help="margin of the triplet loss, default: 0.1")
    parser.add_argument('--margin2', type=float, default=0.1,
                        help="margin of the triplet loss, default: 0.1")
    # optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--gpu_ids', type=str, default="0")
    parser.add_argument('--num_gpu', type=int, default=2)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument('--evaluate_interval', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_save', type=int, default=10,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=16)
    # metric learning
    # parser.add_argument('--dist-metric', type=str, default='euclidean',
    #                     choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='./data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=None)
    parser.add_argument('--logs-file', type=str, metavar='PATH',
                        default=None)
    test(parser.parse_args())

