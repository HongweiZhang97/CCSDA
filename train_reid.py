from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import random

import numpy as np
import sys
# from apex import amp
# from apex.parallel import DistributedDataParallel

sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime
from save_log import Logger

from reid import data_manager
from reid.data_manager.samplers import RandomIdentityUniqueCameraSampler,\
    NormalCollateFn, RandomIdentityCameraSampler
from reid.cbn_img_trainers import CbnImgTrainer

from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.lr_scheduler import LRScheduler
from reid.models.cbn_model import CbnNetBuilder
from reid.models.models_utils.cam_data_parallel import CamDataParallel
from reid.loss.loss_set import TripletHardLoss, DistanceLoss
from reid.utils.transforms import TrainTransform, TestTransform




def get_data(name, data_dir, height, width, batch_size, num_bn_sample, num_instances,
             workers):
    # Datasets
    if name == 'market1501':
        dataset_name = 'market1501'
        dataset = data_manager.init_imgreid_dataset(
            root=data_dir, name=dataset_name
        )
        dataset.images_dir = ''
        train_transformer = T.Compose([
            T.Random2DTranslation(height, width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

        test_transformer = T.Compose([
            T.RectScale(height, width),
            T.ToTensor(),
        ])
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.images_dir, transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=RandomIdentitySampler(dataset.train, num_instances),
            pin_memory=True, drop_last=True)

    elif name == 'market_sct' or name == 'market_sct_tran' or name == 'duke_sct' or name == 'duke_sct_tran':
        dataset_name = name
        dataset = data_manager.init_imgreid_dataset(
            root=data_dir, name=dataset_name, num_bn_sample=num_bn_sample
        )
        dataset.images_dir = ''
        # dataset_name = name
        pin_memory = True
        collateFn = NormalCollateFn()
        train_loader = DataLoader(
            data_manager.init_datafolder(dataset_name, dataset.train,
                                         TrainTransform(height, width),
                                         ),
            batch_sampler=RandomIdentityCameraSampler(dataset.train, num_instances, batch_size),
            num_workers=workers,
            pin_memory=pin_memory, collate_fn=collateFn,
        )
        # train_loader = DataLoader(
        #     data_manager.init_datafolder(dataset_name, dataset.train,
        #                                  TrainTransform(height, width),
        #                                  ),
        #     batch_sampler=RandomIdentityUniqueCameraSampler(dataset.train, num_instances, batch_size),
        #     num_workers=workers,
        #     pin_memory=pin_memory, collate_fn=collateFn,
        # )

    # Num. of training IDs
    num_classes = dataset.num_train_pids
    return dataset, num_classes, train_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    # check logdir
    assert not os.path.exists(args.logs_dir)
    os.makedirs(args.logs_dir)
    print("create log dir")
    args.logs_file = args.logs_dir + '/train_log.txt'
    assert args.logs_file is not None
    sys.stdout = Logger(args.logs_file)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # prepare summary_writer
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    summary_writer = SummaryWriter(osp.join(args.logs_dir, 'tensorboard_log' + TIMESTAMP))

    # check args
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    assert args.height is not None and args.width is not None, "input size is none"

    # prepare data
    dataset, num_classes, train_loader = \
        get_data(args.dataset, args.data_dir, args.height,
                 args.width, args.batch_size,
                 args.batch_num_bn_estimatation * args.batch_size,
                 args.num_instances, args.workers,
                 )

    # Create model
    print("using cbn model")
    modelPath = 'reid/weights/pre_train/resnet50-19c8e357.pth'
    model = CbnNetBuilder(num_pids=num_classes, last_stride=args.last_stride, model_path=modelPath)
    model = CamDataParallel(model).cuda()

    # Optimizer
    param_groups = model.parameters()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NameError

    lr_scheduler = LRScheduler(base_lr=0.0002, step=[100, 200],
                               factor=0.001, warmup_epoch=20,
                               warmup_begin_lr=0.0002)

    # # loss criterion
    xent = nn.CrossEntropyLoss().cuda()
    distLoss = DistanceLoss(margin=(args.margin1, args.margin2)).cuda()
    # tripletLoss = TripletHardLoss(margin=args.margin)

    def loss_critertion(loss_fun_data, epoch, global_step, summary_writer):
        feat, id_scores, pids, camids, if_real, fake_camids = loss_fun_data
        id_loss = xent(id_scores, pids)
        id_acc = torch.mean((torch.argmax(id_scores, dim=1) == pids).float())
        dist_loss, acc1, acc2 = distLoss(feat, pids, camids, epoch=epoch)
        loss = id_loss + dist_loss
        # tri_loss = (feat, pids)

        # loss = id_loss + dist_loss  # + tri_loss
        summary_writer.add_scalar('loss', loss.item(), global_step)
        summary_writer.add_scalar('id_loss', id_loss.item(), global_step)
        summary_writer.add_scalar('dist_loss', dist_loss.item(), global_step)
        # summary_writer.add_scalar('tri_loss', tri_loss.item(), global_step)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        return loss, id_acc, acc1, acc2

    # Trainer
    print("using cbn imgtrainer!")
    trainer = CbnImgTrainer(args, model, optimizer, loss_critertion, summary_writer)
    start_epoch = 0

    # Start training
    for epoch in range(start_epoch, args.epochs):
        lr = lr_scheduler.update(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('[Info] Epoch [{}] learning rate update to {:.3e}'.format(epoch, lr))
        trainer.train(epoch, train_loader)
        if (epoch + 1) % args.evaluate_interval == 0 and (epoch + 1) >= args.start_save and (epoch + 1) < args.end_save:
            is_best = False
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict()
            }, epoch + 1, is_best, save_interval=1, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))


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
    parser.add_argument('-b', '--batch-size', type=int, default=240)
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 8")
    parser.add_argument('--batch_num_bn_estimatation', type=int, default=50)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--sampler', type=str, default='RICS')
    # model
    parser.add_argument('--last_stride', type=int, default=1)
    # loss
    parser.add_argument('--scala_ce', type=float, default=50.0,
                        help="default: 30.0")
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
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument('--evaluate_interval', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_save', type=int, default=10,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--end_save', type=int, default=210,
                        help="end saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=16)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='./data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=None)
    parser.add_argument('--logs-file', type=str, metavar='PATH',
                        default=None)
    main(parser.parse_args())
