from __future__ import print_function, absolute_import
import time
import sys
import os

import torch
# from apex import amp
from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils.data.transforms import RandomErasing
from .loss.loss_set import TripletHardLoss, DistanceLoss


class BaseTrainer(object):
    def __init__(self, opt, model, optimzier, criterion, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer = optimzier
        self.criterion = criterion
        self.summary_writer = summary_writer
        self.global_step = 0

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()
        for i, inputs in enumerate(data_loader):

            data_time.update(time.time() - start)
            # model optimizer
            self._parse_data(inputs)
            self._forward(epoch)

            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())

            # tensorboard
            self.global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('loss', self.loss.item(), self.global_step)
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            start = time.time()

            if (i + 1) % 1 == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.mean, batch_time.val,
                              data_time.mean, data_time.val,
                              losses.mean, losses.val))

        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, data):
        raise NotImplementedError

    def _backward(self):
        raise NotImplementedError


class CbnImgTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer):
        super().__init__(opt, model, optimizer, criterion, summary_writer)

    def _parse_data(self, inputs):
        # print("inputs", len(inputs))
        imgs, pids, camids, if_real, fake_camid = inputs
        # print("pids", pids)
        self.data = imgs.cuda()
        self.pids = pids.cuda()
        self.camids = camids.cuda()
        self.if_real = if_real.cuda()
        self.fake_camids = fake_camid.cuda()

    def _organize_data(self):
        # 存在camid不匹配的bug，对cbn效果有影响，具体效果待检验
        unique_camids = torch.unique(self.camids).cpu().numpy()
        reorg_data = []
        reorg_pids = []
        reorg_camids = []
        reorg_if_real = []
        reorg_fake_camids = []
        for current_camid in unique_camids:
            current_camid = (self.camids == current_camid).nonzero().view(-1)
            if current_camid.size(0) > 1:
                data = torch.index_select(self.data, index=current_camid, dim=0)
                pids = torch.index_select(self.pids, index=current_camid, dim=0)
                camids = torch.index_select(self.camids, index=current_camid, dim=0)
                if_real = torch.index_select(self.if_real, index=current_camid, dim=0)
                fake_camids = torch.index_select(self.fake_camids, index=current_camid, dim=0)
                reorg_data.append(data)
                reorg_pids.append(pids)
                reorg_camids.append(camids)
                reorg_if_real.append(if_real)
                reorg_fake_camids.append(fake_camids)

        # Sort the list for our modified data-parallel
        # This process helps to increase efficiency when utilizing multiple GPUs
        # However, our experiments show that this process slightly decreases the final performance
        # You can enable the following process if you prefer
        # sort_index = [x.size(0) for x in reorg_pids]
        # sort_index = [i[0] for i in sorted(enumerate(sort_index), key=lambda x: x[1], reverse=True)]
        # reorg_data = [reorg_data[i] for i in sort_index]
        # reorg_pids = [reorg_pids[i] for i in sort_index]
        # ===== The end of the sort process ==== #
        self.data = reorg_data
        self.pids = reorg_pids
        self.camids = reorg_camids
        self.if_real = reorg_if_real
        self.fake_camids = reorg_camids

    def _forward(self, data):
        feat, id_scores = self.model(data)
        return feat, id_scores

    def _backward(self):
#         with amp.scale_loss(self.loss, self.optimizer) as scaled_loss:
#             scaled_loss.backward()
        self.loss.backward()

    def train(self, epoch, data_loader):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        iden_acc = AverageMeter()
        dist_acc1 = AverageMeter()
        dist_acc2 = AverageMeter()
        for i, inputs in enumerate(data_loader):
            # print("inputs0", len(inputs))
            self._parse_data(inputs)
            self._organize_data()
            torch.cuda.synchronize()
            tic = time.time()
            feat, id_scores = self._forward(self.data)

            pids = torch.cat(self.pids, dim=0).long()
            camids = torch.cat(self.camids, dim=0)
            if_real = torch.cat(self.if_real, dim=0)
            fake_camids = torch.cat(self.fake_camids, dim=0)

            loss_fun_data = (feat, id_scores, pids, camids, if_real, fake_camids)
            # print("size:", feat.size(), id_scores.size(), pids.size(), camids.size())
            self.loss, id_acc, acc1, acc2= self.criterion(loss_fun_data, epoch, self.global_step, self.summary_writer)
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            torch.cuda.synchronize()

            batch_time.update(time.time() - tic)
            losses.update(self.loss.item())
            iden_acc.update(float(id_acc))
            dist_acc1.update(float(acc1))
            dist_acc2.update(float(acc2))
            # tensorboard
            self.global_step = epoch * len(data_loader) + i

            if (i + 1) % 1 == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'iden_acc {:.2%} ({:.2%})\t'
                      'dist_acc1 {:.2%} ({:.2%})\t'
                      'dist_acc2 {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.mean, batch_time.val,
                              losses.mean, losses.val,
                              iden_acc.mean, iden_acc.val,
                             dist_acc1.mean, dist_acc1.val,
                             dist_acc2.mean, dist_acc2.val))
            # torch.cuda.empty_cache()
            # assert 0 > 1

        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
