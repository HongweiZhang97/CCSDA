from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
	Args:
	  x: pytorch Variable
	Returns:
	  x: pytorch Variable, same shape as input
	"""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	Returns:
	  dist: pytorch Variable, with shape [m, n]
	"""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	"""
    x_normed = F.normalize(x, p=2, dim=1)
    y_normed = F.normalize(y, p=2, dim=1)
    return 1 - torch.mm(x_normed, y_normed.t())


def cosine_similarity(x, y):
    """
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	"""
    x_normed = F.normalize(x, p=2, dim=1)
    y_normed = F.normalize(y, p=2, dim=1)
    return torch.mm(x_normed, y_normed.t())


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
	Args:
	  dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
	  labels: pytorch LongTensor, with shape [N]
	  return_inds: whether to return the indices. Save time if `False`(?)
	Returns:
	  dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
	  dist_an: pytorch Variable, distance(anchor, negative); shape [N]
	  p_inds: pytorch LongTensor, with shape [N];
		indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
	  n_inds: pytorch LongTensor, with shape [N];
		indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
	NOTE: Only consider the case in which all labels have same num of samples,
	  thus we can cope with all anchors in parallel.
	"""
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def adaptive_margin_loss(dist_ap, dist_an, adaptive_margin):
    diff = torch.add(torch.sub(dist_ap, dist_an), adaptive_margin).unsqueeze(1)
    zeros = dist_an.new().resize_as_(dist_an).fill_(0.0).unsqueeze(1)
    diff_zeros = torch.cat((diff, zeros), 1)
    loss_value = torch.max(diff_zeros, 1).values
    loss = loss_value.mean()
    return loss


# ==============
#  Triplet Loss
# ==============
class TripletHardLoss(object):

    def __init__(self, margin=None, metric="euclidean"):
        self.margin = margin
        self.metric = metric
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False, alpha=None, belta=None):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        if self.metric == "euclidean":
            dist_mat = euclidean_dist(global_feat, global_feat)
        elif self.metric == "cosine":
            dist_mat = cosine_dist(global_feat, global_feat)
        else:
            raise NameError

        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss



def distance_mining(dist_mat, labels, cameras):
    # check dist_mat
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)

    # get dist_mat W
    N = dist_mat.size(0)

    # labels dist
    # get positive index mat
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # & cameras.expand(N,N).eq(cameras.expand(N,N).t())
    # get negative index mat
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())  # | cameras.expand(N,N).ne(cameras.expand(N,N).t())
    # get hard positive dist, shape bs * 1
    d1 = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]
    d1 = d1.squeeze(1)
    # dist_neg=dist_mat[is_neg].contiguous().view(N,-1)
    d2 = d1.new().resize_as_(d1).fill_(0)
    d3 = d1.new().resize_as_(d1).fill_(0)
    d2ind = []
    for i in range(N):
        sorted_tensor, sorted_index = torch.sort(dist_mat[i])
        cam_id = cameras[i]
        B, C = False, False
        for ind in sorted_index:
            if labels[ind] == labels[i]:
                continue
            if B == False and cam_id == cameras[ind]:
                d3[i] = dist_mat[i][ind]
                B = True
            if C == False and cam_id != cameras[ind]:
                d2[i] = dist_mat[i][ind]
                C = True
                d2ind.append(ind)
            if B and C:
                break

    # return d1, d2, d3, d2ind
    return d1, d2, d3

class DistanceLoss(torch.nn.Module):
    """Multi-camera negative loss
        In a mini-batch,
       d1=(A,A'), A' is the hardest true positive.
       d2=(A,C), C is the hardest negative in another camera.
       d3=(A,B), B is the hardest negative in same camera as A.
       let d1<d2<d3
    """
    def __init__(self, loader=None, margin=None):
        super(DistanceLoss, self).__init__()
        print("dist loss")
        self.margin = margin
        self.texture_loader = loader
        if margin is not None:
            self.ranking_loss1 = nn.MarginRankingLoss(margin=margin[0], reduction="mean")
            self.ranking_loss2 = nn.MarginRankingLoss(margin=margin[1], reduction="mean")
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction="mean")

    def forward(self, feat, labels, cameras, model=None, paths=None, epoch=0, normalize_feature=False):
        if normalize_feature:  # default: don't normalize , distance [0,1]
            feat = normalize(feat, axis=-1)
        dist_mat = euclidean_dist(feat, feat)
        d1, d2, d3 = distance_mining(dist_mat, labels, cameras)

        y = d1.new().resize_as_(d1).fill_(1)
        if self.margin is not None:
            l1 = self.ranking_loss1(d2, d1, y)
            l2 = self.ranking_loss2(d3, d2, y)
        else:
            l1 = self.ranking_loss(d2-d1, y)
            l2 = self.ranking_loss(d3-d2, y)
        loss = l2+l1
        # loss = l1
        accuracy1 = torch.mean((d1 < d2).float())
        accuracy2 = torch.mean((d2 < d3).float())
        # return loss, accuracy1, accuracy2
        return loss, accuracy1, accuracy2
