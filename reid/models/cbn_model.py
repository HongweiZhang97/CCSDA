from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from getpass import getuser

from torch import nn
import torch.nn.functional as F

from reid.models.models_utils.resnet_cbn import ResNet

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class CbnNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_pids=None, last_stride=1, model_path=None):
        super().__init__()
        self.num_pids = num_pids
        self.base = ResNet(last_stride)
#         model_path = './frameworks/models/resnet50-19c8e357.pth'
        if model_path is not None:
            self.base.load_param(model_path)
        bn_neck = nn.BatchNorm1d(2048, momentum=None)
        bn_neck.bias.requires_grad_(False)
        self.bottleneck = nn.Sequential(bn_neck)
        self.bottleneck.apply(weights_init_kaiming)
        if self.num_pids is not None:
            self.classifier = nn.Linear(2048, self.num_pids, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat_before_bn = self.base(x)
        feat_before_bn = F.avg_pool2d(feat_before_bn, feat_before_bn.shape[2:])
        feat_before_bn = feat_before_bn.view(feat_before_bn.shape[0], -1)
        feat_after_bn = self.bottleneck(feat_before_bn)
        if self.num_pids is not None:
            classification_results = self.classifier(feat_after_bn)
            return feat_after_bn, classification_results
        else:
            return feat_after_bn

    def get_optim_policy(self):
        base_param_group = filter(lambda p: p.requires_grad, self.base.parameters())
        add_param_group = filter(lambda p: p.requires_grad, self.bottleneck.parameters())
        cls_param_group = filter(lambda p: p.requires_grad, self.classifier.parameters())

        all_param_groups = []
        all_param_groups.append({'params': base_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': add_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': cls_param_group, "weight_decay": 0.0005})
        return all_param_groups
