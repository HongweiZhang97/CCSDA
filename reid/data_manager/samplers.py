from __future__ import absolute_import

from collections import defaultdict

import numpy as np
import torch

from torch.utils.data.sampler import Sampler
import copy
import random


class IdentitySampler(Sampler):
    def __init__(self,  data_source, num_instances=8, batch_size=256):
        print("using IS")
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances  # approximate
        self.index_dic = defaultdict(list)
        for index, (_, pid, camid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        copy_pids = copy.deepcopy(self.pids)
        for x in range(self.__len__()):
            identities = []
            ret = []
            selected_size = self.num_pids_per_batch
            replace = False if len(copy_pids) >= selected_size else True
            selected_pid = np.random.choice(copy_pids, size=selected_size, replace=replace)
            for pid in selected_pid:
                inds = self.index_dic[pid]
                replace = False if len(inds) >= self.num_instances else True
                inds = np.random.choice(inds, size=self.num_instances, replace=replace)
                ret.extend(inds)
                copy_pids.remove(pid)
                if len(ret) == self.batch_size:
                    break
            assert len(ret) == self.batch_size
            yield ret

    def __len__(self):
        return self.num_identities*8 // self.batch_size


class RandomIdentityCameraSampler(Sampler):

    def __init__(self, data_source, num_instances=8, batch_size=256):
        print("using RISC")
        self.data_source=data_source  # 源数据
        self.num_instances=num_instances  # 每个身份采样个数
        self.pid_dic=defaultdict(list)  # 身份id字典
        self.camera_dic=defaultdict(list)  # 相机id字典
        self.pid_cid_dic=defaultdict(list)  # 相机_身份id字典
        pid_cam=defaultdict(set)

        for index, (_, pid, cid, _, _) in enumerate(data_source):
            self.pid_dic[pid].append(index)  # 记录id索引
            self.pid_cid_dic[(pid, cid)].append(index)  # 记录身份相机id对索引
            pid_cam[pid].add(cid)
            if not pid in self.camera_dic[cid]:
                self.camera_dic[cid].append(pid)

        self.pids=list(self.pid_dic.keys())
        self.num_identities=len(self.pids)
        self.cids=list(self.camera_dic.keys())
        self.num_cameras=len(self.cids)
        self.batch_size=batch_size
        self.identities_per_batch=batch_size//num_instances
        self.identities_per_cam=self.identities_per_batch//min(8, self.num_cameras)
        l=0
        print(len(self.pid_dic))
        for p in pid_cam:
            l+=len(pid_cam[p])
        print("Camera/Person Value:{}".format(l/self.num_identities))

    def __len__(self):
        return self.num_identities*8 // self.batch_size

    def __iter__(self):
        for x in range(self.__len__()):
            cameras = torch.randperm(self.num_cameras)
            identities = []
            ret = []
            for c in cameras:
                cid = self.cids[c]
                t = self.camera_dic[cid]
                if len(t) < self.identities_per_cam:
                    continue
                replace = False if len(t) >= self.identities_per_cam else True
                t = np.random.choice(t, size=self.identities_per_cam, replace=replace)
                for pid in t:
                    inds = self.pid_cid_dic[(pid, cid)]
                    # inds = self.pid_dic[pid] #self.pid_cid_dic[(pid,cid)]
                    replace = False if len(inds)>=self.num_instances else True
                    inds = np.random.choice(inds,size=self.num_instances,replace=replace)
                    ret.extend(inds)
                if len(ret) == self.batch_size:
                    break
            assert len(ret) == self.batch_size
            yield ret


class RandomIdentityUniqueCameraSampler(Sampler):

    def __init__(self, data_source, num_instances=8, batch_size=256):
        print("using RIUCS")
        self.data_source=data_source  # 源数据
        self.num_instances=num_instances  # 每个身份采样个数
        self.pid_dic=defaultdict(list)  # 身份id字典
        self.camera_dic=defaultdict(list)  # 相机id字典
        self.pid_cid_dic=defaultdict(list)  # 相机_身份id字典
        pid_cam=defaultdict(set)

        for index, (_, pid, cid, _, _) in enumerate(data_source):
            self.pid_dic[pid].append(index)  # 记录id索引
            self.pid_cid_dic[(pid, cid)].append(index)  # 记录身份相机id对索引
            pid_cam[pid].add(cid)
            if not pid in self.camera_dic[cid]:
                self.camera_dic[cid].append(pid)

        self.pids=list(self.pid_dic.keys())
        self.num_identities=len(self.pids)
        self.cids=list(self.camera_dic.keys())
        self.num_cameras=len(self.cids)
        self.batch_size=batch_size
        self.identities_per_batch=batch_size//num_instances
        self.identities_per_cam=self.identities_per_batch//min(8, self.num_cameras)
        l=0
        print(len(self.pid_dic))
        for p in pid_cam:
            l+=len(pid_cam[p])
        print("Camera/Person Value:{}".format(l/self.num_identities))

    def __len__(self):
        return self.num_identities*8 // self.batch_size

    def __iter__(self):
        copy_pids = copy.deepcopy(self.camera_dic[0])
        for x in range(self.__len__()):
            cameras = torch.randperm(self.num_cameras)
            ret = []
            # print("info")
            # print("pid_list", len(pid_list))
            selected_size = self.identities_per_cam*self.num_cameras
            replace = False if len(copy_pids) >= selected_size else True
            selected_pid = np.random.choice(copy_pids, size=selected_size, replace=replace)
            # print("selected_pid", len(selected_pid))
            for idx in range(len(cameras)):
                cid = self.cids[int(cameras[idx])]
                t = selected_pid[(idx * self.identities_per_cam):(idx * self.identities_per_cam + self.identities_per_cam)]
                # print("t", len(t))
                for pid in t:
                    inds = self.pid_cid_dic[(pid, cid)]
                    replace = False if len(inds) >= self.num_instances else True
                    inds = np.random.choice(inds, size=self.num_instances, replace=replace)
                    ret.extend(inds)
                    # copy_pids.remove(pid)
                if len(ret) == self.batch_size:
                    break
            assert len(ret) == self.batch_size
            # assert 0 > 1
            yield ret


class NormalCollateFn():
    def __call__(self, batch):
        N = len(batch)
        img_tensor = [x[0] for x in batch]
        pids = np.array([x[1] for x in batch])
        camids = np.array([x[2] for x in batch])
        if_real = np.array([x[3] for x in batch])
        fake_camids = np.array([x[4] for x in batch])

        return torch.stack(img_tensor, dim=0), torch.from_numpy(np.array(pids))\
            , torch.from_numpy(np.array(camids)), torch.from_numpy(np.array(if_real))\
            , torch.from_numpy(np.array(fake_camids))# ,paths
