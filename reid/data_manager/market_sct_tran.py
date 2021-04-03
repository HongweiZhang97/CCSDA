import glob
import re
from os import path as osp
from .camera_utils import reorganize_images_by_camera


class MarketSCTTran(object):
    dataset_dir = 'market_sct_tran'

    def __init__(self, root='data',pidoffset=0,cidoffset=0, **kwargs):
        self.name = 'market_sct_tran'
        self.pidoffset=pidoffset
        self.cidoffset=cidoffset
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train_sct')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Augment Market-SCT loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_cids=6
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        self.train_per_cam, self.train_per_cam_sampled = reorganize_images_by_camera(self.train,
                                                                                     kwargs['num_bn_sample'])
        self.query_per_cam, self.query_per_cam_sampled = reorganize_images_by_camera(self.query,
                                                                                     kwargs['num_bn_sample'])
        self.gallery_per_cam, self.gallery_per_cam_sampled = reorganize_images_by_camera(self.gallery,
                                                                                         kwargs['num_bn_sample'])

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        fake_pattern = re.compile(r'([-\d]+)_c(\d)s\d_[\d]+_[\d]+_fake_\dto(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            # index_select not supported on CUDAType for Bool
            if_real = 1
            if 'fake' in img_path:
                pid, camid, fake_camid = map(int, fake_pattern.search(img_path).groups())
                camid = fake_camid
                if_real = 0
            else:
                pid, camid = map(int, pattern.search(img_path).groups())
                fake_camid = camid

            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:pid = pid2label[pid]
            dataset.append((img_path, pid+self.pidoffset, camid+self.cidoffset, if_real, fake_camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
