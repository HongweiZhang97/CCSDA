from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
from torch.utils.data import Dataset
from .cuhk03 import CUHK03
from .market1501 import Market1501
from .market_sct import MarketSCT
from .market_sct_tran import MarketSCTTran
from .duke_sct import DukeSCT
from .duke_sct_tran import DukeSCTTran
from .raw_data_loader import RawImageData


class ReID_Data(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img_path, pid, camid, if_real, fake_camid = self.dataset[item]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, if_real, fake_camid

    def __len__(self):
        return len(self.dataset)

__imgreid_factory = {
	'cuhk03': CUHK03,
	'market1501': Market1501,
	'market_sct': MarketSCT,
	'market_sct_tran': MarketSCTTran,
    'duke_sct': DukeSCT,
    'duke_sct_tran': DukeSCTTran
}

__vidreid_factory = {}

__folder_factory = {
    'market_sct': ReID_Data,
    'market_sct_tran': ReID_Data,
    'duke_sct': ReID_Data,
    'duke_sct_tran': ReID_Data
}


def get_names():
	return list(__imgreid_factory.keys()) + list(__vidreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
	if name not in list(__imgreid_factory.keys()):
		raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
	return __imgreid_factory[name](**kwargs)



def init_datafolder(name, data_list, transforms):
	if name not in __folder_factory.keys():
		raise KeyError("Unknown datasets: {}".format(name))
	return __folder_factory[name](data_list, transforms)

