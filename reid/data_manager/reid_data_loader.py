from PIL import Image
from torch.utils.data import Dataset


class ReID_Data_Loader(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        # print(self.dataset[item])
        # print(len(self.dataset[item]))
        img_path, pid, camid = self.dataset[item]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)