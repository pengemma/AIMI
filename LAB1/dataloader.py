import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from torch.utils import data
import torchvision.transforms as trans
import os

def getData(mode, netnum=None):
    df = pd.read_csv(f'{mode}.csv')
    path = df['path'].tolist()
    label = df['label'].tolist()
    return path, label

class ChestLoader(data.Dataset):
    def __init__(self, root, mode, netnum=None):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.mode = mode
        self.netnum = netnum

        self.img_name, self.label = getData(mode)

        if self.mode == 'train':
            self.transformations = trans.Compose([
                                                  trans.Resize((256, 256)),
                                                  trans.CenterCrop((224, 224)),
                                                  trans.RandomHorizontalFlip(),
                                                  trans.RandomRotation(20),
                                                  trans.ToTensor(),
                                                  trans.Normalize(mean=[0.5], std=[0.5])])
        else:
            self.transformations = trans.Compose([
                                                  trans.Resize((256, 256)),
                                                  trans.CenterCrop((224, 224)),
                                                  trans.ToTensor(),
                                                  trans.Normalize(mean=[0.5], std=[0.5])])
                                                  #trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):

        path = os.path.join(self.root, self.img_name[index])
        img = Image.open(path).convert('L')

        box = img.getbbox()
        region = img.crop((box[0], box[1], box[2], box[3]))

        # region = region.convert('HSV')
        # img = region.filter(ImageFilter.EDGE_ENHANCE)
        img = ImageOps.equalize(region)

        img = self.transformations(img)

        label = self.label[index]
        return img, label

    def get_labels(self):
        return self.label