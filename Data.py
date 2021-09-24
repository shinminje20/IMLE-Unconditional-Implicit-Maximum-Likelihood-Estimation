from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from torchvision.datasets import CIFAR10
from torchvision import transforms

dataset2input_dim = {
    "cifar10": (3, 32, 32)
}

dataset2n_classes = {
    "cifar10": 10,
}

grayscale_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])

no_transform = transforms.Compose([
    transforms.ToTensor()])

################################################################################
# Non-realistic augmentations. These represent an important baseline to beat.
################################################################################
cifar_augs_tr = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])

cifar10_augs_te = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])

################################################################################
# Datasets
################################################################################
class ImagesFromTransformsDataset(Dataset):
    """Wraps an internal dataset for which queries produce a tuple in which an
    image is the first result. This dataset computes transformations [x] and
    [y]; the object for a neural net using the generated data is to generate [y]
    conditioned on [x] and a random vector z, or to contrast [x] and [y].

    [x] might be a downsampled and greyscaled version of the image, while [y]
    might be a random crop. This trains a generator to simulteneously do
    super-resolution, colorization, and cropping.
    """

    def __init__(self, data, x_transform, y_transform):
        """
        Args:
        data        -- the wrapped dataset, should return tuples wherein the
                        first element is an image
        x_transform -- the transform giving input images
        y_transform -- the transform giving targets
        """
        super(ImagesFromTransformsDataset, self).__init__()
        self.data = data
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        return self.x_transform(image), self.y_transform(image)

class WithZDataset(Dataset):
    """A dataset where data returned from the __getitem__() method has a random
    vector [z] included at the end.
    """

    def __init__(self, data):
        """
        Args:
        data    -- the wrapped dataset
        z_dim   -- the dimensionality of the random vector to return
        """
        super(ImageImageZDataset, self).__init__()
        self.data = data
        self.z_dim = z_dim

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if isinstance(data, tuple):
            return data + torch.rand(self.z_dim)
        else:
            return (data, torch.rand(self.z_dim))

class FeatureDataset(Dataset):
    """A dataset of model features."""

    def __init__(self, F, data):
        """Args:
        F       -- a feature extractor
        data    -- a dataset of XY pairs
        """
        super(FeatureDataset, self).__init__()
        loader = DataLoader(data, batch_size=64)

        data_x, data_y
        F = F.to(device)
        F.eval()
        with torch.no_grad():
            for x,y in tqmd(loader, desc="Building validation dataset", leave=False, file=sys.stdout):
                data_x.append(F(x.to(device).cpu()))
                data_y.append(y)

        self.data = zip(torch.cat(data_x, 0), torch.cat(data_y), 0)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]
