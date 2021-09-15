import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

dataset2input_dim = {
    "cifar10": (3, 28, 28)
}

class IMLEDataset(Dataset):
    """A dataset wrapping [data]; appends a [z_dim]-dimensional uniform
    random on [-1, 1] vector to the result of [data.__getitem__()].
    """

    def __init__(self, data, transform):
        """
        Args:
        data        -- the wrapped dataset, should return tuples wherein the
                        first element is an image
        z_dim       -- the dimensionality of the random vector to add
        transform   -- transform to apply to output data, eg. ToTensor()

        """
        super(IMLEDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx][0]), torch.rand(self.z_dim)
