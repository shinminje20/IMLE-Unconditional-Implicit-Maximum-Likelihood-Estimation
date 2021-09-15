import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

dataset2input_dim = {
    "cifar10": (3, 32, 32)
}

class IMLEDataset(Dataset):
    """A dataset wrapping [data]; appends a [z_dim]-dimensional uniform
    random on [-1, 1] vector to the result of [data.__getitem__()].
    """

    def __init__(self, data, z_dim):
        """
        Args:
        data        -- the wrapped dataset, should return tuples wherein the
                        first element is an image
        z_dim       -- the dimensionality of the random vector to add
        """
        super(IMLEDataset, self).__init__()
        self.data = data
        self.z_dim = z_dim

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx][0], torch.rand(self.z_dim)
