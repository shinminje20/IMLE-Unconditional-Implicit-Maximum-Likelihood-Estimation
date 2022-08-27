from KorKMinusOne import KorKMinusOne
from torch.utils.data import Dataset, DataLoader, Subset
import queue
from typing import Optional
from itertools import chain
from utils.Utils import *
from random import sample

class CIMLEDataLoader(object):
    def __init__(self, dataset, kkm, model, corruptor, z_gen, loss_fn, num_samples, sample_parallelism, code_bs,
                    subsample_size=None,
                    num_iteration=1,
                    pin_memory: bool = False,
                    shuffle: Optional[bool] = None,
                    batch_size: Optional[int] = 1,
                    num_workers: int = 0,
                    drop_last: bool = False):
        self.dataset = dataset
        self.kkm = kkm
        self.subsample_size = subsample_size if subsample_size is not None else len(self.dataset)
        self.num_iteration = num_iteration
        self.loader_generate_cycle = (self.num_iteration // (self.subsample_size // batch_size)) + 1 if self.num_iteration > (self.subsample_size // batch_size) else 1
        self.model = model
        self.z_gen = z_gen
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.sample_parallelism = sample_parallelism
        self.code_bs = code_bs
        self.corruptor = corruptor
        self.shuffle = shuffle
        self.chain_loaders = []
        self.data_len = 0
    def __iter__(self):
        # (len datset // subsample_size) * num of epochs = total num_samplings
        # --num_samplings


        # num_iteration: # of iteration per samples
        loaders = []
        iter_data = Subset(self.dataset, indices=[self.kkm.pop() for _ in range(self.subsample_size)])

        codes, corrupted, targets  = get_codes_in_chunks(iter_data, self.model, self.corruptor, self.z_gen, self.loss_fn, num_samples=self.num_samples,
                                    sample_parallelism=self.sample_parallelism, code_bs=self.code_bs)
        corrupted_dataset = CorruptedCodeYDataset(corrupted, codes, targets)       
            
        self.data_len = 0
        for i in range(self.loader_generate_cycle):
            if i == self.loader_generate_cycle - 1:
                if self.num_iteration % (self.subsample_size // self.batch_size) != 0 and self.num_iteration > self.subsample_size:
                    Subset_corrupted_dataset = Subset(corrupted_dataset, sample(range(len(corrupted_dataset)),
                                                     self.num_iteration % (self.subsample_size // self.batch_size)))
                    loader = DataLoader(Subset_corrupted_dataset, 
                        pin_memory=self.pin_memory,
                        shuffle=self.shuffle,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last)
                else:
                    break
            else:
                loader = DataLoader(corrupted_dataset, 
                        pin_memory=self.pin_memory,
                        shuffle=self.shuffle,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last)

            self.data_len += len(loader)
            loaders.append(loader)
        self.chain_loaders = chain(*loaders) 

        return self

    def __next__(self):
        try:    
            return next(self.chain_loaders)
        except IndexError:
            self.data_len = 0
            raise StopIteration

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return self.data_len
        
        

class CorruptedCodeYDataset(Dataset):
    """
    Args:
    cx      -- BSxCxHxW tensor of corrupted images
    codes   -- list of codes of shape BSxCODE_DIM. Elements in the list should
                be codes for sequentially greater resolutions
    ys      -- list of BSxCxHxW target images. Elements in the list should be
                for sequentially greater resolutions
    """
    def __init__(self, cx, codes, ys, expand_factor=1):
        super(CorruptedCodeYDataset, self).__init__()
        assert len(codes) == len(ys)
        assert all([len(c) == len(y) == cx.shape[0] for c,y in zip(codes, ys)])
        self.cx = cx.cpu()
        self.codes = [c.cpu() for c in codes]
        self.ys = [y.cpu() for y in ys]
        self.expand_factor = expand_factor

    def __len__(self): return len(self.cx) * self.expand_factor

    def __getitem__(self, idx):
        idx = idx // self.expand_factor
        cx = self.cx[idx]
        codes = [c[idx] for c in self.codes]
        ys = [y[idx] for y in self.ys]
        return cx, codes, ys

def get_new_codes(cx, y, model, z_gen, loss_fn, num_samples=16, sample_parallelism=16):
    """Returns a list of new latent codes found via hierarchical sampling.

    Args:
    cx          -- a BSxCxHxW tensor of corrupted images, on device
    model       -- model backbone. Must support a 'loi' argument and a tensor of
                    losses, one for each element in an input batch
    z_gen       -- function mapping from batch sizes and levels to z_dims
    sp          -- list of sample parallelisms, one for each level
    num_samples -- list of numbers of samples, one for each level
    """
    num_samples = make_list(num_samples, len(y))
    sample_parallelism = make_list(sample_parallelism, len(y))

    bs = len(cx)
    level_codes = z_gen(bs, level="all")
    with torch.no_grad():
        for level_idx in tqdm(range(len(num_samples)),
            desc="Sampling: levels",
            leave=False,
            dynamic_ncols=True):

            # Get inputs for sampling for the current level. We need to
            # store the least losses we have for each example, and to find
            # the level-specific number of samples [ns], sample parallelism
            # [sp], and shape to sample noise in [shape].
            old_codes = level_codes[:level_idx]
            least_losses = torch.ones(bs, device=device) * float("inf")
            ns = num_samples[level_idx]
            sp = min(ns, sample_parallelism[level_idx])

            # Handle arbitrary sample parallelism. If [sp] evenly divides
            # [ns], then we just run [ns // sp] tries. Otherwise, we run an
            # extra try where the sample parallelism is [ns % sp].
            if ns % sp == 0:
                iter_range = range(ns // sp)
                sps = make_list(sp, len(iter_range))
            else:
                iter_range = range(ns // sp + 1)
                sps = make_list(sp, length=ns // sp) + [ns % sp]

            for idx in tqdm(iter_range,
                desc="Sampling: iterations over level",
                leave=False,
                dynamic_ncols=True):

                # Get the sample parallelism for this trial. Then, get new
                # codes to sample for the CAMNet level currently being
                # sampled with while using the prior best old codes.
                sp = sps[idx]
                new_codes = z_gen(bs * sp, level=level_idx)
                test_codes = old_codes + [new_codes]
                

                # Compute loss for the new codes.
                outputs = model(cx, test_codes, loi=level_idx)
                print("=====================================")
                print("len(level_codes[:level_idx]): ", len(level_codes[:level_idx]))
                # print("level_codes[:level_idx].shape: ", level_codes[:level_idx][-1].shape)
                print("new_codes.shape: ", new_codes.shape)
                print("test_codes.shape: ", test_codes[-1].shape)
                print("new_codes.shape", new_codes.shape)
                print("ouputs.shape: ", outputs.shape)
                print("y[level_idx].shape: ", y[-1].shape)
                print("=====================================")
                losses = loss_fn(outputs, y[level_idx])

                # [losses] may have multiple values for each input example
                # due to using sample parallelism. Therefore, we find the
                # best-comuted loss for each example, giving a tensor of new
                # losses of the same size as [least_losses]. We do the same
                # with the newly sampled codes.
                _, idxs = torch.min(losses.view(bs, sp), axis=1)
                new_codes = new_codes.view((bs, sp) + new_codes.shape[1:])
                new_codes = new_codes[torch.arange(bs), idxs]
                losses = losses.view(bs, sp)[torch.arange(bs), idxs]

                # Update [level_codes] and [last_losses] to reflect new
                # codes that get least loss.
                change_idxs = losses < least_losses
                level_codes[level_idx][change_idxs] = new_codes[change_idxs]
                least_losses[change_idxs] = losses[change_idxs]

    return level_codes

def get_codes_in_chunks(data, model, corruptor, z_gen, loss_fn, num_samples=16,
    sample_parallelism=16, code_bs=128):
    """Returns a list of new latent codes found via hierarchical sampling with
    the batch dimension chunked to allow running larger batches.
    Args:
    data        -- GeneratorDataset, or Subset thereof
    model       -- model backbone. Must support a 'loi' argument and a tensor of
                    losses, one for each element in an input batch
    z_gen       -- function mapping from batch sizes and levels to z_dims
    sp          -- list of sample parallelisms, one for each level
    num_samples -- list of numbers of samples, one for each level
    code_bs     -- the size of each batch dimension chunk
    """
    level_codes = z_gen(len(data), level="all")
    loader = DataLoader(data,
        batch_size=code_bs,
        pin_memory=True,
        num_workers=24,
        drop_last=False)

    corrupted_images = []
    targets_images = None
    
    for idx,(x,ys) in tqdm(enumerate(loader),
        desc="Sampling chunks of batch",
        total=len(loader),
        leave=False,
        dynamic_ncols=True):
        ys = [y.to(device, non_blocking=True) for y in ys]
        cx = corruptor(x.to(device, non_blocking=True))
        chunk_codes = get_new_codes(cx, ys, model, z_gen, loss_fn,
            num_samples=num_samples,
            sample_parallelism=sample_parallelism)
        
        indices = range(idx * code_bs, min(len(data), (idx+1) * code_bs))
        indices = torch.tensor(indices)
        for level_idx in range(len(chunk_codes)):
            level_codes[level_idx][indices] = chunk_codes[level_idx]
        
        # Save the resulting images
        corrupted_images.append(cx.cpu())
        if targets_images is None:
            targets_images = [[y.cpu()] for y in ys]
        else:
            for t,y in zip(targets_images, ys):
                t.append(y.cpu())

    corrupted_images = torch.cat(corrupted_images, axis=0)
    targets_images = [torch.cat(t, axis=0) for t in targets_images]
    
    return level_codes, corrupted_images, targets_images