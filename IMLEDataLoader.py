from torch.utils.data import Dataset, DataLoader, Subset
import queue
from typing import Optional
from itertools import chain
from utils.Utils import *
from random import sample

class IMLEDataLoader(object):
    def __init__(self, dataset, kkm, model, z_gen, loss_fn, num_samples, sample_parallelism, code_bs,
                    total_depth,
                    curr_depth,
                    subsample_size=None,
                    num_iteration=1,
                    pin_memory: bool = False,
                    shuffle: Optional[bool] = None,
                    batch_size: Optional[int] = 1,
                    num_workers: int = 0,
                    drop_last: bool = False):
        self.dataset = dataset
        self.kkm = kkm
        self.model = model
        self.subsample_size = subsample_size if subsample_size is not None else len(self.dataset)
        self.num_iteration = num_iteration
        # self.loader_generate_cycle = (self.num_iteration // self.subsample_size ) + 1 if self.num_iteration > self.subsample_size  else 1
        self.z_gen = z_gen
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.sample_parallelism = sample_parallelism
        self.code_bs = code_bs
        self.shuffle = shuffle
        self.chain_loaders = []
        self.data_len = 0
        self.depth = total_depth
        self.curr_depth = curr_depth
    
    def __iter__(self):
        # (len datset // subsample_size) * num of epochs = total num_samplings
        # --num_samplings


        # num_iteration: # of iteration per samples
        loaders = []
        iter_data = Subset(self.dataset, indices=[self.kkm.pop() for _ in range(self.subsample_size)])

        codes, targets  = get_codes_in_chunks(iter_data, self.model, self.z_gen, self.loss_fn, self.depth, self.curr_depth, num_samples=self.num_samples,
                                    sample_parallelism=self.sample_parallelism, code_bs=self.code_bs)

        codeYdataset = CodeYDataset(codes, targets)  
        self.data_len = 0
        num_chained_loaders = self.num_iteration // len(codeYdataset) + 1
        for i in range(num_chained_loaders):
            if i == num_chained_loaders - 1:
                if self.num_iteration % len(codeYdataset) != 0 and self.num_iteration > len(codeYdataset):
                    Subset_codeYdataset = Subset(codeYdataset, sample(range(len(codeYdataset)), 
                                                    self.num_iteration % len(codeYdataset)))
                    loader = DataLoader(codeYdataset, 
                        pin_memory=self.pin_memory,
                        shuffle=self.shuffle,
                        batch_size=1,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last)
                else:
                    break
            else:
                loader = DataLoader(codeYdataset, 
                        pin_memory=self.pin_memory,
                        shuffle=self.shuffle,
                        batch_size=1,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last)
                
            self.data_len += len(loader)
            loaders.append(loader)
        print("self.data_len: ", self.data_len)
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


class CodeYDataset(Dataset):
    """
    Args:
    cx      -- BSxCxHxW tensor of corrupted images
    codes   -- list of codes of shape BSxCODE_DIM. Elements in the list should
                be codes for sequentially greater resolutions
    ys      -- list of BSxCxHxW target images. Elements in the list should be
                for sequentially greater resolutions
    """
    def __init__(self, codes, ys):
        super(CodeYDataset, self).__init__()
        print("_init__")
        self.codes = [c.cpu() for c in codes]
        self.ys = [y.cpu() for y in ys]

    def __len__(self): return len(self.ys)

    def __getitem__(self, idx):
        return self.codes[idx], self.ys[idx]

def get_new_codes(y, model, z_gen, loss_fn, depth, curr_depth, num_samples=16, sample_parallelism=16):
    """Returns a list of new latent codes found via hierarchical sampling.

    Args:
    cx          -- a BSxCxHxW tensor of corrupted images, on device
    model       -- model backbone. Must support a 'loi' argument and a tensor of
                    losses, one for each element in an input batch
    z_gen       -- function mapping from batch sizes and levels to z_dims
    sp          -- list of sample parallelisms, one for each level
    num_samples -- list of numbers of samples, one for each level
    """
    num_samples = num_samples[0]
    sample_parallelism = sample_parallelism[0]

    bs = len(y)
    code = z_gen(bs)
    with torch.no_grad():
        # for level_idx in tqdm(range(len(num_samples)),
        #     desc="Sampling: levels",
        #     leave=False,
        #     dynamic_ncols=True):

            # Get inputs for sampling for the current level. We need to
            # store the least losses we have for each example, and to find
            # the level-specific number of samples [ns], sample parallelism
            # [sp], and shape to sample noise in [shape].
        least_losses = torch.ones(bs, device=device) * float("inf")
        ns = num_samples
        sp = min(ns, sample_parallelism)

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
            new_codes = z_gen(bs * sp)
            
            
            
            # Compute loss for the new codes.
            outputs = model(new_codes, curr_depth, 1)
            losses = loss_fn(outputs, y, reduction="none")
            print("losses:", losses)
            # [losses] may have multiple values for each input example
            # due to using sample parallelism. Therefore, we find the
            # best-comuted loss for each example, giving a tensor of new
            # losses of the same size as [least_losses]. We do the same
            # with the newly sampled codes.
            _, idxs = torch.min(losses.view(bs, sp), axis=1)
            new_codes = new_codes.view((bs, sp) + new_codes.shape[1:])
            new_codes = new_codes[torch.arange(bs), idxs]
            losses = losses.view(bs, sp)[torch.arange(bs), idxs]

            # Update [codes] and [last_losses] to reflect new
            # codes that get least loss.
            change_idxs = losses < least_losses
            code[change_idxs] = new_codes[change_idxs]
            least_losses = losses

    return code

def get_codes_in_chunks(data, model, z_gen, loss_fn, depth, curr_depth, num_samples=16,
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
    codes = []

    loader = DataLoader(data,
        batch_size=code_bs,
        pin_memory=True,
        num_workers=24,
        drop_last=False)

    targets_images = None
    
    for idx, y in tqdm(enumerate(loader),
        desc="Sampling chunks of batch",
        total=len(loader),
        leave=False,
        dynamic_ncols=True):
        
        y = y.to(device, non_blocking=True)
        down_y = progressive_down_sampling(y, depth, curr_depth, 1)
        code = get_new_codes(down_y, model, z_gen, loss_fn, depth, curr_depth,
            num_samples=num_samples,
            sample_parallelism=sample_parallelism)
         
        codes.append(code)
          
        if targets_images is None:
            targets_images = [down_y.cpu()]
        else:
            targets_images.append(down_y)
    
    # assert 0
    return codes, targets_images

def progressive_down_sampling(real_batch, depth, curr_depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate


        # down_sample the real_batch for the given depth
        # print("depth, curr_depth,: ", depth, curr_depth)
        down_sample_factor = int(np.power(2, depth - curr_depth - 1))
        prior_down_sample_factor = max(int(np.power(2, depth - curr_depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if curr_depth > 3:
            # print(" curr_depth > 3")
            prior_ds_real_samples = interpolate(AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            # print(" curr_depth == 3")
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        # print("ds_real_samples: ", ds_real_samples.shape)
        # print("prior_ds_real_samples: ", prior_ds_real_samples.shape)
 
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples
# def get_new_codes(y, model, z_gen, loss_fn, num_samples=16, sample_parallelism=16):
#     """Returns a list of new latent codes found via hierarchical sampling.

#     Args:
#     cx          -- a BSxCxHxW tensor of corrupted images, on device
#     model       -- model backbone. Must support a 'loi' argument and a tensor of
#                     losses, one for each element in an input batch
#     z_gen       -- function mapping from batch sizes and levels to z_dims
#     sp          -- list of sample parallelisms, one for each level
#     num_samples -- list of numbers of samples, one for each level
#     """
#     num_samples = make_list(num_samples, len(y))
#     sample_parallelism = make_list(sample_parallelism, len(y))

#     bs = len(y[0])
#     level_codes = z_gen(bs, level="all")
#     with torch.no_grad():
#         for level_idx in tqdm(range(len(num_samples)),
#             desc="Sampling: levels",
#             leave=False,
#             dynamic_ncols=True):

#             # Get inputs for sampling for the current level. We need to
#             # store the least losses we have for each example, and to find
#             # the level-specific number of samples [ns], sample parallelism
#             # [sp], and shape to sample noise in [shape].
#             old_codes = level_codes[:level_idx]
#             least_losses = torch.ones(bs, device=device) * float("inf")
#             ns = num_samples[level_idx]
#             sp = min(ns, sample_parallelism[level_idx])

#             # Handle arbitrary sample parallelism. If [sp] evenly divides
#             # [ns], then we just run [ns // sp] tries. Otherwise, we run an
#             # extra try where the sample parallelism is [ns % sp].
#             if ns % sp == 0:
#                 iter_range = range(ns // sp)
#                 sps = make_list(sp, len(iter_range))
#             else:
#                 iter_range = range(ns // sp + 1)
#                 sps = make_list(sp, length=ns // sp) + [ns % sp]

#             for idx in tqdm(iter_range,
#                 desc="Sampling: iterations over level",
#                 leave=False,
#                 dynamic_ncols=True):

#                 # Get the sample parallelism for this trial. Then, get new
#                 # codes to sample for the CAMNet level currently being
#                 # sampled with while using the prior best old codes.
#                 sp = sps[idx]
#                 new_codes = z_gen(bs * sp, level=level_idx)
#                 test_codes = old_codes + [new_codes]

#                 # Compute loss for the new codes.
#                 print("=====================================")
#                 print("len(level_codes[:level_idx]): ", len(level_codes[:level_idx]))
#                 # print("level_codes[:level_idx].shape: ", level_codes[:level_idx][-1].shape)
#                 print("new_codes.shape: ", new_codes.shape)
#                 print("test_codes.shape: ", test_codes[-1].shape)
#                 print("new_codes.shape", new_codes.shape)
#                 # print("ouputs.shape: ", outputs.shape)
#                 print("y[level_idx].shape: ", y[-1].shape)
#                 print("=====================================")
#                 # raise "easdfjklasd"
#                 outputs = model(new_codes)
                
#                 losses = loss_fn(outputs, y[level_idx], 'none')

#                 # [losses] may have multiple values for each input example
#                 # due to using sample parallelism. Therefore, we find the
#                 # best-comuted loss for each example, giving a tensor of new
#                 # losses of the same size as [least_losses]. We do the same
#                 # with the newly sampled codes.
                
#                 # losses = torch.permute(losses, (1, 0 , 2))
#                 # print("=-=-=-=-===-")
#                 # print("outputs.shape", outputs.shape)
#                 # print("losses.shape", losses.shape)
#                 # print("y.shape", y[0].shape, bs, sp)
#                 # print("new_codes.shape", new_codes.shape)
#                 # print("=-=-=-=-===-")

#                 # _, idxs = torch.min(losses.view(bs, sp*bs), axis=1)
#                 # new_codes= new_codes.unsqueeze(0).expand(bs, -1 , -1)
#                 # new_codes = new_codes.view((bs, sp*bs) + new_codes.shape[1:])

#                 _, idxs = torch.min(losses.view(bs, sp), axis=1)
#                 new_codes = new_codes.view((bs, sp) + new_codes.shape[1:])
#                 new_codes = new_codes[torch.arange(bs), idxs]
#                 losses = losses.view(bs, sp)[torch.arange(bs), idxs]

#                 # Update [level_codes] and [last_losses] to reflect new
#                 # codes that get least loss.
#                 change_idxs = losses < least_losses
#                 level_codes[level_idx][change_idxs] = new_codes[change_idxs]
#                 least_losses[change_idxs] = losses[change_idxs]

#     return level_codes

# def get_codes_in_chunks(data, model, z_gen, loss_fn, num_samples=16,
#     sample_parallelism=16, code_bs=128):
#     """Returns a list of new latent codes found via hierarchical sampling with
#     the batch dimension chunked to allow running larger batches.
#     Args:
#     data        -- GeneratorDataset, or Subset thereof
#     model       -- model backbone. Must support a 'loi' argument and a tensor of
#                     losses, one for each element in an input batch
#     z_gen       -- function mapping from batch sizes and levels to z_dims
#     sp          -- list of sample parallelisms, one for each level
#     num_samples -- list of numbers of samples, one for each level
#     code_bs     -- the size of each batch dimension chunk
#     """
#     level_codes = z_gen(len(data), level="all")
#     loader = DataLoader(data,
#         batch_size=code_bs,
#         pin_memory=True,
#         num_workers=24,
#         drop_last=False)

#     targets_images = None
    
#     for idx, (_, ys) in tqdm(enumerate(loader),
#         desc="Sampling chunks of batch",
#         total=len(loader),
#         leave=False,
#         dynamic_ncols=True):
#         print("=============")
#         print("len(ys): ", len(ys))
#         print("ys[-1].shape: ", ys[-1].shape)
#         print("=============")
#         ys = [y.to(device, non_blocking=True) for y in ys]
#         chunk_codes = get_new_codes(ys, model, z_gen, loss_fn,
#             num_samples=num_samples,
#             sample_parallelism=sample_parallelism)
        
#         indices = range(idx * code_bs, min(len(data), (idx+1) * code_bs))
#         indices = torch.tensor(indices)
#         for level_idx in range(len(chunk_codes)):
#             level_codes[level_idx][indices] = chunk_codes[level_idx]
        
#         # Save the resulting images
#         if targets_images is None:
#             targets_images = [[y.cpu()] for y in ys]
#         else:
#             for t,y in zip(targets_images, ys):
#                 t.append(y.cpu())

#     targets_images = [torch.cat(t, axis=0) for t in targets_images]
    
#     return level_codes, targets_images


# def new_get_codes_in_chunks(data, model, z_gen, loss_fn, num_samples=16,
#     sample_parallelism=16, code_bs=128):
#     """Returns a list of new latent codes found via hierarchical sampling with
#     the batch dimension chunked to allow running larger batches.
#     Args:
#     data        -- GeneratorDataset, or Subset thereof
#     model       -- model backbone. Must support a 'loi' argument and a tensor of
#                     losses, one for each element in an input batch
#     z_gen       -- function mapping from batch sizes and levels to z_dims
#     sp          -- list of sample parallelisms, one for each level
#     num_samples -- list of numbers of samples, one for each level
#     code_bs     -- the size of each batch dimension chunk
#     """
#     # level_codes = z_gen(len(data), level="all")
#     loader = DataLoader(data,
#         batch_size=code_bs,
#         pin_memory=True,
#         num_workers=24,
#         drop_last=False)

#     targets_images = None
    
#     for idx, (_, ys) in tqdm(enumerate(loader),
#         desc="Sampling chunks of batch",
#         total=len(loader),
#         leave=False,
#         dynamic_ncols=True):
        
#         ys = [y.to(device, non_blocking=True) for y in ys]
#         # chunk_codes = get_new_codes(ys, model, z_gen, loss_fn,
#         #     num_samples=num_samples,
#         #     sample_parallelism=sample_parallelism)
#         torch.randn((bs,) + z_dims[level])
#         noise = torch.randn(iter_data.shape[0], self.code_bs).to(device)
#         fake_samples = model(noise)
#         # indices = range(idx * code_bs, min(len(data), (idx+1) * code_bs))
#         # indices = torch.tensor(indices)
#         # for level_idx in range(len(chunk_codes)):
#         #     level_codes[level_idx][indices] = chunk_codes[level_idx]
        
#         # Save the resulting images
#         if targets_images is None:
#             targets_images = [[y.cpu()] for y in ys]
#         else:
#             for t,y in zip(targets_images, ys):
#                 t.append(y.cpu())

#     targets_images = [torch.cat(t, axis=0) for t in targets_images]
    
#     return level_codes, targets_images