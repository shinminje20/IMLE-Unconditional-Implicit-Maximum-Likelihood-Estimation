from torch.utils.data import Subset, ConcatDataset, DataLoader
from CAMNet import *
from Corruptions import Corruption
from Data import *
from Losses import *

# ###################################### DataLoader ######################################
# def GenericDataLoader(model, loader_tr, cx, ys, z_gen, loss_fn, corruptor, mini_bs, num_samples=16, sample_parallelism=16, code_bs=128, expand_factor=1):

#     GenericDataLoader.counter += 1
    
#     loader = DataLoader(gdataset, batch_size=mini_bs, shuffle=True, num_workers=8)

#     return loader

class CorruptedXDataset_new_code(Dataset):
    """
    Args:
    cx      -- BSxCxHxW tensor of corrupted images
    codes   -- list of codes of shape BSxCODE_DIM. Elements in the list should
                be codes for sequentially greater resolutions
    ys      -- list of BSxCxHxW target images. Elements in the list should be
                for sequentially greater resolutions
    """
    def __init__(self, data, model, z_gen, loss_fn, num_samples, sample_parallelism, code_bs, expand_factor=1):
        super(CorruptedXDataset_new_code, self).__init__()
        
        self.codes = []
        self.__sampling__(data, model, z_gen, loss_fn, num_samples, sample_parallelism, code_bs)

        self.cx = self.cx.cpu()
        self.codes = [c.cpu() for c in self.codes]
        self.ys = [y.cpu() for y in self.ys]
        self.expand_factor = self.expand_factor
    
    def __len__(self): return len(self.cx) * self.expand_factor

    def __getitem__(self, idx):
        idx = idx // self.expand_factor
        cx = self.cx[idx]
        codes = [c[idx] for c in self.codes]
        ys = [y[idx] for y in self.ys]
        return cx, codes, ys
    
    def __sampling__(self, data, model, z_gen, loss_fn, num_samples, sample_parallelism, code_bs):
        print("================================== Call __sampling__() ==================================")
        
        x, ys = data
        ys = [y.to(device, non_blocking=True) for y in ys]
        cx = corruptor(x.to(device, non_blocking=True))
        self.codes = get_codes_in_chunks(cx, ys, model, z_gen, loss_fn,
            num_samples=num_samples,
            sample_parallelism=sample_parallelism,
            code_bs=code_bs)

        self.cx = self.cx.cpu()
        self.codes = [c.cpu() for c in self.codes]
        self.ys = [y.cpu() for y in self.ys]

# ###################################### Dataset ######################################
# class DynamicDataset(Dataset):
#     def __init__(self, model, cx, ys, z_gen, loss_fn, num_samples, sample_parallelism, code_bs, expand_factor, k, subsample_epoch_size):
#         super(GenericDataset, self).__init__()
#         self.model = model
#         self.z_gen = z_gen
#         self.loss_fn = loss_fn
#         self.num_samples = num_samples
#         self.sample_parallelism = sample_parallelism
#         self.code_bs = code_bs
#         self.expand_factor = expand_factor
#         self.cx = cx.cpu()
#         self.ys = [y.cpu() for y in ys]
#         self.codes = get_codes_in_chunks(self.cx, self.ys, self.model, self.z_gen, self.loss_fn,
#                                         num_samples=self.num_samples,
#                                         sample_parallelism=self.sample_parallelism,
#                                         code_bs=self.code_bs)
#         self.batch_iter = 0

#     def __len__(self): return len(self.data)

#     def __getitem__(self, idx):

#         self.batch_iter  += 1

#         if self.batch_iter  % self.k * len(self.cx) == 0: #maybe self.k * self.code_bs
#             print("////////////////////////// Re-sampling //////////////////////////")
#             self.codes = get_codes_in_chunks(cx, ys, self.model, self.z_gen, self.loss_fn,
#                                         num_samples=self.num_samples,
#                                         sample_parallelism=self.sample_parallelism,
#                                         code_bs=self.code_bs)
#         # Handler concurrency
#         # self._num_yielded += 1
#         # prefetch
#         # self._next_data()
#         idx = idx // self.expand_factor
#         cx = cx[idx]
#         codes = [c[idx] for c in codes]
#         ys = [y[idx] for y in ys]

#         return cx, codes, ys

################################## Helper Functions ##################################
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

def get_codes_in_chunks(cx, y, model, z_gen, loss_fn, num_samples=16,
    sample_parallelism=16, code_bs=128):
    """Returns a list of new latent codes found via hierarchical sampling with
    the batch dimension chunked to allow running larger batches.

    Args:
    cx          -- a BSxCxHxW tensor of corrupted images, on device
    model       -- model backbone. Must support a 'loi' argument and a tensor of
                    losses, one for each element in an input batch
    z_gen       -- function mapping from batch sizes and levels to z_dims
    sp          -- list of sample parallelisms, one for each level
    num_samples -- list of numbers of samples, one for each level
    code_bs     -- the size of each batch dimension chunk
    """
    def partition_into_batches(x, chunks):
        """Returns [x] split into [chunks] sections along each constituent
        tensor's zero dimension.
        """
        if isinstance(x, (list, tuple)):
            return [partition_into_batches(x_, chunks) for x_ in x]
        elif isinstance(x, torch.Tensor):
            return torch.tensor_split(x, chunks)
        else:
            raise ValueError()

    chunks = max(1, len(cx) // code_bs)
    cx = partition_into_batches(cx, chunks)
    y = partition_into_batches(y, chunks)

    level_codes = level_codes = z_gen(0, level="all")
    for cx_ys in tqdm(zip(cx, *y),
        total=chunks,
        desc="Sampling: chunks",
        leave=False,
        dynamic_ncols=True):

        chunk_codes = get_new_codes(cx_ys[0], cx_ys[1:], model, z_gen, loss_fn,
            num_samples=num_samples,
            sample_parallelism=sample_parallelism)
        level_codes = [torch.cat(c) for c in zip(level_codes, chunk_codes)]
    
    return level_codes