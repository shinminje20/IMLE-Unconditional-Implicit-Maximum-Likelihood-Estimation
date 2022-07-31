# import heapq
# from collections import defaultdict
# from torch.utils.data import Dataset, DataLoader, Subset
from random import sample

class KorKMinusOne:
    def __init__(self, idxs, shuffle=False):
        self.counter = 0
        self.shuffle = shuffle
        self.idxs = idxs
    def pop(self):
        if self.counter == len(self.idxs):
            self.counter = 0
            self.idxs = sample(self.idxs, k=len(self.idxs)) if self.shuffle else self.idxs
        
        result = self.idxs[self.counter]
        self.counter += 1
        return result
        
# min-Heapq that can push back an element into heapq with incremented key by one
# class AutoPushHeap(object):
#    def __init__(self, idx_list, subsample_size):
        
#         self._data = [(0, idx) for idx in enumerate(idx_list)]
#         heapq.heapify(self._data)

#    def pop(self):
#         frequency, idx = heapq.heappop(self._data)
#         heapq.heappush(self._data, (frequency + 1, idx))
        
#         return idx

# class FrequencyDict(object):
#     def __init__(self, idx_list):
#         self._dict = defaultdict(set)
#         self._dict[0] = set(idx_list)

#     def pop(self):
        
#         popping_freq = -1
#         key_to_remove = None
#         for freq in self._dict.keys():
#             if len(self._dict[freq]) == 0:
#                 key_to_remove = freq
#                 continue

#             popping_freq = freq
#             break
        
#         if key_to_remove is not None:
#             self._dict.pop(key_to_remove)
#             key_to_remove = None
        
#         # Pop random index among the set of lowest frequenced indicies
#         popped_idx = self._dict[popping_freq].pop()
#         # Push popped index into the next frequency set (which is popping_freq + 1)
#         self._dict[popping_freq + 1].add(popped_idx)
        
#         return int(popped_idx)



# class SubsampleDataset(Dataset):
#     def __init__(self, data, subsample_size):
#         super(SubsampleDataset, self).__init__()
#         idx_list = [i for i in range(len(data))]

#         self._idx_dict = FreqDict(idx_list)

#         def __len__(self): return len(self.data)

#     def __getitem__(self, idx):
#         idx = self._idx_dict.pop()

#         ys = [y[idx] for y in self.ys]
#         return x[idx], ys[idx]
# 


# def Subsample(data_tr, subsample_size, k):

#     # entire_batch = DataLoader(data, batch_size=len(data), shuffle=False)
#     # subsample_Dataset = SubsampleDataset(entire_batch, subsample_size)
#   index_list = [str(idx) for idx in range(len(data_tr))]
#   freq_dict = FrequencyDict(index_list)

#   for e in range(k):
#     epoch_data = Subset(data_tr, indices=[freq_dict.pop() for _ in range(subsample_size)])

#     yield epoch_data

# def Subsample(data_tr, subsample_size, k):

#     # entire_batch = DataLoader(data, batch_size=len(data), shuffle=False)
#     # subsample_Dataset = SubsampleDataset(entire_batch, subsample_size)
#     index_list = [str(idx) for idx in range(len(data_tr))]
#     freq_dict = FrequencyDict(index_list)

#     for _ in range(k):
#         epoch_data = Subset(data_tr, indices=[freq_dict.pop() for _ in range(subsample_size)])

#     return epoch_data