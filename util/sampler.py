import math
import torch
from torch.utils.data.sampler import RandomSampler
import random
import numpy as np

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([cur_dataset.__len__() for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)

class ComboBatchSampler(torch.utils.data.sampler.Sampler):
    
    def __init__(self, samplers, batch_size, drop_last):
        
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        self.iterators = [iter(sampler) for sampler in self.samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_idxs = []
        self.sampler_idxs = []
        
        for i,sampler in enumerate(self.samplers):
            self.sampler_idxs.extend(self.list_chunk([i]*len(sampler),batch_size))
            self.dataset_idxs.extend(self.list_chunk([s_idx for s_idx in sampler],batch_size))
        assert len(self.sampler_idxs)==len(self.dataset_idxs)
        idxs = list(range(len(self.sampler_idxs)))
        random.shuffle(idxs)
        self.sampler_idxs = list(np.array(self.sampler_idxs, dtype=object)[idxs])
        self.dataset_idxs = list(np.array(self.dataset_idxs, dtype=object)[idxs])

    def list_chunk(self,lst, n):
        return [lst[i:i+n] for i in range(0, len(lst), n)]

    def set_epoch(self):
        random.seed(42)
        for i in self.samplers:
            self.dataset_idxs.extend(list(range(len(self.samplers))))
        random.shuffle(self.dataset_idxs)

    def __iter__(self):
        
        # define how many batches we will grab
        self.min_batches = min([len(sampler) for sampler in self.samplers])
        self.n_batches = self.min_batches * len(self.samplers)
        
        # define which indicies to use for each batch
        # self.dataset_idxs = []
        # random.seed(42)
        # for j in range((self.n_batches//len(self.samplers) + 1)):
        #     loader_inds = list(range(len(self.samplers)))
        #     random.shuffle(loader_inds)
        #     self.dataset_idxs.extend(loader_inds)
        # self.dataset_idxs = self.dataset_idxs[:self.n_batches]

        
        # return the batch indicies
        # batch = []
        # sampler_idx = [[i for i in s] for s in self.samplers]
        # for dataset_idx in self.dataset_idxs:
        #     for idx in sampler_idx[dataset_idx]:
        #         batch.append((dataset_idx, idx))
        #         if len(batch) == self.batch_size:
        #             yield (batch)
        #             batch = []
        #             break
        #     if len(batch) > 0 and not self.drop_last:
        #         yield batch
        # for i,s in enumerate(self.samplers):
        #     print('{}_length:{}'.format(i,len(s)))
        batch = []
        idx_in_batch=0
        # real_idx = [[i for i in s] for s in self.samplers]
        for s_idx,d_idx in zip(self.sampler_idxs,self.dataset_idxs):
        # for idx in self.samplers[dataset_idx]:
            # print((s_idx,d_idx))
            # s_id = s_idx.item()
            for s_id, d_id in zip(s_idx,d_idx):
                # batch.append((s_id,real_idx[s_id][d_id]
                batch.append((s_id,d_id))
                # batch.append((0,0))
                idx_in_batch+=1
                if idx_in_batch ==self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = []
                    # break
            if idx_in_batch >0 and not self.drop_last:
                yield batch
            idx_in_batch=0
            batch=[]

    def __len__(self) -> int:
        if self.drop_last:
            return (sum([len(sampler) for sampler in self.samplers])) // self.batch_size
        else:
            return (sum([len(sampler) for sampler in self.samplers]) + self.batch_size - 1) // self.batch_size